#include "unit_starpu_task_runner_support.hpp"

#include <dlfcn.h>

#include <array>
#include <chrono>
#include <cstring>
#include <format>
#include <fstream>
#include <iterator>
#include <system_error>

#include "starpu_task_worker/result_dispatcher_component.hpp"

namespace {

using CudaStreamSynchronizeFn = cudaError_t (*)(cudaStream_t);

using CudaStreamCreateWithFlagsFn =
    cudaError_t (*)(cudaStream_t*, unsigned int);

using CudaMemcpyAsyncFn = cudaError_t (*)(
    void*, const void*, std::size_t, cudaMemcpyKind, cudaStream_t);

auto
resolve_real_cuda_stream_synchronize() -> CudaStreamSynchronizeFn
{
  static CudaStreamSynchronizeFn fn = [] {
    void* symbol = dlsym(RTLD_NEXT, "cudaStreamSynchronize");
    if (symbol == nullptr) {
      symbol = dlsym(RTLD_DEFAULT, "cudaStreamSynchronize");
    }
    return reinterpret_cast<CudaStreamSynchronizeFn>(symbol);
  }();
  return fn;
}

auto
resolve_real_cuda_stream_create_with_flags() -> CudaStreamCreateWithFlagsFn
{
  static CudaStreamCreateWithFlagsFn fn = [] {
    void* symbol = dlsym(RTLD_NEXT, "cudaStreamCreateWithFlags");
    if (symbol == nullptr) {
      symbol = dlsym(RTLD_DEFAULT, "cudaStreamCreateWithFlags");
    }
    return reinterpret_cast<CudaStreamCreateWithFlagsFn>(symbol);
  }();
  return fn;
}

auto
resolve_real_cuda_memcpy_async() -> CudaMemcpyAsyncFn
{
  static CudaMemcpyAsyncFn fn = [] {
    void* symbol = dlsym(RTLD_NEXT, "cudaMemcpyAsync");
    if (symbol == nullptr) {
      symbol = dlsym(RTLD_DEFAULT, "cudaMemcpyAsync");
    }
    return reinterpret_cast<CudaMemcpyAsyncFn>(symbol);
  }();
  return fn;
}

std::atomic<bool> g_force_cuda_stream_sync_failure = false;
std::atomic<void*> g_cuda_stream_sync_failure_target = nullptr;
std::atomic<int> g_cuda_stream_sync_failure_count = 0;

std::atomic<bool> g_force_cuda_stream_create_failure = false;
std::atomic<int> g_cuda_stream_create_failure_count = 0;

std::atomic<bool> g_force_cuda_memcpy_failure = false;
std::atomic<void*> g_cuda_memcpy_failure_stream = nullptr;
std::atomic<int> g_cuda_memcpy_failure_count = 0;

SlotHandleLeaseAcquireContext* g_slot_handle_lease_acquire_ctx = nullptr;
SlotHandleLeaseReleaseContext* g_slot_handle_lease_release_ctx = nullptr;

auto
slot_handle_lease_acquire_callback(
    starpu_data_handle_t handle, starpu_data_access_mode) -> int
{
  if (g_slot_handle_lease_acquire_ctx != nullptr &&
      g_slot_handle_lease_acquire_ctx->calls != nullptr) {
    g_slot_handle_lease_acquire_ctx->calls->push_back(handle);
    if (g_slot_handle_lease_acquire_ctx->fail_handle.has_value() &&
        handle == g_slot_handle_lease_acquire_ctx->fail_handle.value()) {
      return g_slot_handle_lease_acquire_ctx->fail_code;
    }
  }
  return 0;
}

void
slot_handle_lease_release_callback(starpu_data_handle_t handle)
{
  if (g_slot_handle_lease_release_ctx != nullptr &&
      g_slot_handle_lease_release_ctx->calls != nullptr) {
    g_slot_handle_lease_release_ctx->calls->push_back(handle);
  }
}

auto
trace_summary_path(const std::filesystem::path& trace_path)
    -> std::filesystem::path
{
  auto summary = trace_path;
  auto stem = summary.stem().string();
  if (stem.empty()) {
    stem = "batching_trace";
  }
  summary.replace_filename(stem + std::string{"_summary.csv"});
  return summary;
}

}  // namespace

ScopedCudaStreamSyncFailure::ScopedCudaStreamSyncFailure(cudaStream_t target)
{
  g_cuda_stream_sync_failure_count.store(0, std::memory_order_relaxed);
  g_cuda_stream_sync_failure_target.store(
      reinterpret_cast<void*>(target), std::memory_order_release);
  g_force_cuda_stream_sync_failure.store(true, std::memory_order_release);
}

ScopedCudaStreamSyncFailure::~ScopedCudaStreamSyncFailure()
{
  g_force_cuda_stream_sync_failure.store(false, std::memory_order_release);
  g_cuda_stream_sync_failure_target.store(nullptr, std::memory_order_release);
}

auto
cuda_stream_sync_failure_count() -> int
{
  return g_cuda_stream_sync_failure_count.load(std::memory_order_relaxed);
}

ScopedCudaStreamCreateFailure::ScopedCudaStreamCreateFailure()
{
  g_cuda_stream_create_failure_count.store(0, std::memory_order_relaxed);
  g_force_cuda_stream_create_failure.store(true, std::memory_order_release);
}

ScopedCudaStreamCreateFailure::~ScopedCudaStreamCreateFailure()
{
  g_force_cuda_stream_create_failure.store(false, std::memory_order_release);
}

auto
cuda_stream_create_failure_count() -> int
{
  return g_cuda_stream_create_failure_count.load(std::memory_order_relaxed);
}

ScopedCudaMemcpyAsyncFailure::ScopedCudaMemcpyAsyncFailure(cudaStream_t target)
{
  g_cuda_memcpy_failure_count.store(0, std::memory_order_relaxed);
  g_cuda_memcpy_failure_stream.store(
      reinterpret_cast<void*>(target), std::memory_order_release);
  g_force_cuda_memcpy_failure.store(true, std::memory_order_release);
}

ScopedCudaMemcpyAsyncFailure::~ScopedCudaMemcpyAsyncFailure()
{
  g_force_cuda_memcpy_failure.store(false, std::memory_order_release);
  g_cuda_memcpy_failure_stream.store(nullptr, std::memory_order_release);
}

auto
cuda_memcpy_failure_count() -> int
{
  return g_cuda_memcpy_failure_count.load(std::memory_order_relaxed);
}

ScopedSlotHandleLeaseAcquireContext::ScopedSlotHandleLeaseAcquireContext(
    SlotHandleLeaseAcquireContext& ctx)
    : previous(g_slot_handle_lease_acquire_ctx), current(&ctx),
      guard(&slot_handle_lease_acquire_callback)
{
  g_slot_handle_lease_acquire_ctx = current;
}

ScopedSlotHandleLeaseAcquireContext::~ScopedSlotHandleLeaseAcquireContext()
{
  g_slot_handle_lease_acquire_ctx = previous;
}

ScopedSlotHandleLeaseReleaseContext::ScopedSlotHandleLeaseReleaseContext(
    SlotHandleLeaseReleaseContext& ctx)
    : previous(g_slot_handle_lease_release_ctx), current(&ctx),
      guard(&slot_handle_lease_release_callback)
{
  g_slot_handle_lease_release_ctx = current;
}

ScopedSlotHandleLeaseReleaseContext::~ScopedSlotHandleLeaseReleaseContext()
{
  g_slot_handle_lease_release_ctx = previous;
}

TraceLoggerSession::TraceLoggerSession()
    : path_(
          std::filesystem::temp_directory_path() /
          std::format(
              "trace_request_ids_{}.json",
              std::chrono::steady_clock::now().time_since_epoch().count()))
{
  starpu_server::BatchingTraceLogger::instance().configure(
      true, path_.string());
}

TraceLoggerSession::~TraceLoggerSession()
{
  close();
  std::error_code ec;
  std::filesystem::remove(path_, ec);
  std::filesystem::remove(trace_summary_path(path_), ec);
}

void
TraceLoggerSession::close()
{
  if (closed_) {
    return;
  }
  starpu_server::BatchingTraceLogger::instance().configure(false, "");
  closed_ = true;
}

[[nodiscard]] auto
TraceLoggerSession::path() const -> const std::filesystem::path&
{
  return path_;
}

auto
read_trace_file(const std::filesystem::path& path) -> std::string
{
  std::ifstream stream(path);
  if (!stream.is_open()) {
    return {};
  }
  return std::string(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
}

void
populate_trace_timing(starpu_server::InferenceJob& job)
{
  using clock = starpu_server::MonotonicClock;
  const auto now = clock::now();
  job.timing_info().enqueued_time = now - std::chrono::milliseconds(3);
  job.timing_info().last_enqueued_time = now - std::chrono::milliseconds(2);
  job.timing_info().batch_collect_start_time =
      now - std::chrono::milliseconds(1);
  job.timing_info().batch_collect_end_time = now;
}

auto
make_aggregated_sub_job(
    const std::shared_ptr<starpu_server::InferenceJob>& job,
    int request_id) -> starpu_server::InferenceJob::AggregatedSubJob
{
  starpu_server::InferenceJob::AggregatedSubJob aggregated{};
  aggregated.job = job;
  aggregated.request_id = request_id;
  aggregated.batch_size = 1;
  if (job) {
    aggregated.arrival_time = job->timing_info().enqueued_time;
  }
  return aggregated;
}

auto
snapshot_vector_interfaces(starpu_data_handle_t handle)
    -> std::vector<VectorInterfaceSnapshot>
{
  std::vector<VectorInterfaceSnapshot> snapshots;
  const unsigned memory_nodes = starpu_memory_nodes_get_count();
  snapshots.reserve(memory_nodes);
  for (unsigned node = 0; node < memory_nodes; ++node) {
    auto* raw_interface = starpu_data_get_interface_on_node(handle, node);
    if (raw_interface == nullptr) {
      continue;
    }
    auto* vector_interface =
        static_cast<starpu_vector_interface*>(raw_interface);
    snapshots.push_back(VectorInterfaceSnapshot{
        vector_interface, static_cast<std::size_t>(vector_interface->elemsize),
        vector_interface->allocsize, vector_interface->nx});
  }
  return snapshots;
}

void
restore_vector_interfaces(const std::vector<VectorInterfaceSnapshot>& snapshots)
{
  for (const auto& snapshot : snapshots) {
    if (snapshot.iface == nullptr) {
      continue;
    }
    snapshot.iface->elemsize =
        static_cast<decltype(snapshot.iface->elemsize)>(snapshot.elemsize);
    snapshot.iface->allocsize = snapshot.allocsize;
    snapshot.iface->nx = static_cast<decltype(snapshot.iface->nx)>(snapshot.nx);
  }
}

std::atomic<int> submit_override_calls = 0;

std::atomic<int> missing_interface_override_hits = 0;
starpu_data_handle_t missing_interface_override_handle = nullptr;

auto
missing_interface_override(starpu_data_handle_t handle, unsigned node) -> void*
{
  if (handle != missing_interface_override_handle) {
    return starpu_test::call_real_starpu_data_get_interface_on_node(
        handle, node);
  }
  if (node == 0) {
    return starpu_test::call_real_starpu_data_get_interface_on_node(
        handle, node);
  }
  missing_interface_override_hits.fetch_add(1, std::memory_order_relaxed);
  return nullptr;
}

auto
two_memory_nodes_override() -> unsigned
{
  return 2;
}

auto
AlwaysFailStarpuSubmit(starpu_task*) -> int
{
  submit_override_calls.fetch_add(1, std::memory_order_relaxed);
  return -17;
}

auto
NoOpStarpuDataAcquire(starpu_data_handle_t, starpu_data_access_mode) -> int
{
  return 0;
}

void
NoOpStarpuDataRelease(starpu_data_handle_t)
{
}

void
starpu_server::StarPUTaskRunnerTestAdapter::
    release_inflight_slot_via_result_dispatcher(StarPUTaskRunner* runner)
{
  if (runner == nullptr) {
    return;
  }
  ResultDispatcher::release_inflight_slot(
      runner->result_dispatcher_, runner->inflight_state_);
}

extern "C" cudaError_t
cudaStreamSynchronize(cudaStream_t stream)
{
  static auto real_fn = resolve_real_cuda_stream_synchronize();
  if (g_force_cuda_stream_sync_failure.load(std::memory_order_acquire)) {
    const auto target =
        g_cuda_stream_sync_failure_target.load(std::memory_order_acquire);
    if (target == nullptr || target == reinterpret_cast<void*>(stream)) {
      g_cuda_stream_sync_failure_count.fetch_add(1, std::memory_order_relaxed);
      return cudaErrorUnknown;
    }
  }
  if (real_fn == nullptr) {
    return cudaErrorUnknown;
  }
  return real_fn(stream);
}

extern "C" cudaError_t
cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags)
{
  static auto real_fn = resolve_real_cuda_stream_create_with_flags();
  if (g_force_cuda_stream_create_failure.load(std::memory_order_acquire)) {
    if (stream != nullptr) {
      *stream = nullptr;
    }
    g_cuda_stream_create_failure_count.fetch_add(1, std::memory_order_relaxed);
    return cudaErrorUnknown;
  }
  if (real_fn == nullptr) {
    if (stream != nullptr) {
      *stream = nullptr;
    }
    return cudaErrorUnknown;
  }
  return real_fn(stream, flags);
}

extern "C" cudaError_t
cudaMemcpyAsync(
    void* dst, const void* src, std::size_t count, cudaMemcpyKind kind,
    cudaStream_t stream)
{
  static auto real_fn = resolve_real_cuda_memcpy_async();
  if (g_force_cuda_memcpy_failure.load(std::memory_order_acquire)) {
    const auto target =
        g_cuda_memcpy_failure_stream.load(std::memory_order_acquire);
    if (target == nullptr || target == reinterpret_cast<void*>(stream)) {
      g_cuda_memcpy_failure_count.fetch_add(1, std::memory_order_relaxed);
      return cudaErrorUnknown;
    }
  }
  if (real_fn == nullptr) {
    return cudaErrorUnknown;
  }
  return real_fn(dst, src, count, kind, stream);
}
