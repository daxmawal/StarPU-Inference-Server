#include "slot_manager_component.hpp"

#include <cuda_runtime_api.h>
#include <starpu.h>
#include <starpu_data_interfaces.h>
#include <torch/torch.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <execution>
#include <format>
#include <memory>
#include <mutex>
#include <new>
#include <ranges>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "exceptions.hpp"
#include "monitoring/metrics.hpp"
#include "monitoring/runtime_observability.hpp"
#include "starpu_vector_resize_utils.hpp"
#include "task_runner_internal.hpp"
#include "utils/batching_trace_logger.hpp"
#include "utils/device_type.hpp"
#include "utils/nvtx.hpp"
#include "utils/tensor_validation.hpp"

namespace starpu_server {

using clock = task_runner_internal::Clock;

namespace {

auto
active_metrics(const std::shared_ptr<RuntimeObservability>& observability)
    -> MetricsRecorder*
{
  return observability != nullptr ? observability->metrics.get() : nullptr;
}

auto
active_tracer(const std::shared_ptr<RuntimeObservability>& observability)
    -> BatchingTraceLogger*
{
  return observability != nullptr ? observability->tracer.get() : nullptr;
}

}  // namespace

inline namespace slot_manager_component_detail {

template <typename Func>
void
parallel_for_each_index(std::size_t count, Func&& func)
{
  if (count == 0) {
    return;
  }

  const auto indices = std::views::iota(std::size_t{0}, count);

  std::atomic abort{false};
  std::exception_ptr first_error;
  std::mutex error_mutex;
  auto&& task = std::forward<Func>(func);

  std::for_each(
      std::execution::par, indices.begin(), indices.end(),
      [&](std::size_t idx) {
        if (abort.load(std::memory_order_acquire)) {
          return;
        }
        try {
          task(idx);
        }
        catch (...) {
          std::lock_guard lock(error_mutex);
          if (!first_error) {
            first_error = std::current_exception();
            abort.store(true, std::memory_order_release);
          }
        }
      });

  if (first_error) {
    std::rethrow_exception(first_error);
  }
}

[[nodiscard]] auto
is_warmup_job(const std::shared_ptr<InferenceJob>& job) -> bool
{
  return job && job->get_fixed_worker_id().has_value();
}

void
log_batch_submitted_if_enabled(
    const std::shared_ptr<InferenceJob>& job, bool warmup_job,
    const std::shared_ptr<RuntimeObservability>& observability)
{
  if (!job) {
    return;
  }
  auto* tracer = active_tracer(observability);
  if (tracer == nullptr) {
    tracer = &BatchingTraceLogger::instance();
  }
  if (!tracer->enabled()) {
    return;
  }

  const auto request_ids =
      task_runner_internal::build_request_ids_for_trace(job);
  const std::size_t logical_jobs = std::max(
      static_cast<std::size_t>(std::max(1, job->logical_job_count())),
      request_ids.size());
  tracer->log_batch_submitted(BatchingTraceLogger::BatchSubmittedLogArgs{
      .batch_id = job->submission_id(),
      .model_name = job->model_name(),
      .logical_job_count = logical_jobs,
      .worker_type = job->get_executed_on(),
      .worker_id = job->get_worker_id(),
      .request_ids = std::span<const int>(request_ids),
      .is_warmup = warmup_job,
      .device_id = job->get_device_id(),
  });
}

void
resize_output_handles_for_job(
    const std::vector<torch::Tensor>& outputs,
    const std::vector<starpu_data_handle_t>& handles)
{
  if (outputs.empty()) {
    return;
  }
  if (handles.size() < outputs.size()) {
    throw InvalidInferenceJobException(
        "Output count mismatch between job and StarPU handles");
  }

  for (std::size_t idx = 0; idx < outputs.size(); ++idx) {
    const auto& tensor = outputs[idx];
    if (!tensor.defined()) {
      continue;
    }
    if (!tensor.is_cpu() || !tensor.is_contiguous()) {
      throw InvalidInferenceJobException(
          "Output tensor must be defined, CPU and contiguous");
    }
    const task_runner_internal::VectorResizeSpec spec{
        static_cast<std::size_t>(tensor.numel()), tensor.nbytes()};
    task_runner_internal::resize_starpu_vector_handle(
        handles[idx], spec, false);
  }
}

}  // namespace slot_manager_component_detail

class SlotHandleLease {
 public:
  SlotHandleLease(
      std::span<const starpu_data_handle_t> handles,
      starpu_data_access_mode mode)
  {
    acquired_.reserve(handles.size());
    for (auto* const handle : handles) {
      if (handle == nullptr) {
        continue;
      }
      const int result_code = starpu_data_acquire(handle, mode);
      if (result_code != 0) {
        release_all();
        throw StarPUDataAcquireException(std::format(
            "starpu_data_acquire failed with code {}", result_code));
      }
      acquired_.push_back(handle);
    }
  }

  SlotHandleLease(const SlotHandleLease&) = delete;
  auto operator=(const SlotHandleLease&) -> SlotHandleLease& = delete;

  SlotHandleLease(SlotHandleLease&& other) noexcept
      : acquired_(std::move(other.acquired_))
  {
    other.acquired_.clear();
  }

  auto operator=(SlotHandleLease&& other) noexcept -> SlotHandleLease&
  {
    if (this != &other) {
      release_all();
      acquired_ = std::move(other.acquired_);
      other.acquired_.clear();
    }
    return *this;
  }

  ~SlotHandleLease() { release_all(); }

 private:
  void release_all() noexcept
  {
    for (auto* const handle : acquired_ | std::views::reverse) {
      if (handle != nullptr) {
        starpu_data_release(handle);
      }
    }
    acquired_.clear();
  }

  std::vector<starpu_data_handle_t> acquired_;
};

class CudaCopyBatch {
 public:
  explicit CudaCopyBatch(bool enable)
  {
    if (!enable) {
      return;
    }
    const auto status =
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    if (status == cudaSuccess) {
      enabled_ = true;
    } else {
      stream_ = nullptr;
    }
  }

  CudaCopyBatch(const CudaCopyBatch&) = delete;
  auto operator=(const CudaCopyBatch&) -> CudaCopyBatch& = delete;
  CudaCopyBatch(CudaCopyBatch&&) = delete;
  auto operator=(CudaCopyBatch&&) -> CudaCopyBatch& = delete;

  ~CudaCopyBatch()
  {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }

  [[nodiscard]] auto active() const -> bool { return enabled_; }
// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  [[nodiscard]] auto stream() const -> cudaStream_t { return stream_; }
  [[nodiscard]] auto pending() const -> bool { return pending_; }
#endif  // SONAR_IGNORE_END
  // GCOVR_EXCL_STOP

  auto enqueue(
      std::byte* dst, const std::byte* src, std::size_t bytes,
      bool allow_async) -> bool
  {
    if (!enabled_ || !allow_async || bytes == 0) {
      return false;
    }
    if (const auto status = cudaMemcpyAsync(
            static_cast<void*>(dst), static_cast<const void*>(src), bytes,
            cudaMemcpyHostToHost, stream_);
        status != cudaSuccess) {
      enabled_ = false;
      pending_ = false;
      return false;
    }
    pending_ = true;
    return true;
  }

  void finalize()
  {
    if (!enabled_ || !pending_) {
      return;
    }
    if (const auto status = cudaStreamSynchronize(stream_);
        status != cudaSuccess) {
      enabled_ = false;
    }
    pending_ = false;
  }

 private:
  cudaStream_t stream_{nullptr};
  bool enabled_{false};
  bool pending_{false};
};

SlotManager::SlotManager(
    StarPUSetup* starpu, const RuntimeConfig* opts,
    torch::jit::script::Module* model_cpu,
    std::vector<torch::jit::script::Module>* models_gpu,
    const std::vector<detail::GpuReplicaAssignment>* gpu_replica_assignments,
    const InferenceTaskDependencies& dependencies,
    std::shared_ptr<RuntimeObservability> observability)
    : starpu_(starpu), opts_(opts), model_cpu_(model_cpu),
      models_gpu_(models_gpu),
      gpu_replica_assignments_(gpu_replica_assignments),
      dependencies_(dependencies), observability_(std::move(observability))
{
}

auto
SlotManager::acquire_pools() const -> StarPUTaskRunner::PoolResources
{
  StarPUTaskRunner::PoolResources pools{};
  if (starpu_ != nullptr && starpu_->has_input_pool()) {
    pools.input_pool = &starpu_->input_pool();
    pools.input_slot = pools.input_pool->acquire();
  }
  if (starpu_ != nullptr && starpu_->has_output_pool()) {
    pools.output_pool = &starpu_->output_pool();
    pools.output_slot = pools.output_pool->acquire();
  }
  return pools;
}

auto
SlotManager::configure_task_context(
    InferenceTask& task, const StarPUTaskRunner::PoolResources& pools,
    std::vector<starpu_data_handle_t> input_handles,
    std::vector<starpu_data_handle_t> output_handles,
    int64_t batch_size) -> std::shared_ptr<InferenceCallbackContext>
{
  auto ctx =
      task.create_context(std::move(input_handles), std::move(output_handles));
  ctx->keep_input_handles = pools.has_input();
  ctx->keep_output_handles = pools.has_output();
  if (pools.has_output()) {
    ctx->output_pool = pools.output_pool;
    ctx->output_slot_id = pools.output_slot;
  }
  ctx->on_finished =
      [input_pool = pools.input_pool, input_slot = pools.input_slot,
       output_pool = pools.output_pool, output_slot = pools.output_slot]() {
        if (input_pool != nullptr && input_slot >= 0) {
          input_pool->release(input_slot);
        }
        if (output_pool != nullptr && output_slot >= 0) {
          output_pool->release(output_slot);
        }
      };
  if (ctx->job) {
    resize_output_handles_for_job(
        ctx->job->get_output_tensors(), ctx->outputs_handles);
  }
  if (ctx->inference_params) {
    ctx->inference_params->batch_size = batch_size;
  }
  return ctx;
}

void
SlotManager::handle_submission_failure(
    const StarPUTaskRunner::PoolResources& pools,
    const std::shared_ptr<InferenceCallbackContext>& ctx, int submit_code)
{
  InferenceTask::cleanup(ctx);
  if (pools.has_input() && pools.input_slot >= 0) {
    pools.input_pool->release(pools.input_slot);
  }
  if (pools.has_output() && pools.output_slot >= 0) {
    pools.output_pool->release(pools.output_slot);
  }
  throw StarPUTaskSubmissionException(std::format(
      "[ERROR] StarPU task submission failed (code {})", submit_code));
}

auto
SlotManager::resolve_batch_size(const std::shared_ptr<InferenceJob>& job) const
    -> int64_t
{
  return task_runner_internal::resolve_batch_size_for_job(opts_, job);
}

[[nodiscard]] auto
SlotManager::validate_batch_and_copy_inputs(
    const std::shared_ptr<InferenceJob>& job, int64_t batch,
    const StarPUTaskRunner::PoolResources& pools) const -> int64_t
{
  if (!pools.has_input()) {
    return batch;
  }

  const auto& inputs = job->get_input_tensors();
  const auto& base_ptrs = pools.input_pool->base_ptrs(pools.input_slot);
  if (inputs.size() != base_ptrs.size()) {
    throw InputPoolMismatchException(
        "Input count mismatch between job and slot");
  }
  const auto& buffer_infos =
      pools.input_pool->host_buffer_infos(pools.input_slot);
  if (base_ptrs.size() != buffer_infos.size()) {
    throw InputPoolMismatchException(
        "Input slot base pointers mismatch buffer info metadata");
  }

  if (batch < 1 || batch > pools.input_pool->max_batch_size()) {
    throw InputPoolCapacityException("Batch size exceeds input pool capacity");
  }

  NvtxRange nvtx_copy_scope("HtoD-staged host copy (pooled inputs)");
  const auto& handles = pools.input_pool->handles(pools.input_slot);

  // Track copy metrics (duration + total bytes).
  const auto copy_start = clock::now();
  std::atomic<std::size_t> total_bytes_copied{0};

  const std::span<const starpu_data_handle_t> handle_span(
      handles.data(), handles.size());
  SlotHandleLease handle_lease(handle_span, STARPU_W);

  const bool want_cuda_copy = opts_ != nullptr && opts_->devices.use_cuda &&
                              torch::cuda::is_available();
  const bool slot_has_pinned_buffers = std::ranges::any_of(
      buffer_infos, [](const InputSlotPool::HostBufferInfo& info) {
        return info.cuda_pinned || info.starpu_pinned;
      });
  CudaCopyBatch copy_batch(want_cuda_copy && slot_has_pinned_buffers);

  const auto prepare_input_for_copy = [&](std::size_t input_idx) {
    const auto& tensor = inputs[input_idx];
    const auto label = std::format("input[{}]", input_idx);
    if (auto error =
            tensor_validation::validate_cpu_contiguous_tensor(tensor, label)) {
      throw InvalidInputTensorException(*error);
    }
    const task_runner_internal::VectorResizeSpec spec{
        static_cast<std::size_t>(tensor.numel()), tensor.nbytes()};
    task_runner_internal::resize_starpu_vector_handle(
        handles[input_idx], spec, true);
    total_bytes_copied.fetch_add(spec.byte_count, std::memory_order_relaxed);
    return spec;
  };

  const auto copy_input_data =
      [&](std::size_t input_idx, const torch::Tensor& tensor,
          const task_runner_internal::VectorResizeSpec& spec) {
        const auto& buffer_info = buffer_infos.at(input_idx);
        const bool allow_async =
            buffer_info.cuda_pinned || buffer_info.starpu_pinned;
        if (!copy_batch.enqueue(
                base_ptrs[input_idx],
                static_cast<const std::byte*>(tensor.data_ptr()),
                spec.byte_count, allow_async)) {
          std::memcpy(base_ptrs[input_idx], tensor.data_ptr(), spec.byte_count);
        }
      };

  const auto copy_single_input = [&](std::size_t input_idx) {
    const auto& tensor = inputs[input_idx];
    const auto spec = prepare_input_for_copy(input_idx);
    copy_input_data(input_idx, tensor, spec);
  };

  if (job->has_pending_sub_jobs()) {
    auto pending_jobs = job->take_pending_sub_jobs();
    total_bytes_copied.fetch_add(
        copy_job_inputs_to_slot(
            job, pending_jobs, handles, base_ptrs, buffer_infos, copy_batch),
        std::memory_order_relaxed);
    release_pending_jobs(job, pending_jobs);
  } else if (copy_batch.active()) {
    for (std::size_t idx = 0; idx < inputs.size(); ++idx) {
      copy_single_input(idx);
    }
  } else {
    parallel_for_each_index(inputs.size(), copy_single_input);
  }

  copy_batch.finalize();

  const auto copy_end = clock::now();
  const auto bytes = total_bytes_copied.load(std::memory_order_relaxed);
  if (bytes > 0) {
    const double duration_ms =
        std::chrono::duration<double, std::milli>(copy_end - copy_start)
            .count();
    const auto worker_type_label =
        std::string_view(to_string(job->get_executed_on()));
    if (auto* metrics = active_metrics(observability_); metrics != nullptr) {
      metrics->observe_io_copy_latency(
          "h2d", job->get_worker_id(), job->get_device_id(), worker_type_label,
          duration_ms);
      metrics->increment_transfer_bytes(
          "h2d", job->get_worker_id(), job->get_device_id(), worker_type_label,
          bytes);
    } else {
      observe_io_copy_latency(
          "h2d", job->get_worker_id(), job->get_device_id(), worker_type_label,
          duration_ms);
      increment_transfer_bytes(
          "h2d", job->get_worker_id(), job->get_device_id(), worker_type_label,
          bytes);
    }
  }

  return batch;
}

auto
SlotManager::submit_inference_task(
    const std::shared_ptr<InferenceJob>& job) const -> void
{
  task_runner_internal::invoke_submit_inference_task_hook();

  auto label =
      std::format("submit job {}", task_runner_internal::job_identifier(*job));
  NvtxRange nvtx_job_scope(label);
  const bool warmup_job = is_warmup_job(job);
  if (starpu_ == nullptr ||
      !(starpu_->has_input_pool() || starpu_->has_output_pool())) {
    InferenceTask task(
        starpu_, job, model_cpu_, models_gpu_, opts_, dependencies_,
        gpu_replica_assignments_);
    task.submit();
    log_batch_submitted_if_enabled(job, warmup_job, observability_);
    return;
  }

  struct SubmitPipelineContext {
    std::shared_ptr<InferenceJob> job;
    bool warmup_job = false;
    StarPUTaskRunner::PoolResources pools{};
    int64_t batch_size = 1;
    struct Handles {
      std::vector<starpu_data_handle_t> input_storage;
      std::vector<starpu_data_handle_t> output_storage;
      std::vector<starpu_data_handle_t> input_for_context;
      std::vector<starpu_data_handle_t> output_for_context;
    };
    Handles handles{};
    std::shared_ptr<InferenceCallbackContext> callback_context;
    starpu_task* task_ptr = nullptr;
  };

  struct SlotPoolReleaseGuard {
    explicit SlotPoolReleaseGuard(
        const StarPUTaskRunner::PoolResources& pool_resources) noexcept
        : pools(pool_resources)
    {
    }
    SlotPoolReleaseGuard(const SlotPoolReleaseGuard&) = delete;
    SlotPoolReleaseGuard(SlotPoolReleaseGuard&&) = delete;
    auto operator=(const SlotPoolReleaseGuard&) -> SlotPoolReleaseGuard& =
                                                       delete;
    auto operator=(SlotPoolReleaseGuard&&) -> SlotPoolReleaseGuard& = delete;
    ~SlotPoolReleaseGuard() noexcept
    {
      if (!active) {
        return;
      }
      release();
    }

    void dismiss() noexcept { active = false; }

   private:
    void release() noexcept
    {
      if (pools.has_input() && pools.input_slot >= 0) {
        pools.input_pool->release(pools.input_slot);
      }
      if (pools.has_output() && pools.output_slot >= 0) {
        pools.output_pool->release(pools.output_slot);
      }
    }

    const StarPUTaskRunner::PoolResources& pools;
    bool active{true};
  };

  SubmitPipelineContext context{};
  context.job = job;
  context.warmup_job = warmup_job;
  context.pools = acquire_pools();
  SlotPoolReleaseGuard pool_guard(context.pools);
  context.batch_size = resolve_batch_size(context.job);
  if (context.pools.has_input()) {
    context.batch_size = validate_batch_and_copy_inputs(
        context.job, context.batch_size, context.pools);
  }

  InferenceTask task(
      starpu_, job, model_cpu_, models_gpu_, opts_, dependencies_,
      gpu_replica_assignments_);

  if (context.pools.has_input()) {
    context.handles.input_for_context =
        context.pools.input_pool->handles(context.pools.input_slot);
  } else {
    context.handles.input_storage = task.prepare_input_handles();
    context.handles.input_for_context = context.handles.input_storage;
  }

  if (context.pools.has_output()) {
    context.handles.output_for_context =
        context.pools.output_pool->handles(context.pools.output_slot);
  } else {
    context.handles.output_storage = task.prepare_output_handles();
    context.handles.output_for_context = context.handles.output_storage;
  }

  context.callback_context = configure_task_context(
      task, context.pools, std::move(context.handles.input_for_context),
      std::move(context.handles.output_for_context), context.batch_size);
  context.task_ptr = task.create_task(
      context.callback_context->inputs_handles,
      context.callback_context->outputs_handles, context.callback_context);

  const auto submitted_at = clock::now();
  context.job->update_timing_info([submitted_at](detail::TimingInfo& timing) {
    timing.before_starpu_submitted_time = submitted_at;
  });

  const int submit_code = starpu_task_submit(context.task_ptr);
  if (submit_code != 0) {
    pool_guard.dismiss();
    handle_submission_failure(
        context.pools, context.callback_context, submit_code);
  }

  pool_guard.dismiss();
  log_batch_submitted_if_enabled(
      context.job, context.warmup_job, observability_);
}

auto
SlotManager::copy_job_inputs_to_slot(
    const std::shared_ptr<InferenceJob>& job,
    std::span<const std::shared_ptr<InferenceJob>> pending_jobs,
    std::span<const starpu_data_handle_t> handles,
    std::span<std::byte* const> base_ptrs,
    std::span<const InputSlotPool::HostBufferInfo> buffer_infos,
    CudaCopyBatch& copy_batch) -> std::size_t
{
  if (!job) {
    return 0;
  }

  std::size_t total_bytes_copied = 0;
  const auto& master_inputs = job->get_input_tensors();
  for (std::size_t input_idx = 0; input_idx < master_inputs.size();
       ++input_idx) {
    auto* dst = base_ptrs[input_idx];
    std::size_t offset = 0;
    std::size_t total_numel = 0;
    std::size_t total_bytes = 0;
    const auto& buffer_info = buffer_infos[input_idx];
    auto buffer_span = std::span<std::byte>(dst, buffer_info.bytes);
    const bool allow_async =
        buffer_info.cuda_pinned || buffer_info.starpu_pinned;

    const auto input_label = std::format("input[{}]", input_idx);
    const auto validate_tensor = [&input_label](const torch::Tensor& tensor) {
      if (auto error = tensor_validation::validate_cpu_contiguous_tensor(
              tensor, input_label)) {
        throw InvalidInputTensorException(*error);
      }
    };
    auto ensure_capacity = [&](size_t bytes) {
      if (buffer_span.size() < offset + bytes) {
        throw InputPoolCapacityException(
            "Input tensor exceeds allocated slot buffer capacity");
      }
    };
    auto transfer_tensor_data = [&](const torch::Tensor& tensor, size_t bytes) {
      auto* destination = buffer_span.subspan(offset, bytes).data();
      if (!copy_batch.enqueue(
              destination, static_cast<const std::byte*>(tensor.data_ptr()),
              bytes, allow_async)) {
        std::memcpy(destination, tensor.data_ptr(), bytes);
      }
    };
    auto copy_tensor = [&](const torch::Tensor& tensor) {
      validate_tensor(tensor);
      const size_t bytes = tensor.nbytes();
      const auto numel = static_cast<size_t>(tensor.numel());
      ensure_capacity(bytes);
      transfer_tensor_data(tensor, bytes);
      offset += bytes;
      total_bytes += bytes;
      total_numel += numel;
      total_bytes_copied += bytes;
    };

    copy_tensor(master_inputs[input_idx]);
    for (const auto& pending : pending_jobs) {
      if (!pending) {
        continue;
      }
      const auto& pending_inputs = pending->get_input_tensors();
      if (input_idx >= pending_inputs.size()) {
        throw InconsistentInputTensorCountException(
            "Inconsistent input tensor count");
      }
      copy_tensor(pending_inputs[input_idx]);
    }

    const task_runner_internal::VectorResizeSpec spec{total_numel, total_bytes};
    task_runner_internal::resize_starpu_vector_handle(
        handles[input_idx], spec, true);
  }

  return total_bytes_copied;
}

void
SlotManager::release_pending_jobs(
    const std::shared_ptr<InferenceJob>& job,
    std::vector<std::shared_ptr<InferenceJob>>& pending_jobs)
{
  if (pending_jobs.empty()) {
    return;
  }

  std::vector<std::shared_ptr<InferenceJob>> release_jobs;
  release_jobs.reserve(pending_jobs.size() + 1);
  release_jobs.push_back(job);
  for (auto& pending : pending_jobs) {
    release_jobs.push_back(std::move(pending));
  }
  task_runner_internal::release_inputs_from_additional_jobs(release_jobs);
  pending_jobs.clear();
}

// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
auto
SlotManager::validate_batch_and_copy_inputs_for_test(
    SlotManager* manager, const std::shared_ptr<InferenceJob>& job,
    int64_t batch, InputSlotPool* input_pool, int input_slot,
    OutputSlotPool* output_pool, int output_slot) -> int64_t
{
  if (manager == nullptr) {
    return -1;
  }
  StarPUTaskRunner::PoolResources pools{};
  pools.input_pool = input_pool;
  pools.input_slot = input_slot;
  pools.output_pool = output_pool;
  pools.output_slot = output_slot;
  return manager->validate_batch_and_copy_inputs(job, batch, pools);
}

namespace task_runner_internal::testing {

void
resize_starpu_vector_interface(
    starpu_vector_interface* vector_interface, VectorResizeSpecShim spec,
    bool is_input_handle)
{
  task_runner_internal::resize_starpu_vector_interface(
      vector_interface,
      task_runner_internal::VectorResizeSpec{
          spec.element_count, spec.byte_count},
      is_input_handle);
}

auto
cuda_copy_batch_create(bool enable) -> void*
{
  auto batch = std::make_unique<CudaCopyBatch>(enable);
  return batch.release();
}

void
cuda_copy_batch_destroy(void* batch)
{
  std::unique_ptr<CudaCopyBatch> batch_owner(
      static_cast<CudaCopyBatch*>(batch));
  static_cast<void>(batch_owner);
}

auto
cuda_copy_batch_enqueue(
    void* batch, std::byte* dst, const std::byte* src, std::size_t bytes,
    bool allow_async) -> bool
{
  if (batch == nullptr) {
    return false;
  }
  return static_cast<CudaCopyBatch*>(batch)->enqueue(
      dst, src, bytes, allow_async);
}

void
cuda_copy_batch_finalize(void* batch)
{
  if (batch == nullptr) {
    return;
  }
  static_cast<CudaCopyBatch*>(batch)->finalize();
}

auto
cuda_copy_batch_enabled(const void* batch) -> bool
{
  return batch != nullptr ? static_cast<const CudaCopyBatch*>(batch)->active()
                          : false;
}

auto
cuda_copy_batch_pending(const void* batch) -> bool
{
  return batch != nullptr ? static_cast<const CudaCopyBatch*>(batch)->pending()
                          : false;
}

auto
cuda_copy_batch_stream(const void* batch) -> cudaStream_t
{
  return batch != nullptr ? static_cast<const CudaCopyBatch*>(batch)->stream()
                          : nullptr;
}

void
slot_handle_lease_construct(
    void* storage, std::span<const starpu_data_handle_t> handles,
    starpu_data_access_mode mode)
{
  if (storage == nullptr) {
    return;
  }
  new (storage) SlotHandleLease(handles, mode);
}

void
slot_handle_lease_destroy(void* storage)
{
  if (storage == nullptr) {
    return;
  }
  std::destroy_at(static_cast<SlotHandleLease*>(storage));
}

auto
slot_manager_copy_job_inputs_to_slot(
    const std::shared_ptr<InferenceJob>& job,
    std::span<const std::shared_ptr<InferenceJob>> pending_jobs,
    std::span<const starpu_data_handle_t> handles,
    std::span<std::byte* const> base_ptrs,
    std::span<const InputSlotPool::HostBufferInfo> buffer_infos,
    void* copy_batch) -> std::size_t
{
  if (copy_batch == nullptr) {
    CudaCopyBatch fallback(false);
    return SlotManager::copy_job_inputs_to_slot(
        job, pending_jobs, handles, base_ptrs, buffer_infos, fallback);
  }
  return SlotManager::copy_job_inputs_to_slot(
      job, pending_jobs, handles, base_ptrs, buffer_infos,
      *static_cast<CudaCopyBatch*>(copy_batch));
}

auto
slot_manager_validate_batch_and_copy_inputs(
    SlotManager* slot_manager, const std::shared_ptr<InferenceJob>& job,
    int64_t batch, InputSlotPool* input_pool, int input_slot,
    OutputSlotPool* output_pool, int output_slot) -> int64_t
{
  return SlotManager::validate_batch_and_copy_inputs_for_test(
      slot_manager, job, batch, input_pool, input_slot, output_pool,
      output_slot);
}

}  // namespace task_runner_internal::testing
#endif  // SONAR_IGNORE_END
// GCOVR_EXCL_STOP

}  // namespace starpu_server
