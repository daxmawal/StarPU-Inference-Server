#include "starpu_task_worker.hpp"

#include <cuda_runtime_api.h>
#include <starpu.h>
#include <starpu_data_interfaces.h>
#include <torch/torch.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <exception>
#include <execution>
#include <format>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "exceptions.hpp"
#include "inference_task.hpp"
#include "logger.hpp"
#include "task_runner_internal.hpp"
#include "utils/batching_trace_logger.hpp"
#include "utils/nvtx.hpp"
#include "utils/perf_observer.hpp"

namespace starpu_server {
namespace task_runner_internal {

template <typename Ptr>
inline void
validate_not_null(Ptr ptr, std::string_view field_name)
{
  static_assert(
      std::is_pointer_v<Ptr>, "validate_not_null expects a pointer argument");
  if (ptr != nullptr) {
    return;
  }
  throw std::invalid_argument(std::format(
      "[ERROR] StarPUTaskRunnerConfig::{} must not be null", field_name));
}

inline auto
batch_size_from_inputs(const std::vector<torch::Tensor>& inputs) -> std::size_t
{
  if (inputs.empty()) {
    return 1;
  }

  const auto& first = inputs.front();
  if (first.dim() <= 0) {
    return 1;
  }

  const auto dim0 = first.size(0);
  return dim0 > 0 ? static_cast<std::size_t>(dim0) : std::size_t{1};
}

inline auto
job_identifier(const InferenceJob& job) -> int
{
  const int submission_id = job.submission_id();
  return (submission_id >= 0) ? submission_id : job.get_request_id();
}

struct ExceptionLoggingMessages {
  std::string_view context_prefix;
  std::string_view unknown_message;
};

namespace {
auto
submit_inference_task_hook_storage() -> std::function<void()>&
{
  static std::function<void()> hook;
  return hook;
}
}  // namespace

void
set_submit_inference_task_hook(std::function<void()> hook)
{
  submit_inference_task_hook_storage() = std::move(hook);
}

void
reset_submit_inference_task_hook()
{
  submit_inference_task_hook_storage() = {};
}

static void
invoke_submit_inference_task_hook()
{
  const auto& hook = submit_inference_task_hook_storage();
  if (hook) {
    hook();
  }
}

template <typename Callback>
void
run_with_logged_exceptions(
    Callback&& callback, const ExceptionLoggingMessages& messages)
{
  try {
    std::forward<Callback>(callback)();
  }
  catch (const InferenceEngineException& e) {
    log_error(std::string(messages.context_prefix) + e.what());
  }
  catch (const std::runtime_error& e) {
    log_error(std::string(messages.context_prefix) + e.what());
  }
  catch (const std::logic_error& e) {
    log_error(std::string(messages.context_prefix) + e.what());
  }
  catch (const std::bad_alloc& e) {
    log_error(std::string(messages.context_prefix) + e.what());
  }
  catch (...) {
    log_error(std::string(messages.unknown_message));
  }
}

inline auto
select_earliest_time(Clock::time_point current, Clock::time_point candidate)
    -> Clock::time_point
{
  if (candidate == Clock::time_point{}) {
    return current;
  }
  if (current == Clock::time_point{} || candidate < current) {
    return candidate;
  }
  return current;
}

inline auto
select_latest_time(Clock::time_point current, Clock::time_point candidate)
    -> Clock::time_point
{
  if (candidate == Clock::time_point{}) {
    return current;
  }
  if (current == Clock::time_point{} || candidate > current) {
    return candidate;
  }
  return current;
}

auto
slice_outputs_for_sub_job(
    const std::vector<torch::Tensor>& aggregated_outputs,
    SubJobSliceOptions options) -> SubJobSliceResult
{
  SubJobSliceResult result;
  const int64_t slice_size = std::max<int64_t>(1, options.batch_size);
  result.processed_length = slice_size;

  if (aggregated_outputs.empty()) {
    return result;
  }

  result.outputs.reserve(aggregated_outputs.size());
  bool determined_length = false;
  const auto slice_start = static_cast<int64_t>(options.offset);

  for (const auto& tensor : aggregated_outputs) {
    if (!tensor.defined() || tensor.dim() == 0) {
      result.outputs.push_back(tensor);
      continue;
    }

    const int64_t available = tensor.size(0);
    const int64_t slice_end =
        std::min<int64_t>(available, slice_start + slice_size);
    const int64_t length = std::max<int64_t>(0, slice_end - slice_start);

    if (length <= 0) {
      result.outputs.emplace_back();
      continue;
    }

    if (!determined_length) {
      result.processed_length = length;
      determined_length = true;
    }

    auto slice_view = tensor.narrow(0, slice_start, length);
    if (!slice_view.is_contiguous()) {
      slice_view = slice_view.contiguous();
    }
    result.outputs.push_back(std::move(slice_view));
  }

  return result;
}

auto
aggregate_batch_metadata(const std::vector<std::shared_ptr<InferenceJob>>& jobs)
    -> BatchAggregationInfo
{
  BatchAggregationInfo info;
  if (jobs.empty()) {
    return info;
  }

  info.sub_jobs.reserve(jobs.size());
  info.earliest_start = jobs.front()->get_start_time();
  info.earliest_enqueued = jobs.front()->timing_info().enqueued_time;
  info.latest_enqueued = jobs.front()->timing_info().enqueued_time;
  info.earliest_batch_collect_start =
      jobs.front()->timing_info().batch_collect_start_time;

  for (const auto& job : jobs) {
    const auto job_batch =
        static_cast<int64_t>(batch_size_from_inputs(job->get_input_tensors()));
    info.total_samples += job_batch > 0 ? job_batch : 1;
    info.logical_jobs += std::max(1, job->logical_job_count());
    info.earliest_start =
        select_earliest_time(info.earliest_start, job->get_start_time());
    info.earliest_enqueued = select_earliest_time(
        info.earliest_enqueued, job->timing_info().enqueued_time);
    info.latest_enqueued = select_latest_time(
        info.latest_enqueued, job->timing_info().enqueued_time);
    info.earliest_batch_collect_start = select_earliest_time(
        info.earliest_batch_collect_start,
        job->timing_info().batch_collect_start_time);

    InferenceJob::AggregatedSubJob entry{};
    entry.job = std::weak_ptr<InferenceJob>(job);
    entry.callback = job->get_on_complete();
    entry.batch_size = job_batch;
    entry.request_id = job->get_request_id();
    entry.arrival_time = job->timing_info().enqueued_time;
    info.sub_jobs.push_back(std::move(entry));
  }

  return info;
}

auto
resize_outputs_for_batch(
    const std::vector<torch::Tensor>& prototype_outputs,
    int64_t batch_size) -> std::vector<torch::Tensor>
{
  std::vector<torch::Tensor> resized;
  resized.reserve(prototype_outputs.size());
  for (const auto& out : prototype_outputs) {
    if (!out.defined()) {
      resized.emplace_back(out);
      continue;
    }
    std::vector<int64_t> shape(out.sizes().begin(), out.sizes().end());
    if (!shape.empty()) {
      shape.front() = batch_size;
    }
    resized.emplace_back(torch::empty(shape, out.options()));
  }
  return resized;
}

void
release_inputs_from_additional_jobs(
    std::vector<std::shared_ptr<InferenceJob>>& jobs)
{
  for (size_t idx = 1; idx < jobs.size(); ++idx) {
    if (!jobs[idx]) {
      continue;
    }
    static_cast<void>(jobs[idx]->release_input_tensors());
    jobs[idx]->set_input_memory_holders(
        std::vector<std::shared_ptr<const void>>{});
  }
}
}  // namespace task_runner_internal

namespace {

[[nodiscard]] auto
request_id_from_sub_job(const InferenceJob::AggregatedSubJob& sub_job) -> int
{
  if (sub_job.request_id >= 0) {
    return sub_job.request_id;
  }
  if (auto locked = sub_job.job.lock()) {
    return locked->get_request_id();
  }
  return sub_job.request_id;
}

auto
build_request_ids_for_trace(const std::shared_ptr<InferenceJob>& job)
    -> std::vector<int>
{
  if (!job) {
    return {};
  }

  if (!job->has_aggregated_sub_jobs()) {
    return std::vector<int>{job->get_request_id()};
  }

  const auto& aggregated = job->aggregated_sub_jobs();
  std::vector<int> ids;
  ids.reserve(aggregated.size());
  for (const auto& sub_job : aggregated) {
    ids.push_back(request_id_from_sub_job(sub_job));
  }

  return ids;
}

auto
build_request_arrival_us_for_trace(const std::shared_ptr<InferenceJob>& job)
    -> std::vector<int64_t>
{
  using Clock = std::chrono::high_resolution_clock;
  const auto to_microseconds = [](Clock::time_point time_point) -> int64_t {
    if (time_point == Clock::time_point{}) {
      return 0;
    }
    return std::chrono::duration_cast<std::chrono::microseconds>(
               time_point.time_since_epoch())
        .count();
  };

  if (!job) {
    return {};
  }

  if (!job->has_aggregated_sub_jobs()) {
    return std::vector<int64_t>{
        to_microseconds(job->timing_info().enqueued_time)};
  }

  const auto& aggregated = job->aggregated_sub_jobs();
  std::vector<int64_t> arrivals;
  arrivals.reserve(aggregated.size());
  for (const auto& sub_job : aggregated) {
    auto arrival = sub_job.arrival_time;
    if (arrival == Clock::time_point{}) {
      if (auto locked = sub_job.job.lock()) {
        arrival = locked->timing_info().enqueued_time;
      }
    }
    arrivals.push_back(to_microseconds(arrival));
  }
  return arrivals;
}

inline auto
is_warmup_job(const std::shared_ptr<InferenceJob>& job) -> bool
{
  return job && job->get_fixed_worker_id().has_value();
}

struct VectorResizeSpec {
  std::size_t element_count;
  std::size_t byte_count;
};

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
      std::execution::par_unseq, indices.begin(), indices.end(),
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

inline void
resize_starpu_vector_interface(
    starpu_vector_interface* vector_interface, VectorResizeSpec spec,
    bool is_input_handle)
{
  if (vector_interface == nullptr) {
    return;
  }

  const auto elem_size = vector_interface->elemsize;
  if (elem_size == 0) {
    throw StarPUDataAcquireException(
        "StarPU vector interface reported zero element size");
  }

  if (spec.byte_count % elem_size != 0) {
    if (is_input_handle) {
      throw InvalidInputTensorException(std::format(
          "Input tensor byte size ({}) is not divisible by element size ({})",
          spec.byte_count, elem_size));
    }
    throw InvalidInferenceJobException(std::format(
        "Output tensor byte size ({}) is not divisible by element size ({})",
        spec.byte_count, elem_size));
  }

  const auto required_numel = spec.byte_count / elem_size;
  if (required_numel != spec.element_count) {
    spec.element_count = required_numel;
  }

  const auto alloc_size = vector_interface->allocsize;
  if (alloc_size != std::numeric_limits<size_t>::max() && alloc_size != 0 &&
      spec.byte_count > alloc_size) {
    if (is_input_handle) {
      throw InputPoolCapacityException(std::format(
          "Input tensor requires {} bytes but slot capacity is {} bytes",
          spec.byte_count, alloc_size));
    }
    throw InvalidInferenceJobException(std::format(
        "Output tensor requires {} bytes but slot capacity is {} bytes",
        spec.byte_count, alloc_size));
  }

  vector_interface->nx = spec.element_count;
}

inline void
resize_starpu_vector_handle(
    starpu_data_handle_t handle, VectorResizeSpec spec, bool is_input_handle)
{
  if (handle == nullptr) {
    throw StarPUDataAcquireException("StarPU vector handle is null");
  }

  if (starpu_data_get_interface_id(handle) != STARPU_VECTOR_INTERFACE_ID) {
    throw StarPUDataAcquireException(
        "Expected StarPU vector interface for handle");
  }

  const unsigned memory_nodes = starpu_memory_nodes_get_count();
  for (unsigned node = 0; node < memory_nodes; ++node) {
    auto* raw_interface = starpu_data_get_interface_on_node(handle, node);
    if (raw_interface == nullptr) {
      continue;
    }
    auto* vector_interface =
        static_cast<starpu_vector_interface*>(raw_interface);
    resize_starpu_vector_interface(vector_interface, spec, is_input_handle);
  }
}

inline void
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

  for (size_t idx = 0; idx < outputs.size(); ++idx) {
    const auto& tensor = outputs[idx];
    if (!tensor.defined()) {
      continue;
    }
    if (!tensor.is_cpu() || !tensor.is_contiguous()) {
      throw InvalidInferenceJobException(
          "Output tensor must be defined, CPU and contiguous");
    }
    const VectorResizeSpec spec{
        static_cast<std::size_t>(tensor.numel()), tensor.nbytes()};
    resize_starpu_vector_handle(handles[idx], spec, false);
  }
}

}  // namespace

using clock = task_runner_internal::Clock;

class ResultDispatcher {
 public:
  ResultDispatcher(
      const RuntimeConfig* opts, std::atomic<int>* completed_jobs,
      std::condition_variable* all_done_cv)
      : opts_(opts), completed_jobs_(completed_jobs), all_done_cv_(all_done_cv)
  {
  }

  void prepare_job_completion_callback(
      StarPUTaskRunner& runner, const std::shared_ptr<InferenceJob>& job) const;

  void store_completed_job_result(
      const std::shared_ptr<InferenceJob>& job,
      const std::vector<torch::Tensor>& results, double latency_ms) const;

  static void ensure_callback_timing(detail::TimingInfo& timing);

  void record_job_metrics(
      const std::shared_ptr<InferenceJob>& job,
      StarPUTaskRunner::DurationMs latency, std::size_t batch_size) const;

  void log_job_timings(
      int request_id, StarPUTaskRunner::DurationMs latency,
      const detail::TimingInfo& timing_info) const;

  void finalize_job_completion(const std::shared_ptr<InferenceJob>& job) const;
  static auto resolve_batch_size(
      StarPUTaskRunner& runner,
      const std::shared_ptr<InferenceJob>& job) -> std::size_t;
  static void emit_batch_traces(
      const std::shared_ptr<InferenceJob>& job, StarPUTaskRunner& runner);
  static void invoke_previous_callback(
      const std::function<void(std::vector<torch::Tensor>&&, double)>& previous,
      std::vector<torch::Tensor>& results, double latency_ms);

  static void propagate_completion_to_sub_jobs(
      const std::shared_ptr<InferenceJob>& aggregated_job,
      const std::vector<torch::Tensor>& aggregated_outputs, double latency_ms);

 private:
  const RuntimeConfig* opts_;
  std::atomic<int>* completed_jobs_;
  std::condition_variable* all_done_cv_;
};

class SlotManager {
 public:
  SlotManager(StarPUSetup* starpu, const RuntimeConfig* opts)
      : starpu_(starpu), opts_(opts)
  {
  }

  auto acquire_pools() -> StarPUTaskRunner::PoolResources;

  [[nodiscard]] auto validate_batch_and_copy_inputs(
      const std::shared_ptr<InferenceJob>& job, int64_t batch,
      const StarPUTaskRunner::PoolResources& pools) const -> int64_t;

  static void copy_job_inputs_to_slot(
      const std::shared_ptr<InferenceJob>& job,
      std::span<const std::shared_ptr<InferenceJob>> pending_jobs,
      std::span<const starpu_data_handle_t> handles,
      std::span<std::byte* const> base_ptrs,
      std::span<const InputSlotPool::HostBufferInfo> buffer_infos,
      CudaCopyBatch& copy_batch);

  static void release_pending_jobs(
      const std::shared_ptr<InferenceJob>& job,
      std::vector<std::shared_ptr<InferenceJob>>& pending_jobs);

 private:
  StarPUSetup* starpu_;
  const RuntimeConfig* opts_;
};

struct PreparedBatchingContext {
  std::mutex* prepared_mutex{};
  std::condition_variable* prepared_cv{};
  std::deque<std::shared_ptr<InferenceJob>>* prepared_jobs{};
  bool* batching_done{};
};

class BatchCollector {
 public:
  BatchCollector(
      InferenceQueue* queue, const RuntimeConfig* opts, StarPUSetup* starpu,
      std::shared_ptr<InferenceJob>* pending_job,
      const PreparedBatchingContext& prepared)
      : queue_(queue), opts_(opts), starpu_(starpu), pending_job_(pending_job),
        prepared_mutex_(prepared.prepared_mutex),
        prepared_cv_(prepared.prepared_cv),
        prepared_jobs_(prepared.prepared_jobs),
        batching_done_(prepared.batching_done)
  {
  }

  auto wait_for_next_job() -> std::shared_ptr<InferenceJob>;
  auto collect_batch(const std::shared_ptr<InferenceJob>& first_job)
      -> std::vector<std::shared_ptr<InferenceJob>>;
  auto maybe_build_batched_job(std::vector<std::shared_ptr<InferenceJob>>& jobs)
      -> std::shared_ptr<InferenceJob>;
  void enqueue_prepared_job(const std::shared_ptr<InferenceJob>& job);
  auto wait_for_prepared_job() -> std::shared_ptr<InferenceJob>;
  void batching_loop();

  static auto can_merge_jobs(
      const std::shared_ptr<InferenceJob>& lhs,
      const std::shared_ptr<InferenceJob>& rhs) -> bool;
  static auto merge_input_tensors(
      const std::vector<std::shared_ptr<InferenceJob>>& jobs,
      int64_t total_samples) -> std::vector<torch::Tensor>;
  static auto merge_input_memory_holders(
      const std::vector<std::shared_ptr<InferenceJob>>& jobs)
      -> std::vector<std::shared_ptr<const void>>;

 private:
  [[nodiscard]] auto job_sample_size(
      const std::shared_ptr<InferenceJob>& job) const -> int64_t;
  [[nodiscard]] auto sample_limit_per_batch() const -> int;
  [[nodiscard]] auto try_acquire_next_job(
      bool enable_wait,
      clock::time_point coalesce_deadline) -> std::shared_ptr<InferenceJob>;
  void store_pending_job(const std::shared_ptr<InferenceJob>& job);
  [[nodiscard]] static auto should_hold_job(
      const std::shared_ptr<InferenceJob>& candidate,
      const std::shared_ptr<InferenceJob>& reference,
      const std::optional<int>& target_worker) -> bool;
  [[nodiscard]] auto exceeds_sample_limit(
      int64_t accumulated_samples, const std::shared_ptr<InferenceJob>& job,
      int64_t max_samples_cap) const -> bool;

  InferenceQueue* queue_;
  const RuntimeConfig* opts_;
  StarPUSetup* starpu_;
  std::shared_ptr<InferenceJob>* pending_job_;
  std::mutex* prepared_mutex_;
  std::condition_variable* prepared_cv_;
  std::deque<std::shared_ptr<InferenceJob>>* prepared_jobs_;
  bool* batching_done_;
};

void
ResultDispatcher::prepare_job_completion_callback(
    StarPUTaskRunner& runner, const std::shared_ptr<InferenceJob>& job) const
{
  auto prev_callback = job->get_on_complete();
  const auto* dispatcher = this;
  job->set_on_complete(
      [dispatcher, prev_callback, job_sptr = job, &runner](
          std::vector<torch::Tensor> results, double latency_ms) mutable {
        dispatcher->store_completed_job_result(job_sptr, results, latency_ms);
        ResultDispatcher::ensure_callback_timing(job_sptr->timing_info());
        dispatcher->record_job_metrics(
            job_sptr, StarPUTaskRunner::DurationMs{latency_ms},
            ResultDispatcher::resolve_batch_size(runner, job_sptr));
        ResultDispatcher::emit_batch_traces(job_sptr, runner);
        ResultDispatcher::invoke_previous_callback(
            prev_callback, results, latency_ms);
        dispatcher->finalize_job_completion(job_sptr);
      });
}

void
ResultDispatcher::store_completed_job_result(
    const std::shared_ptr<InferenceJob>& job,
    const std::vector<torch::Tensor>& results, double latency_ms) const
{
  (void)results;
  (void)latency_ms;
  (void)job->release_input_tensors();  // drop inputs once callbacks are done
}

void
ResultDispatcher::ensure_callback_timing(detail::TimingInfo& timing)
{
  const auto zero_tp = clock::time_point{};
  const auto now = clock::now();

  if (timing.callback_start_time == zero_tp) {
    timing.callback_start_time = now;
  }
  if (timing.callback_end_time == zero_tp) {
    timing.callback_end_time = now;
  }
  if (timing.callback_end_time <= timing.callback_start_time) {
    timing.callback_end_time = timing.callback_start_time + clock::duration{1};
  }
  if (timing.enqueued_time == zero_tp ||
      timing.enqueued_time >= timing.callback_end_time) {
    timing.enqueued_time = timing.callback_start_time;
  }
  if (timing.last_enqueued_time == zero_tp ||
      timing.last_enqueued_time < timing.enqueued_time) {
    timing.last_enqueued_time = timing.enqueued_time;
  }
}

void
ResultDispatcher::record_job_metrics(
    const std::shared_ptr<InferenceJob>& job,
    StarPUTaskRunner::DurationMs latency, std::size_t batch_size) const
{
  auto& timing = job->timing_info();
  perf_observer::record_job(
      timing.enqueued_time, timing.callback_end_time, batch_size,
      is_warmup_job(job));

  timing.submission_id = job->submission_id();
  const int job_id = task_runner_internal::job_identifier(*job);
  log_job_timings(job_id, latency, timing);

  auto& tracer = BatchingTraceLogger::instance();
  if (tracer.enabled()) {
    const auto breakdown =
        detail::compute_latency_breakdown(timing, latency.count());
    const auto request_ids = build_request_ids_for_trace(job);
    const auto request_arrivals = build_request_arrival_us_for_trace(job);
    tracer.log_batch_summary(BatchingTraceLogger::BatchSummaryLogArgs{
        .batch_id = job_id,
        .model_name = job->model_name(),
        .batch_size = batch_size,
        .request_ids = request_ids,
        .request_arrival_us = request_arrivals,
        .worker_id = job->get_worker_id(),
        .worker_type = job->get_executed_on(),
        .device_id = job->get_device_id(),
        .queue_ms = breakdown.queue_ms,
        .batch_ms = breakdown.batch_ms,
        .submit_ms = breakdown.submit_ms,
        .scheduling_ms = breakdown.scheduling_ms,
        .codelet_ms = breakdown.codelet_ms,
        .inference_ms = breakdown.inference_ms,
        .callback_ms = breakdown.callback_ms,
        .total_ms = breakdown.total_ms,
        .is_warmup = is_warmup_job(job),
    });
  }
}

void
ResultDispatcher::log_job_timings(
    int request_id, StarPUTaskRunner::DurationMs latency,
    const detail::TimingInfo& timing_info) const
{
  if (!should_log(VerbosityLevel::Stats, opts_->verbosity)) {
    return;
  }

  const auto submission_id = timing_info.submission_id;
  const auto base =
      detail::compute_latency_breakdown(timing_info, latency.count());
  const int job_id = submission_id >= 0 ? submission_id : request_id;
  const auto header = std::format(
      "Job {} done. Latency = {:.3f} ms | Queue = ", job_id, latency.count());

  log_stats(
      opts_->verbosity,
      std::format(
          "{}{:.3f} ms, Batch = {:.3f} ms, Submit = {:.3f} ms, Scheduling = "
          "{:.3f} ms, Codelet = {:.3f} ms, Inference = {:.3f} ms, Callback = "
          "{:.3f} ms",
          header, base.queue_ms, base.batch_ms, base.submit_ms,
          base.scheduling_ms, base.codelet_ms, base.inference_ms,
          base.callback_ms));
}

void
ResultDispatcher::finalize_job_completion(
    const std::shared_ptr<InferenceJob>& job) const
{
  const int logical_jobs = std::max(1, job->logical_job_count());
  completed_jobs_->fetch_add(logical_jobs, std::memory_order_release);
  all_done_cv_->notify_all();
}

void
ResultDispatcher::propagate_completion_to_sub_jobs(
    const std::shared_ptr<InferenceJob>& aggregated_job,
    const std::vector<torch::Tensor>& aggregated_outputs, double latency_ms)
{
  if (!aggregated_job) {
    return;
  }

  const auto& sub_jobs = aggregated_job->aggregated_sub_jobs();
  if (sub_jobs.empty()) {
    return;
  }

  size_t offset = 0;
  for (const auto& entry : sub_jobs) {
    auto job_sp = entry.job.lock();
    const auto slice_size =
        static_cast<std::size_t>(std::max<int64_t>(1, entry.batch_size));
    if (!job_sp) {
      offset += slice_size;
      continue;
    }

    auto slice_result = task_runner_internal::slice_outputs_for_sub_job(
        aggregated_outputs,
        task_runner_internal::SubJobSliceOptions{offset, entry.batch_size});
    auto outputs = std::move(slice_result.outputs);

    job_sp->timing_info() = aggregated_job->timing_info();
    job_sp->get_device_id() = aggregated_job->get_device_id();
    job_sp->get_worker_id() = aggregated_job->get_worker_id();
    job_sp->get_executed_on() = aggregated_job->get_executed_on();
    job_sp->set_submission_id(aggregated_job->submission_id());
    job_sp->timing_info().submission_id = aggregated_job->submission_id();

    if (entry.callback) {
      entry.callback(outputs, latency_ms);
    }

    outputs.clear();
    static_cast<void>(job_sp->release_input_tensors());
    job_sp->release_input_memory_holders();

    offset += static_cast<std::size_t>(
        std::max<int64_t>(1, slice_result.processed_length));
  }

  auto& mutable_job =
      const_cast<std::shared_ptr<InferenceJob>&>(aggregated_job);

  const auto& pending = mutable_job->pending_sub_jobs();
  for (const auto& sub_job : pending) {
    if (sub_job && sub_job->has_on_complete()) {
      sub_job->set_on_complete({});
    }
  }

  mutable_job->set_aggregated_sub_jobs({});
  mutable_job->clear_pending_sub_jobs();

  static_cast<void>(mutable_job->release_input_tensors());
  mutable_job->release_input_memory_holders();
  mutable_job->set_output_tensors({});
}

auto
SlotManager::acquire_pools() -> StarPUTaskRunner::PoolResources
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

  const auto copy_single_input = [&](std::size_t input_idx) {
    const auto& tensor = inputs[input_idx];
    if (!tensor.defined() || !tensor.is_cpu() || !tensor.is_contiguous()) {
      throw InvalidInputTensorException(
          "Input tensor must be defined, CPU and contiguous");
    }
    const VectorResizeSpec spec{
        static_cast<std::size_t>(tensor.numel()), tensor.nbytes()};
    resize_starpu_vector_handle(handles[input_idx], spec, true);

    const auto& buffer_info = buffer_infos.at(input_idx);
    const bool allow_async =
        buffer_info.cuda_pinned || buffer_info.starpu_pinned;
    if (!copy_batch.enqueue(
            base_ptrs[input_idx],
            static_cast<const std::byte*>(tensor.data_ptr()), spec.byte_count,
            allow_async)) {
      std::memcpy(base_ptrs[input_idx], tensor.data_ptr(), spec.byte_count);
    }
  };

  if (job->has_pending_sub_jobs()) {
    auto pending_jobs = job->take_pending_sub_jobs();
    copy_job_inputs_to_slot(
        job, pending_jobs, handles, base_ptrs, buffer_infos, copy_batch);
    release_pending_jobs(job, pending_jobs);
  } else if (copy_batch.active()) {
    for (std::size_t idx = 0; idx < inputs.size(); ++idx) {
      copy_single_input(idx);
    }
  } else {
    parallel_for_each_index(inputs.size(), copy_single_input);
  }

  copy_batch.finalize();

  return batch;
}

void
SlotManager::copy_job_inputs_to_slot(
    const std::shared_ptr<InferenceJob>& job,
    std::span<const std::shared_ptr<InferenceJob>> pending_jobs,
    std::span<const starpu_data_handle_t> handles,
    std::span<std::byte* const> base_ptrs,
    std::span<const InputSlotPool::HostBufferInfo> buffer_infos,
    CudaCopyBatch& copy_batch)
{
  if (!job) {
    return;
  }

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

    const auto validate_tensor = [](const torch::Tensor& tensor) {
      if (!tensor.defined() || !tensor.is_cpu() || !tensor.is_contiguous()) {
        throw InvalidInputTensorException(
            "Input tensor must be defined, CPU and contiguous");
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

    const VectorResizeSpec spec{total_numel, total_bytes};
    resize_starpu_vector_handle(handles[input_idx], spec, true);
  }
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

auto
BatchCollector::wait_for_next_job() -> std::shared_ptr<InferenceJob>
{
  if (pending_job_ != nullptr && *pending_job_) {
    return std::exchange(*pending_job_, nullptr);
  }
  std::shared_ptr<InferenceJob> job;
  if (queue_ == nullptr || !queue_->wait_and_pop(job)) {
    return nullptr;
  }
  return job;
}

auto
BatchCollector::collect_batch(const std::shared_ptr<InferenceJob>& first_job)
    -> std::vector<std::shared_ptr<InferenceJob>>
{
  std::vector<std::shared_ptr<InferenceJob>> jobs;
  if (first_job == nullptr) {
    return jobs;
  }

  jobs.push_back(first_job);
  if (!opts_->batching.dynamic_batching) {
    return jobs;
  }
  if (first_job->has_aggregated_sub_jobs() ||
      first_job->logical_job_count() > 1) {
    return jobs;
  }

  const int max_job_count = std::max(1, opts_->batching.max_batch_size);
  if (max_job_count <= 1) {
    return jobs;
  }

  const int64_t max_samples_cap = sample_limit_per_batch();
  int64_t accumulated_samples = job_sample_size(first_job);

  const bool enable_wait = opts_->batching.batch_coalesce_timeout_ms > 0;
  const auto batch_coalesce_timeout =
      std::chrono::milliseconds(opts_->batching.batch_coalesce_timeout_ms);
  const auto coalesce_deadline =
      enable_wait ? clock::now() + batch_coalesce_timeout : clock::time_point{};
  const auto& target_worker = first_job->get_fixed_worker_id();

  while (jobs.size() < static_cast<size_t>(max_job_count)) {
    auto next = try_acquire_next_job(enable_wait, coalesce_deadline);
    const bool should_break =
        next == nullptr || should_hold_job(next, jobs.front(), target_worker) ||
        exceeds_sample_limit(accumulated_samples, next, max_samples_cap);
    if (next && should_break) {
      store_pending_job(next);
    }
    if (should_break) {
      break;
    }
    accumulated_samples += job_sample_size(next);
    jobs.push_back(std::move(next));
  }

  return jobs;
}

auto
BatchCollector::try_acquire_next_job(
    bool enable_wait,
    clock::time_point coalesce_deadline) -> std::shared_ptr<InferenceJob>
{
  if (queue_ == nullptr) {
    return nullptr;
  }

  while (true) {
    std::shared_ptr<InferenceJob> next;
    bool got_job = queue_->try_pop(next);
    if (!got_job && enable_wait) {
      const auto now = clock::now();
      if (now >= coalesce_deadline) {
        return nullptr;
      }
      got_job = queue_->wait_for_and_pop(next, coalesce_deadline - now);
    }
    if (!got_job) {
      return nullptr;
    }
    if (next) {
      return next;
    }
  }
}

void
BatchCollector::store_pending_job(const std::shared_ptr<InferenceJob>& job)
{
  if (pending_job_ != nullptr) {
    *pending_job_ = job;
  }
}

[[nodiscard]] auto
BatchCollector::should_hold_job(
    const std::shared_ptr<InferenceJob>& candidate,
    const std::shared_ptr<InferenceJob>& reference,
    const std::optional<int>& target_worker) -> bool
{
  if (!candidate) {
    return false;
  }
  if (candidate->is_shutdown() || candidate->has_aggregated_sub_jobs() ||
      candidate->logical_job_count() > 1) {
    return true;
  }
  if (target_worker != candidate->get_fixed_worker_id()) {
    return true;
  }
  return !can_merge_jobs(reference, candidate);
}

[[nodiscard]] auto
BatchCollector::exceeds_sample_limit(
    int64_t accumulated_samples, const std::shared_ptr<InferenceJob>& job,
    int64_t max_samples_cap) const -> bool
{
  if (max_samples_cap <= 0) {
    return false;
  }
  const int64_t next_samples = job_sample_size(job);
  return accumulated_samples + next_samples > max_samples_cap;
}

auto
BatchCollector::can_merge_jobs(
    const std::shared_ptr<InferenceJob>& lhs,
    const std::shared_ptr<InferenceJob>& rhs) -> bool
{
  if (!lhs || !rhs) {
    return false;
  }

  const auto& lhs_inputs = lhs->get_input_tensors();
  const auto& rhs_inputs = rhs->get_input_tensors();
  if (lhs_inputs.size() != rhs_inputs.size()) {
    return false;
  }

  const auto& lhs_types = lhs->get_input_types();
  const auto& rhs_types = rhs->get_input_types();
  if (lhs_types.size() != rhs_types.size()) {
    return false;
  }
  for (size_t idx = 0; idx < lhs_types.size(); ++idx) {
    if (lhs_types[idx] != rhs_types[idx]) {
      return false;
    }
  }

  for (size_t idx = 0; idx < lhs_inputs.size(); ++idx) {
    const auto& lhs_tensor = lhs_inputs[idx];
    const auto& rhs_tensor = rhs_inputs[idx];
    if (!lhs_tensor.defined() || !rhs_tensor.defined()) {
      return false;
    }
    if (lhs_tensor.dim() != rhs_tensor.dim()) {
      return false;
    }
    if (lhs_tensor.dim() <= 0) {
      return false;
    }
    if (lhs_tensor.dim() <= 1) {
      continue;
    }
    for (int64_t dim = 1; dim < lhs_tensor.dim(); ++dim) {
      if (lhs_tensor.size(dim) != rhs_tensor.size(dim)) {
        return false;
      }
    }
  }

  return true;
}

namespace {

void
validate_prototype_tensor(const torch::Tensor& tensor)
{
  if (!tensor.defined()) {
    throw InvalidInputTensorException(
        "Input tensor must be defined before batching");
  }
  if (tensor.dim() <= 0) {
    throw InvalidInputTensorException(
        "Input tensor must have at least one dimension");
  }
}

void
validate_tensor_against_prototype(
    const torch::Tensor& tensor, const torch::Tensor& prototype)
{
  if (!tensor.defined()) {
    throw InvalidInputTensorException(
        "Input tensor must be defined before batching");
  }
  if (tensor.dim() != prototype.dim()) {
    throw InvalidInputTensorException(
        "Input tensor rank mismatch during batching");
  }
  if (tensor.dim() <= 0) {
    throw InvalidInputTensorException(
        "Input tensor must have at least one dimension");
  }
  for (int64_t dim = 1; dim < tensor.dim(); ++dim) {
    if (tensor.size(dim) != prototype.size(dim)) {
      throw InvalidInputTensorException(
          "Input tensor shape mismatch during batching");
    }
  }
}

auto
accumulate_samples_for_tensor(
    const std::vector<std::shared_ptr<InferenceJob>>& jobs, size_t tensor_idx,
    const torch::Tensor& prototype) -> int64_t
{
  int64_t accumulated_samples = 0;
  for (const auto& job : jobs) {
    const auto& tensors = job->get_input_tensors();
    if (tensor_idx >= tensors.size()) {
      throw InconsistentInputTensorCountException(
          "Inconsistent input tensor count");
    }
    const auto& tensor = tensors[tensor_idx];
    validate_tensor_against_prototype(tensor, prototype);
    accumulated_samples += tensor.size(0);
  }
  return accumulated_samples;
}

void
copy_tensor_slices_to_merged(
    const std::vector<std::shared_ptr<InferenceJob>>& jobs, size_t tensor_idx,
    const torch::Tensor& merged_tensor)
{
  int64_t offset = 0;
  for (const auto& job : jobs) {
    const auto& tensor = job->get_input_tensors()[tensor_idx];
    const int64_t slice = tensor.size(0);
    merged_tensor.narrow(0, offset, slice).copy_(tensor);
    offset += slice;
  }
}

}  // namespace

auto
BatchCollector::merge_input_tensors(
    const std::vector<std::shared_ptr<InferenceJob>>& jobs,
    int64_t total_samples) -> std::vector<torch::Tensor>
{
  std::vector<torch::Tensor> merged;
  if (jobs.empty()) {
    return merged;
  }
  const auto& first_inputs = jobs.front()->get_input_tensors();
  merged.reserve(first_inputs.size());
  const bool single_job_batch = jobs.size() == 1;

  for (size_t tensor_idx = 0; tensor_idx < first_inputs.size(); ++tensor_idx) {
    if (single_job_batch) {
      merged.push_back(first_inputs[tensor_idx]);
      continue;
    }

    const auto& prototype = first_inputs[tensor_idx];
    validate_prototype_tensor(prototype);

    const int64_t accumulated_samples =
        accumulate_samples_for_tensor(jobs, tensor_idx, prototype);
    const int64_t target_samples =
        total_samples > 0 ? total_samples : accumulated_samples;
    if (accumulated_samples != target_samples) {
      throw InvalidInputTensorException(
          "Total samples mismatch while batching inputs");
    }

    auto shape = prototype.sizes().vec();
    shape.front() = target_samples;
    auto merged_tensor = torch::empty(shape, prototype.options());
    copy_tensor_slices_to_merged(jobs, tensor_idx, merged_tensor);

    merged.emplace_back(std::move(merged_tensor));
  }

  return merged;
}

auto
BatchCollector::merge_input_memory_holders(
    const std::vector<std::shared_ptr<InferenceJob>>& jobs)
    -> std::vector<std::shared_ptr<const void>>
{
  std::vector<std::shared_ptr<const void>> holders;
  std::size_t total_holders = 0;
  for (const auto& job : jobs) {
    total_holders += job->get_input_memory_holders().size();
  }
  holders.reserve(total_holders);
  for (const auto& job : jobs) {
    const auto& job_holders = job->get_input_memory_holders();
    holders.insert(holders.end(), job_holders.begin(), job_holders.end());
  }
  return holders;
}

auto
BatchCollector::job_sample_size(const std::shared_ptr<InferenceJob>& job) const
    -> int64_t
{
  if (!job) {
    return 0;
  }
  if (const auto effective = job->effective_batch_size();
      effective.has_value()) {
    return std::max<int64_t>(1, *effective);
  }

  const auto& inputs = job->get_input_tensors();
  if (inputs.empty()) {
    return 1;
  }

  if (opts_ != nullptr && !opts_->models.empty() &&
      !opts_->models[0].inputs.empty()) {
    const auto per_sample_rank =
        static_cast<int64_t>(opts_->models[0].inputs[0].dims.size());
    const int64_t rank0 = inputs[0].dim();
    if (rank0 == per_sample_rank + 1 && rank0 > 0) {
      return std::max<int64_t>(1, inputs[0].size(0));
    }
  }

  return static_cast<int64_t>(
      task_runner_internal::batch_size_from_inputs(inputs));
}

auto
BatchCollector::sample_limit_per_batch() const -> int
{
  const int configured_limit =
      opts_ != nullptr ? std::max(1, opts_->batching.max_batch_size) : 1;

  if (starpu_ != nullptr && starpu_->has_input_pool()) {
    const int pool_limit = std::max(1, starpu_->input_pool().max_batch_size());
    return std::min(configured_limit, pool_limit);
  }

  return configured_limit;
}

auto
BatchCollector::maybe_build_batched_job(
    std::vector<std::shared_ptr<InferenceJob>>& jobs)
    -> std::shared_ptr<InferenceJob>
{
  if (jobs.empty()) {
    return nullptr;
  }

  auto master = jobs.front();
  if (jobs.size() == 1) {
    master->set_logical_job_count(1);
    master->set_aggregated_sub_jobs({});
    return master;
  }

  auto batch_info = task_runner_internal::aggregate_batch_metadata(jobs);
  auto earliest_start = batch_info.earliest_start;
  auto earliest_enqueued = batch_info.earliest_enqueued;
  auto earliest_batch_collect_start = batch_info.earliest_batch_collect_start;

  if (earliest_start == clock::time_point{}) {
    earliest_start = earliest_enqueued != clock::time_point{}
                         ? earliest_enqueued
                         : clock::now();
  }
  if (earliest_batch_collect_start == clock::time_point{}) {
    earliest_batch_collect_start = master->timing_info().dequeued_time;
  }

  master->set_logical_job_count(batch_info.logical_jobs);
  master->set_aggregated_sub_jobs(std::move(batch_info.sub_jobs));

  if (auto lifetimes = merge_input_memory_holders(jobs); !lifetimes.empty()) {
    master->set_input_memory_holders(std::move(lifetimes));
  }

  const auto prototype_outputs = master->get_output_tensors();
  const int64_t effective_batch = batch_info.total_samples > 0
                                      ? batch_info.total_samples
                                      : static_cast<int64_t>(jobs.size());
  master->set_output_tensors(task_runner_internal::resize_outputs_for_batch(
      prototype_outputs, effective_batch));

  master->set_start_time(earliest_start);
  master->timing_info().enqueued_time = earliest_enqueued;
  master->timing_info().last_enqueued_time =
      batch_info.latest_enqueued == clock::time_point{}
          ? earliest_enqueued
          : batch_info.latest_enqueued;
  master->timing_info().batch_collect_start_time = earliest_batch_collect_start;

  if (const bool need_materialized_inputs =
          (starpu_ == nullptr || !starpu_->has_input_pool()) ||
          (opts_ != nullptr && opts_->validation.validate_results)) {
    if (auto merged_inputs = merge_input_tensors(jobs, effective_batch);
        !merged_inputs.empty()) {
      master->set_input_tensors(merged_inputs);
    }
    task_runner_internal::release_inputs_from_additional_jobs(jobs);
    master->clear_pending_sub_jobs();
  } else {
    std::vector<std::shared_ptr<InferenceJob>> pending_jobs;
    pending_jobs.reserve(jobs.empty() ? 0 : jobs.size() - 1);
    for (size_t idx = 1; idx < jobs.size(); ++idx) {
      pending_jobs.push_back(jobs[idx]);
      jobs[idx]->set_output_tensors({});
    }
    master->set_pending_sub_jobs(std::move(pending_jobs));
  }

  master->set_effective_batch_size(effective_batch);

  auto master_wp = std::weak_ptr<InferenceJob>(master);
  master->set_on_complete(
      [master_wp](
          const std::vector<torch::Tensor>& aggregated_outputs,
          double latency_ms) {
        if (auto master_sp = master_wp.lock()) {
          ResultDispatcher::propagate_completion_to_sub_jobs(
              master_sp, aggregated_outputs, latency_ms);
        }
      });

  if (opts_ != nullptr && should_log(VerbosityLevel::Trace, opts_->verbosity)) {
    log_trace(
        opts_->verbosity,
        std::format(
            "Formed batch for job ID {} with {} requests ({} samples)",
            master->get_request_id(), jobs.size(), effective_batch));
  }

  return master;
}

void
BatchCollector::enqueue_prepared_job(const std::shared_ptr<InferenceJob>& job)
{
  if (prepared_mutex_ == nullptr || prepared_jobs_ == nullptr ||
      prepared_cv_ == nullptr) {
    return;
  }
  {
    const std::scoped_lock lock(*prepared_mutex_);
    prepared_jobs_->push_back(job);
  }
  prepared_cv_->notify_one();
}

auto
BatchCollector::wait_for_prepared_job() -> std::shared_ptr<InferenceJob>
{
  if (prepared_mutex_ == nullptr || prepared_jobs_ == nullptr ||
      prepared_cv_ == nullptr) {
    return nullptr;
  }
  std::unique_lock lock(*prepared_mutex_);
  prepared_cv_->wait(lock, [this] {
    return !prepared_jobs_->empty() ||
           (batching_done_ != nullptr && *batching_done_);
  });
  if (prepared_jobs_->empty()) {
    return nullptr;
  }
  auto job = prepared_jobs_->front();
  prepared_jobs_->pop_front();
  return job;
}

void
BatchCollector::batching_loop()
{
  bool should_stop = false;
  while (!should_stop) {
    auto job = wait_for_next_job();
    if (!job) {
      should_stop = true;
      continue;
    }

    if (job->is_shutdown()) {
      enqueue_prepared_job(job);
      should_stop = true;
      continue;
    }

    const auto dequeue_time = std::chrono::high_resolution_clock::now();
    job->timing_info().dequeued_time = dequeue_time;
    job->timing_info().batch_collect_start_time = dequeue_time;
    job->timing_info().batch_collect_end_time = dequeue_time;

    auto jobs = collect_batch(job);
    job = maybe_build_batched_job(jobs);
    if (!job) {
      continue;
    }

    job->timing_info().batch_collect_end_time =
        std::chrono::high_resolution_clock::now();

    enqueue_prepared_job(job);
  }

  if (prepared_mutex_ != nullptr && prepared_cv_ != nullptr &&
      batching_done_ != nullptr) {
    {
      const std::scoped_lock lock(*prepared_mutex_);
      *batching_done_ = true;
    }
    prepared_cv_->notify_all();
  }
}
// =============================================================================
// Constructor
// =============================================================================

StarPUTaskRunner::StarPUTaskRunner(const StarPUTaskRunnerConfig& config)
    : queue_(config.queue), model_cpu_(config.model_cpu),
      models_gpu_(config.models_gpu), starpu_(config.starpu),
      opts_(config.opts), completed_jobs_(config.completed_jobs),
      all_done_cv_(config.all_done_cv),
      dependencies_(
          config.dependencies != nullptr ? config.dependencies
                                         : &kDefaultInferenceTaskDependencies),
      batch_collector_(std::make_unique<BatchCollector>(
          queue_, opts_, starpu_, &pending_job_,
          PreparedBatchingContext{
              .prepared_mutex = &prepared_mutex_,
              .prepared_cv = &prepared_cv_,
              .prepared_jobs = &prepared_jobs_,
              .batching_done = &batching_done_,
          })),
      slot_manager_(std::make_unique<SlotManager>(starpu_, opts_)),
      result_dispatcher_(std::make_unique<ResultDispatcher>(
          opts_, completed_jobs_, all_done_cv_))
{
  task_runner_internal::validate_not_null(queue_, "queue");
  task_runner_internal::validate_not_null(model_cpu_, "model_cpu");
  task_runner_internal::validate_not_null(models_gpu_, "models_gpu");
  task_runner_internal::validate_not_null(starpu_, "starpu");
  task_runner_internal::validate_not_null(opts_, "opts");
  task_runner_internal::validate_not_null(completed_jobs_, "completed_jobs");
  task_runner_internal::validate_not_null(all_done_cv_, "all_done_cv");
}

StarPUTaskRunner::~StarPUTaskRunner() = default;


// =============================================================================
// Job Queue Management
// =============================================================================

auto
StarPUTaskRunner::wait_for_next_job() -> std::shared_ptr<InferenceJob>
{
  return batch_collector_->wait_for_next_job();
}

auto
StarPUTaskRunner::should_shutdown(
    const std::shared_ptr<InferenceJob>& job) const -> bool
{
  if (job->is_shutdown()) {
    log_info(
        opts_->verbosity,
        "Received shutdown signal. Exiting StarPUTaskRunner loop.");
    return true;
  }
  return false;
}

// =============================================================================
// Completion Callback Handling
// =============================================================================

void
StarPUTaskRunner::log_job_timings(
    int request_id, DurationMs latency,
    const detail::TimingInfo& timing_info) const
{
  result_dispatcher_->log_job_timings(request_id, latency, timing_info);
}

void
StarPUTaskRunner::prepare_job_completion_callback(
    const std::shared_ptr<InferenceJob>& job)
{
  result_dispatcher_->prepare_job_completion_callback(*this, job);
}

void
StarPUTaskRunner::store_completed_job_result(
    const std::shared_ptr<InferenceJob>& job,
    const std::vector<torch::Tensor>& results, double latency_ms) const
{
  result_dispatcher_->store_completed_job_result(job, results, latency_ms);
}

void
StarPUTaskRunner::ensure_callback_timing(detail::TimingInfo& timing)
{
  ResultDispatcher::ensure_callback_timing(timing);
}

void
StarPUTaskRunner::record_job_metrics(
    const std::shared_ptr<InferenceJob>& job, DurationMs latency,
    std::size_t batch_size) const
{
  result_dispatcher_->record_job_metrics(job, latency, batch_size);
}

void
StarPUTaskRunner::finalize_job_completion(
    const std::shared_ptr<InferenceJob>& job) const
{
  result_dispatcher_->finalize_job_completion(job);
}

void
StarPUTaskRunner::trace_batch_if_enabled(
    const std::shared_ptr<InferenceJob>& job, bool warmup_job,
    int submission_id) const
{
  auto& tracer = BatchingTraceLogger::instance();
  if (!tracer.enabled()) {
    return;
  }

  const auto batch_size = std::max<std::size_t>(
      std::size_t{1}, static_cast<std::size_t>(resolve_batch_size(job)));
  const auto request_ids = build_request_ids_for_trace(job);
  const auto request_ids_span = std::span<const int>(request_ids);
  const auto enqueue_start = job->timing_info().enqueued_time;
  auto enqueue_end = job->timing_info().last_enqueued_time;
  if (const auto zero_tp = clock::time_point{};
      enqueue_end == zero_tp ||
      (enqueue_start != zero_tp && enqueue_end < enqueue_start)) {
    enqueue_end = enqueue_start;
  }

  tracer.log_batch_enqueue_span(
      submission_id, job->model_name(), batch_size,
      BatchingTraceLogger::TimeRange{enqueue_start, enqueue_end},
      request_ids_span, warmup_job);
  tracer.log_batch_build_span(
      submission_id, job->model_name(), batch_size,
      BatchingTraceLogger::TimeRange{
          job->timing_info().batch_collect_start_time,
          job->timing_info().batch_collect_end_time},
      request_ids_span, warmup_job);
}

void
StarPUTaskRunner::finalize_job_after_exception(
    const std::shared_ptr<InferenceJob>& job, const std::exception& exception,
    std::string_view log_prefix, int job_id)
{
  if (!log_prefix.empty()) {
    log_error(
        std::format("{} for job {}: {}", log_prefix, job_id, exception.what()));
  }

  const bool completion_done =
      StarPUTaskRunner::handle_job_exception(job, exception);
  if (!completion_done && job) {
    finalize_job_completion(job);
  }
}

void
StarPUTaskRunner::submit_job_or_handle_failure(
    const std::shared_ptr<InferenceJob>& job, SubmissionInfo submission_info)
{
  try {
    if (should_log(VerbosityLevel::Debug, opts_->verbosity)) {
      log_debug(
          opts_->verbosity,
          std::format("Submitting job ID: {}", submission_info.submission_id));
    }
    submit_inference_task(job);
  }
  catch (const InferenceEngineException& exception) {
    finalize_job_after_exception(job, exception, "", submission_info.job_id);
  }
  catch (const std::runtime_error& exception) {
    finalize_job_after_exception(
        job, exception, "Unexpected runtime error", submission_info.job_id);
  }
  catch (const std::logic_error& exception) {
    finalize_job_after_exception(
        job, exception, "Unexpected logic error", submission_info.job_id);
  }
  catch (const std::bad_alloc& exception) {
    finalize_job_after_exception(
        job, exception, "Memory allocation failure", submission_info.job_id);
  }
}

// =============================================================================
// Error Handling for Failed Jobs
// =============================================================================

auto
StarPUTaskRunner::handle_job_exception(
    const std::shared_ptr<InferenceJob>& job,
    const std::exception& exception) -> bool
{
  const int job_id = job ? task_runner_internal::job_identifier(*job) : -1;
  log_error(std::format("[Exception] Job {}: {}", job_id, exception.what()));

  if (job == nullptr) {
    return false;
  }

  bool completion_invoked = false;
  if (job->has_on_complete()) {
    const auto completion = job->get_on_complete();
    task_runner_internal::run_with_logged_exceptions(
        [&completion, &completion_invoked]() {
          completion({}, -1);
          completion_invoked = true;
        },
        task_runner_internal::ExceptionLoggingMessages{
            "Exception in completion callback: ",
            "Unknown exception in completion callback"});
  }

  return completion_invoked;
}

// =============================================================================
// StarPU Task Submission
// =============================================================================

auto
StarPUTaskRunner::acquire_pools() -> PoolResources
{
  return slot_manager_->acquire_pools();
}

auto
StarPUTaskRunner::validate_batch_and_copy_inputs(
    const std::shared_ptr<InferenceJob>& job,
    const PoolResources& pools) -> int64_t
{
  const int64_t batch = resolve_batch_size(job);
  if (!pools.has_input()) {
    return batch;
  }
  return slot_manager_->validate_batch_and_copy_inputs(job, batch, pools);
}

[[nodiscard]] auto
StarPUTaskRunner::resolve_batch_size(
    const std::shared_ptr<InferenceJob>& job) const -> int64_t
{
  if (!job) {
    return 1;
  }
  if (const auto effective = job->effective_batch_size();
      effective.has_value()) {
    return std::max<int64_t>(1, *effective);
  }

  const auto& inputs = job->get_input_tensors();
  if (inputs.empty()) {
    return 1;
  }

  if (!opts_->models.empty() && !opts_->models[0].inputs.empty()) {
    const auto per_sample_rank =
        static_cast<int64_t>(opts_->models[0].inputs[0].dims.size());
    if (const auto rank0 = inputs[0].dim();
        rank0 == per_sample_rank + 1 && rank0 > 0) {
      return inputs[0].size(0);
    }
    return 1;
  }

  return static_cast<int64_t>(
      task_runner_internal::batch_size_from_inputs(inputs));
}

void
StarPUTaskRunner::release_pending_jobs(
    const std::shared_ptr<InferenceJob>& job,
    std::vector<std::shared_ptr<InferenceJob>>& pending_jobs)
{
  SlotManager::release_pending_jobs(job, pending_jobs);
}

auto
StarPUTaskRunner::collect_batch(const std::shared_ptr<InferenceJob>& first_job)
    -> std::vector<std::shared_ptr<InferenceJob>>
{
  return batch_collector_->collect_batch(first_job);
}

auto
StarPUTaskRunner::can_merge_jobs(
    const std::shared_ptr<InferenceJob>& lhs,
    const std::shared_ptr<InferenceJob>& rhs) -> bool
{
  return BatchCollector::can_merge_jobs(lhs, rhs);
}

auto
StarPUTaskRunner::merge_input_tensors(
    const std::vector<std::shared_ptr<InferenceJob>>& jobs,
    int64_t total_samples) -> std::vector<torch::Tensor>
{
  return BatchCollector::merge_input_tensors(jobs, total_samples);
}

auto
StarPUTaskRunner::merge_input_memory_holders(
    const std::vector<std::shared_ptr<InferenceJob>>& jobs)
    -> std::vector<std::shared_ptr<const void>>
{
  return BatchCollector::merge_input_memory_holders(jobs);
}

auto
StarPUTaskRunner::maybe_build_batched_job(
    std::vector<std::shared_ptr<InferenceJob>>& jobs)
    -> std::shared_ptr<InferenceJob>
{
  return batch_collector_->maybe_build_batched_job(jobs);
}

void
StarPUTaskRunner::enqueue_prepared_job(const std::shared_ptr<InferenceJob>& job)
{
  batch_collector_->enqueue_prepared_job(job);
}

auto
StarPUTaskRunner::wait_for_prepared_job() -> std::shared_ptr<InferenceJob>
{
  return batch_collector_->wait_for_prepared_job();
}

void
StarPUTaskRunner::batching_loop()
{
  batch_collector_->batching_loop();
}

void
StarPUTaskRunner::propagate_completion_to_sub_jobs(
    const std::shared_ptr<InferenceJob>& aggregated_job,
    const std::vector<torch::Tensor>& aggregated_outputs, double latency_ms)
{
  ResultDispatcher::propagate_completion_to_sub_jobs(
      aggregated_job, aggregated_outputs, latency_ms);
}

auto
StarPUTaskRunner::configure_task_context(
    InferenceTask& task, const PoolResources& pools,
    const std::vector<starpu_data_handle_t>& input_handles,
    const std::vector<starpu_data_handle_t>& output_handles,
    int64_t batch_size) -> std::shared_ptr<InferenceCallbackContext>
{
  auto ctx = task.create_context(input_handles, output_handles);
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
        ctx->job->get_output_tensors(), output_handles);
  }
  if (ctx->inference_params) {
    ctx->inference_params->batch_size = batch_size;
  }
  return ctx;
}

void
StarPUTaskRunner::handle_submission_failure(
    const PoolResources& pools,
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

void
StarPUTaskRunner::submit_inference_task(
    const std::shared_ptr<InferenceJob>& job)
{
  task_runner_internal::invoke_submit_inference_task_hook();

  auto label =
      std::format("submit job {}", task_runner_internal::job_identifier(*job));
  NvtxRange nvtx_job_scope(label);
  const bool warmup_job = is_warmup_job(job);
  if (!(starpu_->has_input_pool() || starpu_->has_output_pool())) {
    InferenceTask task(
        starpu_, job, model_cpu_, models_gpu_, opts_, *dependencies_);
    task.submit();
    auto& tracer = BatchingTraceLogger::instance();
    if (tracer.enabled()) {
      const auto request_ids = build_request_ids_for_trace(job);
      const std::size_t logical_jobs = std::max(
          static_cast<std::size_t>(std::max(1, job->logical_job_count())),
          request_ids.size());
      tracer.log_batch_submitted(BatchingTraceLogger::BatchSubmittedLogArgs{
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
    return;
  }

  auto pools = acquire_pools();
  bool copied_ok = !pools.has_input();
  const bool should_release_output_slot =
      pools.has_output() && pools.output_slot >= 0;
  bool release_output_slot_on_exception = false;

  try {
    const auto batch = validate_batch_and_copy_inputs(job, pools);
    copied_ok = true;
    release_output_slot_on_exception = should_release_output_slot;

    InferenceTask task(
        starpu_, job, model_cpu_, models_gpu_, opts_, *dependencies_);

    std::vector<starpu_data_handle_t> input_handles_storage;
    const std::vector<starpu_data_handle_t>* input_handles = nullptr;
    if (pools.has_input()) {
      input_handles = &pools.input_pool->handles(pools.input_slot);
    } else {
      input_handles_storage = task.prepare_input_handles();
      input_handles = &input_handles_storage;
    }

    std::vector<starpu_data_handle_t> output_handles_storage;
    const std::vector<starpu_data_handle_t>* output_handles = nullptr;
    if (pools.has_output()) {
      output_handles = &pools.output_pool->handles(pools.output_slot);
    } else {
      output_handles_storage = task.prepare_output_handles();
      output_handles = &output_handles_storage;
    }

    auto ctx = configure_task_context(
        task, pools, *input_handles, *output_handles, batch);

    starpu_task* task_ptr =
        task.create_task(*input_handles, *output_handles, ctx);

    job->timing_info().before_starpu_submitted_time =
        std::chrono::high_resolution_clock::now();

    const int ret = starpu_task_submit(task_ptr);
    if (ret != 0) {
      release_output_slot_on_exception = false;
      handle_submission_failure(pools, ctx, ret);
    } else {
      auto& tracer = BatchingTraceLogger::instance();
      if (tracer.enabled()) {
        const auto request_ids = build_request_ids_for_trace(job);
        const std::size_t logical_jobs = std::max(
            static_cast<std::size_t>(std::max(1, job->logical_job_count())),
            request_ids.size());
        tracer.log_batch_submitted(BatchingTraceLogger::BatchSubmittedLogArgs{
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
    }
    release_output_slot_on_exception = false;
  }
  catch (...) {
    if (!copied_ok) {
      if (pools.has_input() && pools.input_slot >= 0) {
        pools.input_pool->release(pools.input_slot);
      }
      if (pools.has_output() && pools.output_slot >= 0) {
        pools.output_pool->release(pools.output_slot);
      }
    } else if (release_output_slot_on_exception) {
      pools.output_pool->release(pools.output_slot);
    }
    throw;
  }
}

// =============================================================================
// Main run loop: pull jobs, submit them, handle shutdown and errors
// =============================================================================

void
StarPUTaskRunner::run()
{
  log_info(opts_->verbosity, "StarPUTaskRunner started.");

  {
    const std::scoped_lock lock(prepared_mutex_);
    prepared_jobs_.clear();
    batching_done_ = false;
  }

  batching_thread_ = std::jthread(&StarPUTaskRunner::batching_loop, this);

  while (true) {
    auto job = wait_for_prepared_job();
    if (!job || should_shutdown(job)) {
      break;
    }

    const auto submission_id = next_submission_id_.fetch_add(1);
    job->set_submission_id(submission_id);
    job->timing_info().submission_id = submission_id;

    const int logical_jobs = job->logical_job_count();
    const auto request_id = job->get_request_id();
    const int job_id = task_runner_internal::job_identifier(*job);
    if (should_log(VerbosityLevel::Trace, opts_->verbosity)) {
      log_trace(
          opts_->verbosity,
          std::format(
              "Dequeued job submission {} (request {}), queue size : {}, "
              "aggregated requests: {}",
              job_id, request_id, queue_->size(), logical_jobs));
    }

    const bool warmup_job = is_warmup_job(job);
    trace_batch_if_enabled(job, warmup_job, submission_id);
    prepare_job_completion_callback(job);
    submit_job_or_handle_failure(job, SubmissionInfo{submission_id, job_id});
  }

  if (batching_thread_.joinable()) {
    batching_thread_.join();
  }

  log_info(opts_->verbosity, "StarPUTaskRunner stopped.");
}

auto
ResultDispatcher::resolve_batch_size(
    StarPUTaskRunner& runner,
    const std::shared_ptr<InferenceJob>& job) -> std::size_t
{
  return std::max<std::size_t>(
      std::size_t{1}, static_cast<std::size_t>(runner.resolve_batch_size(job)));
}

void
ResultDispatcher::emit_batch_traces(
    const std::shared_ptr<InferenceJob>& job, StarPUTaskRunner& runner)
{
  auto& tracer = BatchingTraceLogger::instance();
  if (!tracer.enabled()) {
    return;
  }
  const bool warmup_job = is_warmup_job(job);
  auto& timing = job->timing_info();
  const auto zero_tp = clock::time_point{};
  auto compute_start = timing.inference_start_time;
  if (compute_start == zero_tp) {
    compute_start = timing.codelet_start_time;
  }
  auto compute_end = timing.callback_start_time;
  if (compute_end == zero_tp || compute_end < compute_start) {
    compute_end = timing.codelet_end_time;
  }

  tracer.log_batch_compute_span(BatchingTraceLogger::BatchComputeLogArgs{
      .batch_id = job->submission_id(),
      .model_name = job->model_name(),
      .batch_size = resolve_batch_size(runner, job),
      .worker_id = job->get_worker_id(),
      .worker_type = job->get_executed_on(),
      .codelet_times =
          BatchingTraceLogger::TimeRange{compute_start, compute_end},
      .is_warmup = warmup_job,
      .device_id = job->get_device_id(),
  });
}

void
ResultDispatcher::invoke_previous_callback(
    const std::function<void(std::vector<torch::Tensor>&&, double)>& previous,
    std::vector<torch::Tensor>& results, double latency_ms)
{
  if (!previous) {
    return;
  }
  previous(std::move(results), latency_ms);
}
}  // namespace starpu_server
