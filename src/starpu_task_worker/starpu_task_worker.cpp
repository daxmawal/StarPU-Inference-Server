#include "starpu_task_worker.hpp"

#include <starpu.h>
#include <starpu_data_interfaces.h>
#include <torch/torch.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <format>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "batch_collector_component.hpp"
#include "exceptions.hpp"
#include "inference_task.hpp"
#include "logger.hpp"
#include "monitoring/metrics.hpp"
#include "result_dispatcher_component.hpp"
#include "slot_manager_component.hpp"
#include "starpu_vector_resize_utils.hpp"
#include "task_runner_internal.hpp"
#include "utils/batching_trace_logger.hpp"
#include "utils/exception_logging.hpp"
#include "utils/nvtx.hpp"

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
job_identifier(const InferenceJob& job) -> int
{
  const int submission_id = job.submission_id();
  return (submission_id >= 0) ? submission_id : job.get_request_id();
}

namespace {
// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
auto
submit_inference_task_hook_storage() -> std::function<void()>&
{
  static std::function<void()> hook;
  return hook;
}
#endif  // SONAR_IGNORE_END
// GCOVR_EXCL_STOP
}  // namespace

// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
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
#endif  // SONAR_IGNORE_END
// GCOVR_EXCL_STOP

static void
invoke_submit_inference_task_hook()
{
// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  const auto& hook = submit_inference_task_hook_storage();
  if (hook) {
    hook();
  }
#endif  // SONAR_IGNORE_END
  // GCOVR_EXCL_STOP
}

#include "batching_helpers.inl"

}  // namespace task_runner_internal

namespace {

inline auto
is_warmup_job(const std::shared_ptr<InferenceJob>& job) -> bool
{
  return job && job->get_fixed_worker_id().has_value();
}

void
log_batch_submitted_if_enabled(
    const std::shared_ptr<InferenceJob>& job, bool warmup_job)
{
  if (!job) {
    return;
  }
  auto& tracer = BatchingTraceLogger::instance();
  if (!tracer.enabled()) {
    return;
  }

  const auto request_ids =
      task_runner_internal::build_request_ids_for_trace(job);
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
    const task_runner_internal::VectorResizeSpec spec{
        static_cast<std::size_t>(tensor.numel()), tensor.nbytes()};
    task_runner_internal::resize_starpu_vector_handle(
        handles[idx], spec, false);
  }
}

}  // namespace

using clock = task_runner_internal::Clock;

// =============================================================================
// Constructor
// =============================================================================

StarPUTaskRunner::StarPUTaskRunner(const StarPUTaskRunnerConfig& config)
    : queue_(config.queue), model_cpu_(config.model_cpu),
      models_gpu_(config.models_gpu), starpu_(config.starpu),
      opts_(config.opts), completed_jobs_(config.completed_jobs),
      all_done_cv_(config.all_done_cv),
      dependencies_(
          config.dependencies != nullptr ? *config.dependencies
                                         : kDefaultInferenceTaskDependencies),
      inflight_state_(std::make_shared<InflightState>()),
      slot_manager_(std::make_unique<SlotManager>(starpu_, opts_)),
      result_dispatcher_(std::make_shared<ResultDispatcher>(
          opts_, completed_jobs_, all_done_cv_))
{
  task_runner_internal::validate_not_null(queue_, "queue");
  task_runner_internal::validate_not_null(model_cpu_, "model_cpu");
  task_runner_internal::validate_not_null(models_gpu_, "models_gpu");
  task_runner_internal::validate_not_null(starpu_, "starpu");
  task_runner_internal::validate_not_null(opts_, "opts");
  task_runner_internal::validate_not_null(completed_jobs_, "completed_jobs");
  task_runner_internal::validate_not_null(all_done_cv_, "all_done_cv");

  inflight_state_->max_tasks =
      opts_ != nullptr ? opts_->batching.max_inflight_tasks : 0;
  set_max_inflight_tasks(inflight_state_->max_tasks);
  set_inflight_tasks(inflight_state_->tasks.load(std::memory_order_relaxed));
  if (inflight_state_->max_tasks > 0) {
    set_starpu_worker_busy_ratio(0.0);
  }

  batch_collector_ = std::make_unique<BatchCollector>(
      queue_, opts_, starpu_, &pending_job_,
      PreparedBatchingContext{
          .prepared_mutex = &prepared_state_.mutex,
          .prepared_cv = &prepared_state_.cv,
          .prepared_jobs = &prepared_state_.jobs,
          .batching_done = &prepared_state_.batching_done,
      },
      InflightContext{
          .inflight_tasks = &inflight_state_->tasks,
          .inflight_cv = &inflight_state_->cv,
          .inflight_mutex = &inflight_state_->mutex,
          .max_inflight_tasks = inflight_state_->max_tasks,
      });
}

StarPUTaskRunner::~StarPUTaskRunner() = default;


// =============================================================================
// Job Queue Management
// =============================================================================

// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
auto
StarPUTaskRunner::wait_for_next_job() -> std::shared_ptr<InferenceJob>
{
  return batch_collector_->wait_for_next_job();
}
#endif  // SONAR_IGNORE_END
// GCOVR_EXCL_STOP

// =============================================================================
// Completion Callback Handling
// =============================================================================

// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
void
StarPUTaskRunner::log_job_timings(
    int request_id, DurationMs latency,
    const detail::TimingInfo& timing_info) const
{
  result_dispatcher_->log_job_timings(request_id, latency, timing_info);
}
#endif  // SONAR_IGNORE_END
// GCOVR_EXCL_STOP

void
StarPUTaskRunner::prepare_job_completion_callback(
    const std::shared_ptr<InferenceJob>& job)
{
  result_dispatcher_->prepare_job_completion_callback(*this, job);
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
  const auto request_ids =
      task_runner_internal::build_request_ids_for_trace(job);
  const auto request_ids_span = std::span<const int>(request_ids);
  const auto timing = job->timing_info_snapshot();
  const auto enqueue_start = timing.enqueued_time;
  auto enqueue_end = timing.last_enqueued_time;
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
          timing.batch_collect_start_time, timing.batch_collect_end_time},
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

  const std::string model_label = job != nullptr
                                      ? std::string{job->model_name()}
                                      : std::string{"<unknown>"};
  const std::string_view reason = [&exception]() -> std::string_view {
    if (dynamic_cast<const std::bad_alloc*>(&exception) != nullptr) {
      return "bad_alloc";
    }
    if (dynamic_cast<const InferenceEngineException*>(&exception) != nullptr) {
      return "engine_exception";
    }
    if (dynamic_cast<const std::logic_error*>(&exception) != nullptr) {
      return "logic_error";
    }
    if (dynamic_cast<const std::runtime_error*>(&exception) != nullptr) {
      return "runtime_error";
    }
    return "exception";
  }();
  increment_inference_failure("execution", reason, model_label);

  if (job) {
    InferenceJob::FailureInfo failure_info{};
    failure_info.stage = "execution";
    failure_info.reason = std::string(reason);
    if (!log_prefix.empty()) {
      failure_info.message =
          std::format("{}: {}", log_prefix, exception.what());
    } else {
      failure_info.message = exception.what();
    }
    failure_info.metrics_reported = true;
    job->set_failure_info(std::move(failure_info));
  }

  static_cast<void>(StarPUTaskRunner::handle_job_exception(job, exception));

  if (!job) {
    return;
  }

  ResultDispatcher::clear_pending_sub_job_callbacks(job);
  auto pending_jobs = job->take_pending_sub_jobs();
  SlotManager::release_pending_jobs(job, pending_jobs);
  ResultDispatcher::clear_batching_state(job);
  ResultDispatcher::cleanup_terminal_job_payload(job);

  if (job->try_mark_terminal_handled()) {
    result_dispatcher_->finalize_job_completion(job);
    release_inflight_slot();
  }
}

void
StarPUTaskRunner::finalize_job_after_unknown_exception(
    const std::shared_ptr<InferenceJob>& job, std::string_view log_prefix,
    int job_id)
{
  class UnknownTerminalException final : public std::exception {
   public:
    [[nodiscard]] auto what() const noexcept -> const char* override
    {
      return "Unknown non-standard exception";
    }
  };

  const UnknownTerminalException unknown;
  finalize_job_after_exception(job, unknown, log_prefix, job_id);
}

void
StarPUTaskRunner::reserve_inflight_slot()
{
  if (inflight_state_ == nullptr || inflight_state_->max_tasks == 0) {
    return;
  }

  std::unique_lock lock(inflight_state_->mutex);
  inflight_state_->cv.wait(lock, [this] {
    return inflight_state_->tasks.load(std::memory_order_acquire) <
           inflight_state_->max_tasks;
  });
  const auto current =
      inflight_state_->tasks.fetch_add(1, std::memory_order_release) + 1;
  set_inflight_tasks(current);
  const double ratio = static_cast<double>(current) /
                       static_cast<double>(inflight_state_->max_tasks);
  set_starpu_worker_busy_ratio(ratio);
}

void
StarPUTaskRunner::release_inflight_slot()
{
  release_inflight_slot(inflight_state_);
}

void
StarPUTaskRunner::release_inflight_slot(
    const std::shared_ptr<InflightState>& inflight_state)
{
  if (inflight_state == nullptr || inflight_state->max_tasks == 0) {
    return;
  }
  std::size_t previous = inflight_state->tasks.load(std::memory_order_acquire);
  while (true) {
    if (previous == 0) {
      return;
    }
    if (inflight_state->tasks.compare_exchange_weak(
            previous, previous - 1, std::memory_order_acq_rel)) {
      break;
    }
  }
  set_inflight_tasks(previous - 1);
  if (inflight_state->max_tasks > 0 && previous > 0) {
    const double ratio = static_cast<double>(previous - 1) /
                         static_cast<double>(inflight_state->max_tasks);
    set_starpu_worker_busy_ratio(ratio);
  }
  std::scoped_lock lock(inflight_state->mutex);
  inflight_state->cv.notify_one();
}

// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
namespace task_runner_helpers {
void
ensure_callback_timing(detail::TimingInfo& timing)
{
  ResultDispatcher::ensure_callback_timing(timing);
}

void
record_job_metrics(
    StarPUTaskRunner& runner, const std::shared_ptr<InferenceJob>& job,
    std::chrono::duration<double, std::milli> latency, std::size_t batch_size)
{
  if (runner.result_dispatcher_ != nullptr) {
    runner.result_dispatcher_->record_job_metrics(job, latency, batch_size);
  }
}

void
finalize_job_completion(
    StarPUTaskRunner& runner, const std::shared_ptr<InferenceJob>& job)
{
  if (runner.result_dispatcher_ != nullptr) {
    runner.result_dispatcher_->finalize_job_completion(job);
  }
}
}  // namespace task_runner_helpers
#endif  // SONAR_IGNORE_END
// GCOVR_EXCL_STOP

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
    reserve_inflight_slot();
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
  catch (const std::exception& exception) {
    finalize_job_after_exception(
        job, exception, "Unexpected std::exception", submission_info.job_id);
  }
  catch (...) {
    finalize_job_after_unknown_exception(
        job, "Unexpected non-standard exception", submission_info.job_id);
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
  if (auto completion = job->take_on_complete(); completion) {
    run_with_logged_exceptions(
        [completion = std::move(completion), &completion_invoked]() mutable {
          completion({}, -1);
          completion_invoked = true;
        },
        ExceptionLoggingMessages{
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
  return task_runner_internal::resolve_batch_size_for_job(opts_, job);
}

// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
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
#endif  // SONAR_IGNORE_END
// GCOVR_EXCL_STOP

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

// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
void
StarPUTaskRunner::propagate_completion_to_sub_jobs(
    const std::shared_ptr<InferenceJob>& aggregated_job,
    const std::vector<torch::Tensor>& aggregated_outputs, double latency_ms)
{
  ResultDispatcher::propagate_completion_to_sub_jobs(
      aggregated_job, aggregated_outputs, latency_ms);
}
#endif  // SONAR_IGNORE_END
// GCOVR_EXCL_STOP

auto
StarPUTaskRunner::configure_task_context(
    InferenceTask& task, const PoolResources& pools,
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
        starpu_, job, model_cpu_, models_gpu_, opts_, dependencies_);
    task.submit();
    log_batch_submitted_if_enabled(job, warmup_job);
    return;
  }

  auto pools = acquire_pools();
  struct PoolReleaseGuard {
    explicit PoolReleaseGuard(const PoolResources& pools_in) : pools(pools_in)
    {
    }
    PoolReleaseGuard(const PoolReleaseGuard&) = delete;
    PoolReleaseGuard(PoolReleaseGuard&&) = delete;
    auto operator=(const PoolReleaseGuard&) -> PoolReleaseGuard& = delete;
    auto operator=(PoolReleaseGuard&&) -> PoolReleaseGuard& = delete;
    ~PoolReleaseGuard() noexcept
    {
      if (active) {
        release();
      }
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

    const PoolResources& pools;
    bool active{true};
  };

  PoolReleaseGuard pool_guard(pools);

  const auto batch = validate_batch_and_copy_inputs(job, pools);

  InferenceTask task(
      starpu_, job, model_cpu_, models_gpu_, opts_, dependencies_);

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

  std::vector<starpu_data_handle_t> input_handles_for_ctx;
  if (pools.has_input()) {
    input_handles_for_ctx = *input_handles;
  } else {
    input_handles_for_ctx = std::move(input_handles_storage);
  }

  std::vector<starpu_data_handle_t> output_handles_for_ctx;
  if (pools.has_output()) {
    output_handles_for_ctx = *output_handles;
  } else {
    output_handles_for_ctx = std::move(output_handles_storage);
  }

  auto ctx = configure_task_context(
      task, pools, std::move(input_handles_for_ctx),
      std::move(output_handles_for_ctx), batch);

  starpu_task* task_ptr =
      task.create_task(ctx->inputs_handles, ctx->outputs_handles, ctx);

  const auto submitted_at = MonotonicClock::now();
  job->update_timing_info([submitted_at](detail::TimingInfo& timing) {
    timing.before_starpu_submitted_time = submitted_at;
  });

  const int ret = starpu_task_submit(task_ptr);
  if (ret != 0) {
    pool_guard.dismiss();
    handle_submission_failure(pools, ctx, ret);
  } else {
    pool_guard.dismiss();
    log_batch_submitted_if_enabled(job, warmup_job);
  }
}

// =============================================================================
// Main run loop: pull jobs, submit them, handle shutdown and errors
// =============================================================================

void
StarPUTaskRunner::handle_cancelled_job(const std::shared_ptr<InferenceJob>& job)
{
  if (job == nullptr || !job->try_mark_terminal_handled()) {
    return;
  }

  static_cast<void>(job->take_on_complete());
  ResultDispatcher::clear_pending_sub_job_callbacks(job);

  auto pending_jobs = job->take_pending_sub_jobs();
  SlotManager::release_pending_jobs(job, pending_jobs);

  ResultDispatcher::clear_batching_state(job);
  ResultDispatcher::cleanup_terminal_job_payload(job);
  result_dispatcher_->finalize_job_completion(job);
}

void
StarPUTaskRunner::run()
{
  std::mutex thread_exception_mutex;
  std::exception_ptr thread_exception;
  std::string failing_thread_name;
  auto capture_thread_exception =
      [&thread_exception_mutex, &thread_exception, &failing_thread_name](
          std::string_view thread_name, std::exception_ptr exception) {
        std::lock_guard lock(thread_exception_mutex);
        if (thread_exception != nullptr) {
          return;
        }
        thread_exception = std::move(exception);
        failing_thread_name = thread_name;
      };

  const auto notify_batching_thread_failure = [this]() {
    if (queue_ != nullptr) {
      queue_->shutdown();
    }
    {
      const std::scoped_lock lock(prepared_state_.mutex);
      prepared_state_.batching_done = true;
    }
    prepared_state_.cv.notify_all();
  };

  auto rethrow_thread_exception_if_any =
      [&thread_exception_mutex, &thread_exception, &failing_thread_name]() {
        std::exception_ptr captured_exception;
        std::string captured_thread_name;
        {
          std::lock_guard lock(thread_exception_mutex);
          captured_exception = thread_exception;
          captured_thread_name = failing_thread_name;
        }

        if (captured_exception == nullptr) {
          return;
        }

        try {
          std::rethrow_exception(captured_exception);
        }
        catch (const std::exception& e) {
          throw std::runtime_error(std::format(
              "Unhandled exception escaped '{}' thread: {}",
              captured_thread_name, e.what()));
        }
        catch (...) {
          throw std::runtime_error(std::format(
              "Unhandled non-standard exception escaped '{}' thread.",
              captured_thread_name));
        }
      };

  try {
    log_info(opts_->verbosity, "StarPUTaskRunner started.");

    {
      const std::scoped_lock lock(prepared_state_.mutex);
      prepared_state_.jobs.clear();
      prepared_state_.batching_done = false;
      set_starpu_prepared_queue_depth(0);
      set_batch_pending_jobs(0);
    }

    batching_thread_ = std::jthread([this, &capture_thread_exception,
                                     &notify_batching_thread_failure]() {
      try {
        batching_loop();
      }
      catch (...) {
        capture_thread_exception("starpu-batching", std::current_exception());
        notify_batching_thread_failure();
      }
    });

    while (true) {
      auto job = wait_for_prepared_job();
      if (!job) {
        break;
      }

      int job_id = job->get_request_id();
      try {
        if (job->is_cancelled()) {
          handle_cancelled_job(job);
          continue;
        }

        const auto submission_id = next_submission_id_.fetch_add(1);
        job->set_submission_id(submission_id);
        job->update_timing_info([submission_id](detail::TimingInfo& timing) {
          timing.submission_id = submission_id;
        });

        const int logical_jobs = job->logical_job_count();
        const auto request_id = job->get_request_id();
        job_id = task_runner_internal::job_identifier(*job);
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
        submit_job_or_handle_failure(
            job, SubmissionInfo{submission_id, job_id});
      }
      catch (const std::exception& e) {
        finalize_job_after_exception(
            job, e, "Unexpected exception while processing dequeued job",
            job_id);
      }
      catch (...) {
        finalize_job_after_unknown_exception(
            job,
            "Unexpected non-standard exception while processing dequeued job",
            job_id);
      }
    }

    if (batching_thread_.joinable()) {
      batching_thread_.join();
    }
    rethrow_thread_exception_if_any();

    log_info(opts_->verbosity, "StarPUTaskRunner stopped.");
  }
  catch (...) {
    notify_batching_thread_failure();
    if (batching_thread_.joinable()) {
      batching_thread_.join();
    }
    throw;
  }
}

// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
namespace task_runner_internal::testing {

auto
batch_size_from_inputs(const std::vector<torch::Tensor>& inputs) -> std::size_t
{
  return task_runner_internal::batch_size_from_inputs(inputs);
}

auto
resolve_batch_size_for_job(
    const RuntimeConfig* opts,
    const std::shared_ptr<InferenceJob>& job) -> int64_t
{
  return task_runner_internal::resolve_batch_size_for_job(opts, job);
}

}  // namespace task_runner_internal::testing
#endif  // SONAR_IGNORE_END
// GCOVR_EXCL_STOP
}  // namespace starpu_server
