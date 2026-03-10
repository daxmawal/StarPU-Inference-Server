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
#include "batching_helpers.hpp"
#include "exceptions.hpp"
#include "inference_task.hpp"
#include "logger.hpp"
#include "monitoring/metrics.hpp"
#include "result_dispatcher_component.hpp"
#include "slot_manager_component.hpp"
#include "starpu_task_worker_prepared_job_processor.hpp"
#include "starpu_task_worker_submit_pipeline.hpp"
#include "starpu_vector_resize_utils.hpp"
#include "task_runner_internal.hpp"
#include "utils/batching_trace_logger.hpp"
#include "utils/exception_classification.hpp"
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

inline namespace starpu_task_worker_detail {
// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
template <typename Hook>
class TestHookSlot {
 public:
  void set(Hook hook) { hook_ = std::move(hook); }

  void reset() { hook_ = Hook{}; }

  template <typename... Args>
  void invoke(Args&&... args) const
  {
    if (hook_) {
      hook_(std::forward<Args>(args)...);
    }
  }

  template <typename... Args>
  void invoke_once(Args&&... args)
  {
    if (!hook_) {
      return;
    }
    auto one_shot_hook = std::move(hook_);
    hook_ = Hook{};
    one_shot_hook(std::forward<Args>(args)...);
  }

 private:
  Hook hook_{};
};

auto
submit_inference_task_hook_storage() -> TestHookSlot<std::function<void()>>&
{
  static TestHookSlot<std::function<void()>> hook;
  return hook;
}

auto
run_after_batching_thread_start_hook_storage()
    -> TestHookSlot<std::function<void()>>&
{
  static TestHookSlot<std::function<void()>> hook;
  return hook;
}

auto
run_before_submit_hook_storage() -> TestHookSlot<std::function<void()>>&
{
  static TestHookSlot<std::function<void()>> hook;
  return hook;
}

auto
duplicate_batching_thread_exception_capture_for_test() -> std::atomic<bool>&
{
  static std::atomic<bool> enabled{false};
  return enabled;
}
#endif  // SONAR_IGNORE_END
// GCOVR_EXCL_STOP
}  // namespace starpu_task_worker_detail

// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
void
set_submit_inference_task_hook(std::function<void()> hook)
{
  submit_inference_task_hook_storage().set(std::move(hook));
}

void
reset_submit_inference_task_hook()
{
  submit_inference_task_hook_storage().reset();
}

void
set_duplicate_batching_thread_exception_capture_for_test(bool enable)
{
  duplicate_batching_thread_exception_capture_for_test().store(
      enable, std::memory_order_release);
}

void
reset_duplicate_batching_thread_exception_capture_for_test()
{
  set_duplicate_batching_thread_exception_capture_for_test(false);
}

void
set_run_after_batching_thread_start_hook(std::function<void()> hook)
{
  run_after_batching_thread_start_hook_storage().set(std::move(hook));
}

void
reset_run_after_batching_thread_start_hook()
{
  run_after_batching_thread_start_hook_storage().reset();
}

void
set_run_before_submit_hook(std::function<void()> hook)
{
  run_before_submit_hook_storage().set(std::move(hook));
}

void
reset_run_before_submit_hook()
{
  run_before_submit_hook_storage().reset();
}
#endif  // SONAR_IGNORE_END
// GCOVR_EXCL_STOP

void
invoke_submit_inference_task_hook()
{
// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  submit_inference_task_hook_storage().invoke();
#endif  // SONAR_IGNORE_END
  // GCOVR_EXCL_STOP
}

// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
static auto
consume_duplicate_batching_thread_exception_capture_for_test() -> bool
{
  return duplicate_batching_thread_exception_capture_for_test().exchange(
      false, std::memory_order_acq_rel);
}
#endif  // SONAR_IGNORE_END
// GCOVR_EXCL_STOP

void
invoke_run_after_batching_thread_start_hook()
{
// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  run_after_batching_thread_start_hook_storage().invoke_once();
#endif  // SONAR_IGNORE_END
  // GCOVR_EXCL_STOP
}

void
invoke_run_before_submit_hook()
{
// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  run_before_submit_hook_storage().invoke_once();
#endif  // SONAR_IGNORE_END
  // GCOVR_EXCL_STOP
}

}  // namespace task_runner_internal

inline namespace starpu_task_worker_detail {

inline auto
is_warmup_job(const std::shared_ptr<InferenceJob>& job) -> bool
{
  return job && job->get_fixed_worker_id().has_value();
}

struct RunThreadExceptionState {
  std::mutex mutex;
  std::exception_ptr exception;
  std::string thread_name;

  void capture(std::string_view source_thread, const std::exception_ptr& caught)
  {
    std::lock_guard lock(mutex);
    if (exception != nullptr) {
      return;
    }
    exception = caught;
    thread_name = source_thread;
  }

  auto take() -> std::pair<std::exception_ptr, std::string>
  {
    std::lock_guard lock(mutex);
    return {exception, std::move(thread_name)};
  }
};

void
rethrow_runner_thread_exception_if_any(RunThreadExceptionState& state)
{
  auto [captured_exception, captured_thread_name] = state.take();
  if (captured_exception == nullptr) {
    return;
  }

  try {
    std::rethrow_exception(captured_exception);
  }
  catch (const std::exception& e) {
    throw WorkerThreadException(std::format(
        "Unhandled exception escaped '{}' thread: {}", captured_thread_name,
        e.what()));
  }
  catch (...) {
    throw WorkerThreadException(std::format(
        "Unhandled non-standard exception escaped '{}' thread.",
        captured_thread_name));
  }
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

}  // namespace starpu_task_worker_detail

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
  ResultDispatcher::prepare_job_completion_callback(*this, job);
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
  const std::string_view reason = [&exception]() {
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
    ResultDispatcher::release_inflight_slot(inflight_state_);
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

auto
submit_exception_log_prefix(ExceptionCategory category) -> std::string_view
{
  using enum starpu_server::ExceptionCategory;
  switch (category) {
    case InferenceEngine:
      return "";
    case RuntimeError:
      return "Unexpected runtime error";
    case LogicError:
      return "Unexpected logic error";
    case BadAlloc:
      return "Memory allocation failure";
    case StdException:
      return "Unexpected std::exception";
    case Unknown:
      return {};
  }
  return {};
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
  catch (...) {
    classify_and_handle_exception(
        std::current_exception(),
        [this, job, job_id = submission_info.job_id](
            ExceptionCategory category, const std::exception* exception) {
          if (exception == nullptr || category == ExceptionCategory::Unknown) {
            finalize_job_after_unknown_exception(
                job, "Unexpected non-standard exception", job_id);
            return;
          }
          finalize_job_after_exception(
              job, *exception, submit_exception_log_prefix(category), job_id);
        });
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
  ResultDispatcher::release_inflight_slot(inflight_state_);
  result_dispatcher_->finalize_job_completion(job);
}

struct StarPUTaskRunner::RunPipelineContext {
  RunThreadExceptionState thread_exception_state;
};

void
StarPUTaskRunner::setup_run_pipeline(RunPipelineContext& /*context*/)
{
  const std::scoped_lock lock(prepared_state_.mutex);
  prepared_state_.jobs.clear();
  prepared_state_.batching_done = false;
  set_starpu_prepared_queue_depth(0);
  set_batch_pending_jobs(0);
}

void
StarPUTaskRunner::abort_run_pipeline(RunPipelineContext& /*context*/) noexcept
{
  if (queue_ != nullptr) {
    queue_->shutdown();
  }
  {
    const std::scoped_lock lock(prepared_state_.mutex);
    prepared_state_.batching_done = true;
  }
  prepared_state_.cv.notify_all();
}

void
StarPUTaskRunner::launch_batching_thread(RunPipelineContext& context)
{
  auto capture_thread_exception = [&context](
                                      std::string_view thread_name,
                                      const std::exception_ptr& exception) {
    context.thread_exception_state.capture(thread_name, exception);
  };

  batching_thread_ = std::jthread([this, &context, capture_thread_exception]() {
    try {
      batching_loop();
    }
    catch (...) {
      auto current_exception = std::current_exception();
      capture_thread_exception("starpu-batching", current_exception);
// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
      if (task_runner_internal::
              consume_duplicate_batching_thread_exception_capture_for_test()) {
        capture_thread_exception(
            "starpu-batching-secondary",
            std::make_exception_ptr(
                std::runtime_error("secondary batching thread failure")));
      }
#endif  // SONAR_IGNORE_END
        // GCOVR_EXCL_STOP
      abort_run_pipeline(context);
    }
  });
  task_runner_internal::invoke_run_after_batching_thread_start_hook();
}

void
StarPUTaskRunner::drain_prepared_jobs_pipeline()
{
  while (true) {
    auto job = wait_for_prepared_job();
    if (!job) {
      break;
    }
    process_prepared_job(job);
  }
}

void
StarPUTaskRunner::finish_run_pipeline(RunPipelineContext& context)
{
  if (batching_thread_.joinable()) {
    batching_thread_.join();
  }
  rethrow_runner_thread_exception_if_any(context.thread_exception_state);
}

void
StarPUTaskRunner::run()
{
  RunPipelineContext context{};

  try {
    log_info(opts_->verbosity, "StarPUTaskRunner started.");
    setup_run_pipeline(context);
    launch_batching_thread(context);
    drain_prepared_jobs_pipeline();
    finish_run_pipeline(context);
    log_info(opts_->verbosity, "StarPUTaskRunner stopped.");
  }
  catch (...) {
    abort_run_pipeline(context);
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
