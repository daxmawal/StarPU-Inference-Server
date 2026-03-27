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

}  // namespace starpu_task_worker_detail

using clock = task_runner_internal::Clock;

// =============================================================================
// Constructor
// =============================================================================

StarPUTaskRunner::StarPUTaskRunner(const StarPUTaskRunnerConfig& config)
    : queue_(config.queue), model_cpu_(config.model_cpu),
      models_gpu_(config.models_gpu),
      gpu_replica_assignments_(config.gpu_replica_assignments),
      starpu_(config.starpu), opts_(config.opts),
      completed_jobs_(config.completed_jobs), all_done_cv_(config.all_done_cv),
      dependencies_([&config]() {
        auto dependencies = config.dependencies != nullptr
                                ? *config.dependencies
                                : kDefaultInferenceTaskDependencies;
        dependencies.observability = config.observability;
        return dependencies;
      }()),
      observability_(config.observability),
      slot_manager_(std::make_unique<SlotManager>(
          starpu_, opts_, model_cpu_, models_gpu_, gpu_replica_assignments_,
          dependencies_, observability_)),
      result_dispatcher_(std::make_shared<ResultDispatcher>(
          opts_, completed_jobs_, all_done_cv_, observability_))
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
  if (observability_ != nullptr && observability_->metrics != nullptr) {
    observability_->metrics->set_max_inflight_tasks(inflight_state_->max_tasks);
    observability_->metrics->set_inflight_tasks(inflight_state_->tasks.load());
  } else {
    set_max_inflight_tasks(inflight_state_->max_tasks);
    set_inflight_tasks(inflight_state_->tasks.load());
  }
  if (inflight_state_->max_tasks > 0) {
    if (observability_ != nullptr && observability_->metrics != nullptr) {
      observability_->metrics->set_starpu_worker_busy_ratio(0.0);
    } else {
      set_starpu_worker_busy_ratio(0.0);
    }
  }

  batch_collector_ = std::make_unique<BatchCollector>(
      queue_, opts_, starpu_, &pending_job_, observability_,
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
  ResultDispatcher::prepare_job_completion_callback(
      result_dispatcher_, inflight_state_, job);
}

void
StarPUTaskRunner::trace_batch_if_enabled(
    const std::shared_ptr<InferenceJob>& job, bool warmup_job,
    int submission_id) const
{
  if (result_dispatcher_ != nullptr) {
    result_dispatcher_->trace_batch_if_enabled(job, warmup_job, submission_id);
  }
}

void
StarPUTaskRunner::finalize_job_after_exception(
    const std::shared_ptr<InferenceJob>& job, const std::exception& exception,
    std::string_view log_prefix, int job_id)
{
  ResultDispatcher::finalize_job_after_exception(
      result_dispatcher_, inflight_state_, job, exception, log_prefix, job_id);
}

void
StarPUTaskRunner::finalize_job_after_unknown_exception(
    const std::shared_ptr<InferenceJob>& job, std::string_view log_prefix,
    int job_id)
{
  ResultDispatcher::finalize_job_after_unknown_exception(
      result_dispatcher_, inflight_state_, job, log_prefix, job_id);
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
  if (auto completion = job->completion().take_on_complete(); completion) {
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

auto
StarPUTaskRunner::configure_task_context(
    InferenceTask& task, const PoolResources& pools,
    std::vector<starpu_data_handle_t> input_handles,
    std::vector<starpu_data_handle_t> output_handles,
    int64_t batch_size) -> std::shared_ptr<InferenceCallbackContext>
{
  return SlotManager::configure_task_context(
      task, pools, std::move(input_handles), std::move(output_handles),
      batch_size);
}

void
StarPUTaskRunner::handle_submission_failure(
    const PoolResources& pools,
    const std::shared_ptr<InferenceCallbackContext>& ctx, int submit_code)
{
  SlotManager::handle_submission_failure(pools, ctx, submit_code);
}

auto
StarPUTaskRunner::resolve_batch_size(
    const std::shared_ptr<InferenceJob>& job) const -> int64_t
{
  return task_runner_internal::resolve_batch_size_for_job(opts_, job);
}

void
StarPUTaskRunner::submit_inference_task(
    const std::shared_ptr<InferenceJob>& job)
{
  slot_manager_->submit_inference_task(job);
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

// =============================================================================
// Main run loop: pull jobs, submit them, handle shutdown and errors
// =============================================================================

void
StarPUTaskRunner::handle_cancelled_job(const std::shared_ptr<InferenceJob>& job)
{
  ResultDispatcher::handle_cancelled_job(
      result_dispatcher_, inflight_state_, job);
}

struct StarPUTaskRunner::RunPipelineContext {
  RunThreadExceptionState thread_exception_state;
};

void
StarPUTaskRunner::setup_run_pipeline(RunPipelineContext& /*context*/)
{
  batch_collector_->reset_prepared_queue_state();
}

void
StarPUTaskRunner::abort_run_pipeline(RunPipelineContext& /*context*/) noexcept
{
  if (queue_ != nullptr) {
    queue_->shutdown();
  }
  batch_collector_->abort_prepared_queue();
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
