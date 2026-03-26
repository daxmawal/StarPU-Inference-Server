#include "result_dispatcher_component.hpp"

#include <algorithm>
#include <chrono>
#include <format>
#include <new>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "exceptions.hpp"
#include "logger.hpp"
#include "monitoring/congestion_monitor.hpp"
#include "monitoring/metrics.hpp"
#include "monitoring/runtime_observability.hpp"
#include "slot_manager_component.hpp"
#include "task_runner_internal.hpp"
#include "utils/batching_trace_logger.hpp"
#include "utils/device_type.hpp"
#include "utils/exception_logging.hpp"
#include "utils/perf_observer.hpp"

namespace starpu_server {

using clock = task_runner_internal::Clock;

inline namespace result_dispatcher_component_detail {

inline auto
is_warmup_job(const std::shared_ptr<InferenceJob>& job) -> bool
{
  return job && job->get_fixed_worker_id().has_value();
}

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
auto
prepare_job_completion_callback_test_hooks()
    -> ResultDispatcher::PrepareJobCompletionCallbackTestHooks&
{
  static ResultDispatcher::PrepareJobCompletionCallbackTestHooks hooks{};
  return hooks;
}
#endif  // SONAR_IGNORE_END

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

auto
active_congestion_monitor(const std::shared_ptr<RuntimeObservability>&
                              observability) -> congestion::Monitor*
{
  return observability != nullptr ? observability->congestion_monitor.get()
                                  : nullptr;
}

auto
active_batch_tracer(const std::shared_ptr<RuntimeObservability>& observability)
    -> BatchingTraceLogger*
{
  auto* tracer = active_tracer(observability);
  return tracer != nullptr ? tracer : &BatchingTraceLogger::instance();
}

struct BatchMetricsCounts {
  std::size_t batch_size{0};
  std::size_t logical_jobs{0};
};

struct WorkerExecutionIdentity {
  int worker_id{-1};
  int device_id{-1};
  std::string_view worker_type_label;
};

void
observe_batch_metrics(
    MetricsRecorder* metrics, const BatchMetricsCounts& counts)
{
  if (metrics != nullptr) {
    metrics->observe_batch_size(counts.batch_size);
    metrics->observe_logical_batch_size(counts.logical_jobs);
    return;
  }
  observe_batch_size(counts.batch_size);
  observe_logical_batch_size(counts.logical_jobs);
}

void
record_completed_inference(
    MetricsRecorder* metrics, std::string_view model_name,
    std::size_t logical_jobs)
{
  if (metrics != nullptr) {
    metrics->increment_inference_completed(model_name, logical_jobs);
    return;
  }
  increment_inference_completed(model_name, logical_jobs);
}

auto
resolve_codelet_runtime_ms(const detail::TimingInfo& timing)
    -> std::optional<double>
{
  const auto codelet_start = timing.codelet_start_time;
  const auto codelet_end = timing.codelet_end_time;
  if (codelet_start == clock::time_point{} ||
      codelet_end == clock::time_point{} || codelet_end <= codelet_start) {
    return std::nullopt;
  }
  return std::chrono::duration<double, std::milli>(codelet_end - codelet_start)
      .count();
}

void
observe_task_runtime_metrics(
    MetricsRecorder* metrics, const WorkerExecutionIdentity& worker,
    double task_runtime_ms)
{
  if (metrics != nullptr) {
    metrics->observe_starpu_task_runtime(task_runtime_ms);
    metrics->observe_task_runtime_by_worker(
        worker.worker_id, worker.device_id, worker.worker_type_label,
        task_runtime_ms);
    return;
  }
  observe_starpu_task_runtime(task_runtime_ms);
  observe_task_runtime_by_worker(
      worker.worker_id, worker.device_id, worker.worker_type_label,
      task_runtime_ms);
}

auto
resolve_compute_time_range(const detail::TimingInfo& timing)
    -> BatchingTraceLogger::TimeRange
{
  auto compute_start = timing.inference_start_time;
  if (compute_start == clock::time_point{}) {
    compute_start = timing.codelet_start_time;
  }
  auto compute_end = timing.callback_start_time;
  if (compute_end == clock::time_point{} || compute_end < compute_start) {
    compute_end = timing.codelet_end_time;
  }
  return BatchingTraceLogger::TimeRange{compute_start, compute_end};
}

auto
has_valid_time_range(const BatchingTraceLogger::TimeRange& time_range) -> bool
{
  return time_range.start != clock::time_point{} &&
         time_range.end > time_range.start;
}

void
observe_compute_latency_metrics(
    MetricsRecorder* metrics, const WorkerExecutionIdentity& worker,
    const BatchingTraceLogger::TimeRange& compute_time_range)
{
  if (!has_valid_time_range(compute_time_range)) {
    return;
  }
  const double compute_ms =
      std::chrono::duration<double, std::milli>(
          compute_time_range.end - compute_time_range.start)
          .count();
  if (metrics != nullptr) {
    metrics->observe_compute_latency_by_worker(
        worker.worker_id, worker.device_id, worker.worker_type_label,
        compute_ms);
    return;
  }
  observe_compute_latency_by_worker(
      worker.worker_id, worker.device_id, worker.worker_type_label, compute_ms);
}

void
record_completion_metrics(
    congestion::Monitor* monitor, std::size_t logical_jobs,
    const detail::BaseLatencyBreakdown& breakdown, double latency_ms)
{
  const auto latencies = congestion::CompletionLatencies{
      .queue_latency_ms = breakdown.queue_ms,
      .e2e_latency_ms = latency_ms,
  };
  if (monitor != nullptr) {
    monitor->record_completion(logical_jobs, latencies);
    return;
  }
  congestion::record_completion(logical_jobs, latencies);
}

auto
current_congestion_state(
    const std::shared_ptr<RuntimeObservability>& observability) -> bool
{
  if (auto* monitor = active_congestion_monitor(observability);
      monitor != nullptr) {
    return monitor->congested();
  }
  return congestion::is_congested();
}

void
log_batch_summary_if_enabled(
    BatchingTraceLogger* tracer,
    const std::shared_ptr<RuntimeObservability>& observability,
    const std::shared_ptr<InferenceJob>& job,
    const detail::BaseLatencyBreakdown& breakdown, int job_id,
    std::size_t batch_size, bool warmup)
{
  if (!tracer->enabled()) {
    return;
  }
  const auto request_ids =
      task_runner_internal::build_request_ids_for_trace(job);
  const auto request_arrivals =
      task_runner_internal::build_request_arrival_us_for_trace(job);
  tracer->log_batch_summary(BatchingTraceLogger::BatchSummaryLogArgs{
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
      .is_warmup = warmup,
      .congested = current_congestion_state(observability),
  });
}

}  // namespace result_dispatcher_component_detail

ResultDispatcher::ResultDispatcher(
    const RuntimeConfig* opts, std::atomic<std::size_t>* completed_jobs,
    std::condition_variable* all_done_cv,
    std::shared_ptr<RuntimeObservability> observability)
    : opts_(opts), completed_jobs_(completed_jobs), all_done_cv_(all_done_cv),
      observability_(std::move(observability))
{
}

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
void
ResultDispatcher::SetPrepareJobCompletionCallbackTestHooks(
    PrepareJobCompletionCallbackTestHooks hooks)
{
  prepare_job_completion_callback_test_hooks() = std::move(hooks);
}

void
ResultDispatcher::ClearPrepareJobCompletionCallbackTestHooks()
{
  prepare_job_completion_callback_test_hooks() =
      PrepareJobCompletionCallbackTestHooks{};
}
#endif  // SONAR_IGNORE_END

void
ResultDispatcher::dispatch_terminal_completion(
    const std::shared_ptr<ResultDispatcher>& dispatcher,
    const InferenceJob::CompletionCallback& prev_callback,
    const std::shared_ptr<InferenceJob>& job,
    const std::shared_ptr<StarPUTaskRunner::InflightState>& inflight_state,
    std::vector<torch::Tensor>& results, double latency_ms)
{
  if (job == nullptr || !job->try_mark_terminal_handled()) {
    return;
  }

  try {
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
    auto& test_hooks = prepare_job_completion_callback_test_hooks();
    if (test_hooks.before_dispatch) {
      test_hooks.before_dispatch();
    }
#endif  // SONAR_IGNORE_END
    if (dispatcher != nullptr) {
      dispatcher->handle_job_completion(
          job, prev_callback, results, latency_ms);
    } else {
      ResultDispatcher::cleanup_terminal_job_payload(job);
    }
  }
  catch (const std::exception& e) {
    log_error(std::format(
        "Unhandled exception in terminal completion path: {}", e.what()));
    ResultDispatcher::cleanup_terminal_job_payload(job);
  }
  catch (...) {
    log_error("Unhandled non-std exception in terminal completion path");
    ResultDispatcher::cleanup_terminal_job_payload(job);
  }

  ResultDispatcher::release_inflight_slot(dispatcher, inflight_state);
  if (dispatcher != nullptr) {
    dispatcher->finalize_job_completion(job);
  } else {
    log_error(
        "Missing ResultDispatcher in terminal completion path; "
        "completion counter may be inconsistent");
  }
}

void
ResultDispatcher::prepare_job_completion_callback(
    const std::shared_ptr<ResultDispatcher>& dispatcher,
    const std::shared_ptr<StarPUTaskRunner::InflightState>& inflight_state,
    const std::shared_ptr<InferenceJob>& job)
{
  if (job == nullptr) {
    return;
  }
  auto prev_callback = job->get_on_complete();
  job->set_on_complete(
      [dispatcher, prev_callback, job_sptr = job, inflight_state](
          std::vector<torch::Tensor> results, double latency_ms) mutable {
        ResultDispatcher::dispatch_terminal_completion(
            dispatcher, prev_callback, job_sptr, inflight_state, results,
            latency_ms);
      });
}

void
ResultDispatcher::trace_batch_if_enabled(
    const std::shared_ptr<InferenceJob>& job, bool warmup_job,
    int submission_id) const
{
  auto* tracer = active_tracer(observability_);
  if (tracer == nullptr) {
    tracer = &BatchingTraceLogger::instance();
  }
  if (!tracer->enabled()) {
    return;
  }

  const auto batch_size =
      std::max<std::size_t>(std::size_t{1}, resolve_batch_size(opts_, job));
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

  tracer->log_batch_enqueue_span(
      submission_id, job->model_name(), batch_size,
      BatchingTraceLogger::TimeRange{enqueue_start, enqueue_end},
      request_ids_span, warmup_job);
  tracer->log_batch_build_span(
      submission_id, job->model_name(), batch_size,
      BatchingTraceLogger::TimeRange{
          timing.batch_collect_start_time, timing.batch_collect_end_time},
      request_ids_span, warmup_job);
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
  const int submission_id = job->submission_id();
  job->update_timing_info([submission_id](detail::TimingInfo& timing) {
    ResultDispatcher::ensure_callback_timing(timing);
    timing.submission_id = submission_id;
  });
  const auto timing = job->timing_info_snapshot();
  const auto worker = WorkerExecutionIdentity{
      .worker_id = job->get_worker_id(),
      .device_id = job->get_device_id(),
      .worker_type_label = std::string_view(to_string(job->get_executed_on())),
  };
  const bool warmup = is_warmup_job(job);
  auto* metrics = active_metrics(observability_);
  const auto logical_jobs =
      static_cast<std::size_t>(std::max(1, job->logical_job_count()));
  observe_batch_metrics(
      metrics, BatchMetricsCounts{
                   .batch_size = batch_size,
                   .logical_jobs = logical_jobs,
               });
  const auto breakdown =
      detail::compute_latency_breakdown(timing, latency.count());
  if (!warmup) {
    record_completed_inference(metrics, job->model_name(), logical_jobs);
    if (const auto task_runtime_ms = resolve_codelet_runtime_ms(timing);
        task_runtime_ms.has_value()) {
      observe_task_runtime_metrics(metrics, worker, *task_runtime_ms);
    }
    observe_compute_latency_metrics(
        metrics, worker, resolve_compute_time_range(timing));
    record_completion_metrics(
        active_congestion_monitor(observability_), logical_jobs, breakdown,
        latency.count());
  }
  perf_observer::record_job(
      timing.enqueued_time, timing.callback_end_time, batch_size, warmup);

  const int job_id = submission_id >= 0 ? submission_id : job->get_request_id();
  log_job_timings(job_id, latency, timing);

  log_batch_summary_if_enabled(
      active_batch_tracer(observability_), observability_, job, breakdown,
      job_id, batch_size, warmup);
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
  if (job == nullptr || completed_jobs_ == nullptr || all_done_cv_ == nullptr) {
    return;
  }
  const auto logical_jobs =
      static_cast<std::size_t>(std::max(1, job->logical_job_count()));
  completed_jobs_->fetch_add(logical_jobs, std::memory_order_release);
  all_done_cv_->notify_all();
}

auto
ResultDispatcher::handle_job_exception(
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

void
ResultDispatcher::handle_cancelled_job(
    const std::shared_ptr<ResultDispatcher>& dispatcher,
    const std::shared_ptr<StarPUTaskRunner::InflightState>& inflight_state,
    const std::shared_ptr<InferenceJob>& job)
{
  if (job == nullptr || !job->try_mark_terminal_handled()) {
    return;
  }

  static_cast<void>(job->take_on_complete());
  clear_pending_sub_job_callbacks(job);

  auto pending_jobs = job->take_pending_sub_jobs();
  SlotManager::release_pending_jobs(job, pending_jobs);

  clear_batching_state(job);
  cleanup_terminal_job_payload(job);
  release_inflight_slot(dispatcher, inflight_state);

  if (dispatcher != nullptr) {
    dispatcher->finalize_job_completion(job);
  }
}

void
ResultDispatcher::finalize_job_after_exception(
    const std::shared_ptr<ResultDispatcher>& dispatcher,
    const std::shared_ptr<StarPUTaskRunner::InflightState>& inflight_state,
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
  if (dispatcher != nullptr &&
      active_metrics(dispatcher->observability_) != nullptr) {
    active_metrics(dispatcher->observability_)
        ->increment_inference_failure("execution", reason, model_label);
  } else {
    increment_inference_failure("execution", reason, model_label);
  }

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

  static_cast<void>(handle_job_exception(job, exception));

  if (!job) {
    return;
  }

  clear_pending_sub_job_callbacks(job);
  auto pending_jobs = job->take_pending_sub_jobs();
  SlotManager::release_pending_jobs(job, pending_jobs);
  clear_batching_state(job);
  cleanup_terminal_job_payload(job);

  if (!job->try_mark_terminal_handled()) {
    return;
  }

  if (dispatcher != nullptr) {
    dispatcher->finalize_job_completion(job);
  } else {
    log_error(
        "Missing ResultDispatcher in terminal completion path; "
        "completion counter may be inconsistent");
  }
  release_inflight_slot(dispatcher, inflight_state);
}

void
ResultDispatcher::finalize_job_after_unknown_exception(
    const std::shared_ptr<ResultDispatcher>& dispatcher,
    const std::shared_ptr<StarPUTaskRunner::InflightState>& inflight_state,
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
  finalize_job_after_exception(
      dispatcher, inflight_state, job, unknown, log_prefix, job_id);
}

void
ResultDispatcher::handle_job_completion(
    const std::shared_ptr<InferenceJob>& job,
    const InferenceJob::CompletionCallback& prev_callback,
    std::vector<torch::Tensor>& results, double latency_ms) const
{
  run_with_logged_exceptions(
      [this, prev_callback, job, &results, latency_ms] {
        if (!job) {
          return;
        }
        static_cast<void>(job->release_input_tensors());
        record_job_metrics(
            job, StarPUTaskRunner::DurationMs{latency_ms},
            ResultDispatcher::resolve_batch_size(opts_, job));
        emit_batch_traces(job);
        run_with_logged_exceptions(
            [prev_callback, &results, latency_ms] {
              ResultDispatcher::invoke_previous_callback(
                  prev_callback, results, latency_ms);
            },
            ExceptionLoggingMessages{
                "Exception in completion callback: ",
                "Unknown exception in completion callback"});
      },
      ExceptionLoggingMessages{
          "Exception while finalizing job completion: ",
          "Unknown exception while finalizing job completion"});
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

  const auto aggregated_timing = aggregated_job->timing_info_snapshot();
  const auto aggregated_device_id = aggregated_job->get_device_id();
  const auto aggregated_worker_id = aggregated_job->get_worker_id();
  const auto aggregated_executed_on = aggregated_job->get_executed_on();
  const auto aggregated_submission_id = aggregated_job->submission_id();

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

    job_sp->set_timing_info(aggregated_timing);
    job_sp->set_executed_on(aggregated_executed_on);
    job_sp->set_device_id(aggregated_device_id);
    job_sp->set_worker_id(aggregated_worker_id);
    job_sp->set_submission_id(aggregated_submission_id);

    run_with_logged_exceptions(
        [&entry, &outputs, latency_ms] {
          if (entry.callback) {
            entry.callback(outputs, latency_ms);
          }
        },
        ExceptionLoggingMessages{
            "Exception in sub-job completion callback: ",
            "Unknown exception in sub-job completion callback"});

    outputs.clear();
    cleanup_terminal_job_payload(job_sp);

    offset += static_cast<std::size_t>(
        std::max<int64_t>(1, slice_result.processed_length));
  }

  clear_pending_sub_job_callbacks(aggregated_job);
  clear_batching_state(aggregated_job);
  cleanup_terminal_job_payload(aggregated_job);
}

auto
ResultDispatcher::resolve_batch_size(
    const RuntimeConfig* opts,
    const std::shared_ptr<InferenceJob>& job) -> std::size_t
{
  return std::max<std::size_t>(
      std::size_t{1},
      static_cast<std::size_t>(
          task_runner_internal::resolve_batch_size_for_job(opts, job)));
}

void
ResultDispatcher::emit_batch_traces(
    const std::shared_ptr<InferenceJob>& job) const
{
  if (!job) {
    return;
  }
  auto* tracer = active_batch_tracer(observability_);
  if (!tracer->enabled()) {
    return;
  }
  const bool warmup_job = is_warmup_job(job);
  const auto timing = job->timing_info_snapshot();
  const auto compute_time_range = resolve_compute_time_range(timing);

  tracer->log_batch_compute_span(BatchingTraceLogger::BatchComputeLogArgs{
      .batch_id = job->submission_id(),
      .model_name = job->model_name(),
      .batch_size = resolve_batch_size(opts_, job),
      .worker_id = job->get_worker_id(),
      .worker_type = job->get_executed_on(),
      .codelet_times = compute_time_range,
      .is_warmup = warmup_job,
      .device_id = job->get_device_id(),
  });
}

void
ResultDispatcher::release_inflight_slot(
    const std::shared_ptr<ResultDispatcher>& dispatcher,
    const std::shared_ptr<StarPUTaskRunner::InflightState>& inflight_state)
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
  if (dispatcher != nullptr && dispatcher->observability_ != nullptr &&
      dispatcher->observability_->metrics != nullptr) {
    dispatcher->observability_->metrics->set_inflight_tasks(previous - 1);
  } else {
    set_inflight_tasks(previous - 1);
  }
  if (inflight_state->max_tasks > 0 && previous > 0) {
    const double ratio = static_cast<double>(previous - 1) /
                         static_cast<double>(inflight_state->max_tasks);
    if (dispatcher != nullptr && dispatcher->observability_ != nullptr &&
        dispatcher->observability_->metrics != nullptr) {
      dispatcher->observability_->metrics->set_starpu_worker_busy_ratio(ratio);
    } else {
      set_starpu_worker_busy_ratio(ratio);
    }
  }
  std::scoped_lock lock(inflight_state->mutex);
  inflight_state->cv.notify_one();
}

void
ResultDispatcher::invoke_previous_callback(
    const InferenceJob::CompletionCallback& previous,
    std::vector<torch::Tensor>& results, double latency_ms)
{
  if (!previous) {
    return;
  }
  previous(std::move(results), latency_ms);
}

void
ResultDispatcher::clear_pending_sub_job_callbacks(
    const std::shared_ptr<InferenceJob>& job)
{
  if (job == nullptr) {
    return;
  }
  const auto& pending = job->pending_sub_jobs();
  for (const auto& sub_job : pending) {
    if (sub_job != nullptr) {
      static_cast<void>(sub_job->take_on_complete());
    }
  }
}

void
ResultDispatcher::clear_batching_state(const std::shared_ptr<InferenceJob>& job)
{
  if (job == nullptr) {
    return;
  }
  job->set_aggregated_sub_jobs({});
  job->clear_pending_sub_jobs();
}

void
ResultDispatcher::cleanup_terminal_job_payload(
    const std::shared_ptr<InferenceJob>& job)
{
  if (job == nullptr) {
    return;
  }
  static_cast<void>(job->release_input_tensors());
  job->release_input_memory_holders();
  job->set_output_tensors({});
}

}  // namespace starpu_server
