#include "result_dispatcher_component.hpp"

#include <algorithm>
#include <chrono>
#include <format>
#include <string_view>
#include <utility>

#include "logger.hpp"
#include "monitoring/congestion_monitor.hpp"
#include "monitoring/metrics.hpp"
#include "task_runner_internal.hpp"
#include "utils/batching_trace_logger.hpp"
#include "utils/device_type.hpp"
#include "utils/exception_logging.hpp"
#include "utils/perf_observer.hpp"

namespace starpu_server {

namespace {

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

}  // namespace

using clock = task_runner_internal::Clock;

ResultDispatcher::ResultDispatcher(
    const RuntimeConfig* opts, std::atomic<std::size_t>* completed_jobs,
    std::condition_variable* all_done_cv)
    : opts_(opts), completed_jobs_(completed_jobs), all_done_cv_(all_done_cv)
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
ResultDispatcher::prepare_job_completion_callback(
    StarPUTaskRunner& runner, const std::shared_ptr<InferenceJob>& job)
{
  auto prev_callback = job->get_on_complete();
  auto dispatcher = runner.result_dispatcher_;
  auto inflight_state = runner.inflight_state_;
  job->set_on_complete(
      [dispatcher, prev_callback, job_sptr = job, inflight_state](
          std::vector<torch::Tensor> results, double latency_ms) mutable {
        if (job_sptr == nullptr || !job_sptr->try_mark_terminal_handled()) {
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
                job_sptr, prev_callback, results, latency_ms);
          } else {
            ResultDispatcher::cleanup_terminal_job_payload(job_sptr);
          }
        }
        catch (const std::exception& e) {
          log_error(std::format(
              "Unhandled exception in terminal completion path: {}", e.what()));
          ResultDispatcher::cleanup_terminal_job_payload(job_sptr);
        }
        catch (...) {
          log_error("Unhandled non-std exception in terminal completion path");
          ResultDispatcher::cleanup_terminal_job_payload(job_sptr);
        }

        ResultDispatcher::release_inflight_slot(inflight_state);
        if (dispatcher != nullptr) {
          dispatcher->finalize_job_completion(job_sptr);
        } else {
          log_error(
              "Missing ResultDispatcher in terminal completion path; "
              "completion counter may be inconsistent");
        }
      });
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
  const auto worker_type_label =
      std::string_view(to_string(job->get_executed_on()));
  const int worker_id = job->get_worker_id();
  const int device_id = job->get_device_id();
  const auto zero_tp = clock::time_point{};
  const bool warmup = is_warmup_job(job);
  observe_batch_size(batch_size);
  const auto logical_jobs =
      static_cast<std::size_t>(std::max(1, job->logical_job_count()));
  observe_logical_batch_size(logical_jobs);
  const auto breakdown =
      detail::compute_latency_breakdown(timing, latency.count());
  if (!warmup) {
    increment_inference_completed(job->model_name(), logical_jobs);
    const auto codelet_end = timing.codelet_end_time;
    if (const auto codelet_start = timing.codelet_start_time;
        codelet_end > codelet_start && codelet_start != clock::time_point{} &&
        codelet_end != clock::time_point{}) {
      const double task_runtime_ms =
          std::chrono::duration<double, std::milli>(codelet_end - codelet_start)
              .count();
      observe_starpu_task_runtime(task_runtime_ms);
      observe_task_runtime_by_worker(
          worker_id, device_id, worker_type_label, task_runtime_ms);
    }

    auto compute_start = timing.inference_start_time;
    if (compute_start == zero_tp) {
      compute_start = timing.codelet_start_time;
    }
    auto compute_end = timing.callback_start_time;
    if (compute_end == zero_tp || compute_end < compute_start) {
      compute_end = timing.codelet_end_time;
    }
    if (compute_start != zero_tp && compute_end > compute_start) {
      const double compute_ms =
          std::chrono::duration<double, std::milli>(compute_end - compute_start)
              .count();
      observe_compute_latency_by_worker(
          worker_id, device_id, worker_type_label, compute_ms);
    }
    congestion::record_completion(
        logical_jobs, congestion::CompletionLatencies{
                          .queue_latency_ms = breakdown.queue_ms,
                          .e2e_latency_ms = latency.count(),
                      });
  }
  perf_observer::record_job(
      timing.enqueued_time, timing.callback_end_time, batch_size, warmup);

  const int job_id = submission_id >= 0 ? submission_id : job->get_request_id();
  log_job_timings(job_id, latency, timing);

  auto& tracer = BatchingTraceLogger::instance();
  if (tracer.enabled()) {
    const auto request_ids =
        task_runner_internal::build_request_ids_for_trace(job);
    const auto request_arrivals =
        task_runner_internal::build_request_arrival_us_for_trace(job);
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
        .congested = congestion::is_congested(),
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
  if (job == nullptr || completed_jobs_ == nullptr || all_done_cv_ == nullptr) {
    return;
  }
  const std::size_t logical_jobs =
      static_cast<std::size_t>(std::max(1, job->logical_job_count()));
  completed_jobs_->fetch_add(logical_jobs, std::memory_order_release);
  all_done_cv_->notify_all();
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
        ResultDispatcher::emit_batch_traces(opts_, job);
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

    if (entry.callback) {
      entry.callback(outputs, latency_ms);
    }

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
    const RuntimeConfig* opts, const std::shared_ptr<InferenceJob>& job)
{
  if (!job) {
    return;
  }
  auto& tracer = BatchingTraceLogger::instance();
  if (!tracer.enabled()) {
    return;
  }
  const bool warmup_job = is_warmup_job(job);
  const auto timing = job->timing_info_snapshot();
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
      .batch_size = resolve_batch_size(opts, job),
      .worker_id = job->get_worker_id(),
      .worker_type = job->get_executed_on(),
      .codelet_times =
          BatchingTraceLogger::TimeRange{compute_start, compute_end},
      .is_warmup = warmup_job,
      .device_id = job->get_device_id(),
  });
}

void
ResultDispatcher::release_inflight_slot(
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
  set_inflight_tasks(previous - 1);
  if (inflight_state->max_tasks > 0 && previous > 0) {
    const double ratio = static_cast<double>(previous - 1) /
                         static_cast<double>(inflight_state->max_tasks);
    set_starpu_worker_busy_ratio(ratio);
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
