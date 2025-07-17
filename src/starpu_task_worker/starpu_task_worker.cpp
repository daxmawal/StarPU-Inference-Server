#include "starpu_task_worker.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <format>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "exceptions.hpp"
#include "inference_queue.hpp"
#include "inference_runner.hpp"
#include "inference_task.hpp"
#include "logger.hpp"
#include "runtime_config.hpp"
#include "starpu_setup.hpp"

namespace starpu_server {
// =============================================================================
// Constructor
// =============================================================================

StarPUTaskRunner::StarPUTaskRunner(const StarPUTaskRunnerConfig& config)
    : queue_(config.queue), model_cpu_(config.model_cpu),
      models_gpu_(config.models_gpu), starpu_(config.starpu),
      opts_(config.opts), results_(config.results),
      results_mutex_(config.results_mutex),
      completed_jobs_(config.completed_jobs), all_done_cv_(config.all_done_cv)
{
}


// =============================================================================
// Job Queue Management
// =============================================================================

auto
StarPUTaskRunner::wait_for_next_job() -> std::shared_ptr<InferenceJob>
{
  std::shared_ptr<InferenceJob> job;
  queue_->wait_and_pop(job);
  return job;
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
    int job_id, double latency_ms, const detail::TimingInfo& timing_info) const
{
  using duration_f = std::chrono::duration<double, std::milli>;
  const auto queue_ms =
      duration_f(timing_info.dequeued_time - timing_info.enqueued_time).count();
  const auto submit_ms =
      duration_f(
          timing_info.before_starpu_submitted_time - timing_info.dequeued_time)
          .count();
  const auto scheduling_ms = duration_f(
                                 timing_info.codelet_start_time -
                                 timing_info.before_starpu_submitted_time)
                                 .count();
  const auto codelet_ms =
      duration_f(timing_info.codelet_end_time - timing_info.codelet_start_time)
          .count();
  const auto inference_ms =
      duration_f(
          timing_info.callback_start_time - timing_info.inference_start_time)
          .count();
  const auto callback_ms =
      duration_f(
          timing_info.callback_end_time - timing_info.callback_start_time)
          .count();

  log_stats(
      opts_->verbosity,
      std::format(
          "Job {} done. Latency = {:.3f} ms | Queue = {:.3f} ms, Submit = "
          "{:.3f} ms, Scheduling = {:.3f} ms, Codelet = {:.3f} ms, Inference = "
          "{:.3f} ms, Callback = {:.3f} ms",
          job_id, latency_ms, queue_ms, submit_ms, scheduling_ms, codelet_ms,
          inference_ms, callback_ms));
}

void
StarPUTaskRunner::prepare_job_completion_callback(
    const std::shared_ptr<InferenceJob>& job)
{
  const auto job_id = job->get_job_id();
  const auto inputs = job->get_input_tensors();
  const auto& executed_on = job->get_executed_on();
  const auto& timing_info = job->timing_info();
  const auto& device_id = job->get_device_id();
  const auto& worker_id = job->get_worker_id();

  auto prev_callback = job->get_on_complete();
  job->set_on_complete(
      [this, job_id, inputs, &executed_on, &timing_info, &device_id, &worker_id,
       prev_callback](
          const std::vector<torch::Tensor>& results, double latency_ms) {
        {
          const std::scoped_lock lock(*results_mutex_);
          results_->emplace_back(
              job_id, inputs, results, latency_ms, executed_on, device_id,
              worker_id, timing_info);
        }

        log_job_timings(job_id, latency_ms, timing_info);

        if (prev_callback) {
          prev_callback(results, latency_ms);
        }

        completed_jobs_->fetch_add(1);
        all_done_cv_->notify_one();
      });
}

// =============================================================================
// Error Handling for Failed Jobs
// =============================================================================

void
StarPUTaskRunner::handle_job_exception(
    const std::shared_ptr<InferenceJob>& job, const std::exception& exception)
{
  const auto job_id = job->get_job_id();
  log_error(std::format("[Exception] Job {}: {}", job_id, exception.what()));

  if (job->has_on_complete()) {
    job->get_on_complete()({}, -1);
  }
}

// =============================================================================
// StarPU Task Submission
// =============================================================================

void
StarPUTaskRunner::submit_inference_task(
    const std::shared_ptr<InferenceJob>& job)
{
  InferenceTask task(starpu_, job, model_cpu_, models_gpu_, opts_);
  task.submit();
}

// =============================================================================
// Main run loop: pull jobs, submit them, handle shutdown and errors
// =============================================================================

void
StarPUTaskRunner::run()
{
  log_info(opts_->verbosity, "StarPUTaskRunner started.");

  while (true) {
    auto job = wait_for_next_job();
    if (should_shutdown(job)) {
      break;
    }

    const auto job_id = job->get_job_id();
    log_trace(opts_->verbosity, std::format("Dequeued job ID: {}", job_id));

    prepare_job_completion_callback(job);

    try {
      job->timing_info().dequeued_time =
          std::chrono::high_resolution_clock::now();

      log_debug(opts_->verbosity, std::format("Submitting job ID: {}", job_id));

      submit_inference_task(job);
    }
    catch (const InferenceEngineException& exception) {
      StarPUTaskRunner::handle_job_exception(job, exception);
    }
  }

  log_info(opts_->verbosity, "StarPUTaskRunner stopped.");
}
}  // namespace starpu_server
