#include "server_worker.hpp"

#include <iostream>

#include "exceptions.hpp"
#include "inference_task.hpp"

// =============================================================================
// Constructor
// =============================================================================
ServerWorker::ServerWorker(
    InferenceQueue& queue, torch::jit::script::Module& model_cpu,
    std::vector<torch::jit::script::Module>& models_gpu, StarPUSetup& starpu,
    const ProgramOptions& opts, std::vector<InferenceResult>& results,
    std::mutex& results_mutex, std::atomic<unsigned int>& completed_jobs,
    std::condition_variable& all_done_cv)
    : queue_(queue), model_cpu_(model_cpu), models_gpu_(models_gpu),
      starpu_(starpu), opts_(opts), results_(results),
      results_mutex_(results_mutex), completed_jobs_(completed_jobs),
      all_done_cv_(all_done_cv)
{
}

// =============================================================================
// Main loop
// =============================================================================
void
ServerWorker::run()
{
  log_info(opts_.verbosity, "ServerWorker started.");

  while (true) {
    std::shared_ptr<InferenceJob> job;
    queue_.wait_and_pop(job);

    if (job->is_shutdown()) {
      log_info(
          opts_.verbosity,
          "Received shutdown signal. Exiting ServerWorker loop.");
      break;
    }

    log_trace(
        opts_.verbosity, "Dequeued job ID: " + std::to_string(job->job_id));

    // Completion callback for this job
    job->on_complete =
        [this, id = job->job_id, inputs = job->input_tensors,
         &executed_on = job->executed_on, &timing_info = job->timing_info,
         &device_id = job->device_id,
         &worker_id = job->worker_id](torch::Tensor result, double latency_ms) {
          {
            std::lock_guard<std::mutex> lock(results_mutex_);
            results_.emplace_back(InferenceResult{
                id, inputs, result, latency_ms, executed_on, device_id,
                worker_id, timing_info});
          }

          log_stats(
              opts_.verbosity, "Completed job ID: " + std::to_string(id) +
                                   ", latency: " + std::to_string(latency_ms) +
                                   " ms");

          completed_jobs_.fetch_add(1);
          all_done_cv_.notify_one();
        };

    // Submission with error handling
    try {
      job->timing_info.dequeued_time =
          std::chrono::high_resolution_clock::now();

      log_debug(
          opts_.verbosity, "Submitting job ID: " + std::to_string(job->job_id));

      InferenceTask inferenceTask(starpu_, job, model_cpu_, models_gpu_, opts_);
      inferenceTask.submit();
    }
    catch (const InferenceEngineException& e) {
      log_error(
          "[Inference Error] Job " + std::to_string(job->job_id) + ": " +
          e.what());
      if (job->on_complete)
        job->on_complete(torch::Tensor(), -1);
    }
    catch (const std::exception& e) {
      log_error(
          "[Unhandled Exception] Job " + std::to_string(job->job_id) + ": " +
          e.what());
      if (job->on_complete)
        job->on_complete(torch::Tensor(), -1);
    }
  }

  log_info(opts_.verbosity, "ServerWorker stopped.");
}
