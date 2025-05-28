#include "server_worker.hpp"

#include <iostream>

#include "exceptions.hpp"
#include "inference_task.hpp"

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

    job->on_complete =
        [this, id = job->job_id, inputs = job->input_tensors,
         &executed_on = job->executed_on, &timing_info = job->timing_info,
         &device_id = job->device_id](torch::Tensor result, double latency_ms) {
          {
            std::lock_guard<std::mutex> lock(results_mutex_);
            results_.push_back(
                {id, inputs, result, latency_ms, executed_on, timing_info,
                 device_id});
          }

          log_stats(
              opts_.verbosity, "Completed job ID: " + std::to_string(id) +
                                   ", latency: " + std::to_string(latency_ms) +
                                   " ms");

          completed_jobs_.fetch_add(1);
          all_done_cv_.notify_one();
        };

    auto fail_job = [&](const std::string& error_msg) {
      log_error(error_msg);
      if (job->on_complete) {
        job->on_complete(torch::Tensor(), -1);
      }
    };

    try {
      job->timing_info.dequeued_time =
          std::chrono::high_resolution_clock::now();

      log_debug(
          opts_.verbosity, "Submitting job ID: " + std::to_string(job->job_id));

      InferenceTask inferenceTask(starpu_, job, model_cpu_, models_gpu_, opts_);
      inferenceTask.submit();
    }
    catch (const InferenceEngineException& e) {
      fail_job(
          "[Inference Error] Job " + std::to_string(job->job_id) + ": " +
          e.what());
    }
    catch (const std::exception& e) {
      fail_job(
          "[Unhandled Exception] Job " + std::to_string(job->job_id) + ": " +
          e.what());
    }
  }

  log_info(opts_.verbosity, "ServerWorker stopped.");
}