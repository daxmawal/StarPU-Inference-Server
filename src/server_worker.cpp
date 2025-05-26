#include "server_worker.hpp"

#include <iostream>

#include "exceptions.hpp"
#include "inference_task.hpp"

void
ServerWorker::run()
{
  while (true) {
    std::shared_ptr<InferenceJob> job;
    queue_.wait_and_pop(job);

    if (job->is_shutdown())
      break;

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

          completed_jobs_.fetch_add(1);
          all_done_cv_.notify_one();
        };

    auto fail_job = [&](const std::string& error_msg) {
      std::cerr << error_msg << std::endl;
      if (job->on_complete) {
        job->on_complete(torch::Tensor(), -1);
      }
    };

    try {
      job->timing_info.dequeued_time =
          std::chrono::high_resolution_clock::now();
      InferenceTask inferenceTask(starpu_, job, model_cpu_, model_gpu_, opts_);
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
}