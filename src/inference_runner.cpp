#include "inference_runner.hpp"

#include <torch/script.h>

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "exceptions.hpp"
#include "inference_task.hpp"
#include "inference_validator.hpp"
#include "input_generator.hpp"

class InferenceQueue {
 public:
  void push(const std::shared_ptr<InferenceJob>& job)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    queue_.push(job);
    cv_.notify_one();
  }

  void wait_and_pop(std::shared_ptr<InferenceJob>& job)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return !queue_.empty(); });
    job = queue_.front();
    queue_.pop();
  }

  void shutdown() { push(InferenceJob::make_shutdown_job()); }

 private:
  std::queue<std::shared_ptr<InferenceJob>> queue_;
  std::mutex mutex_;
  std::condition_variable cv_;
};

void
client_worker(
    InferenceQueue& queue, const ProgramOptions& opts,
    const torch::Tensor& output_ref, int iterations)
{
  for (int i = 0; i < iterations; ++i) {
    auto job = std::make_shared<InferenceJob>();
    job->input_tensors =
        generate_random_inputs(opts.input_shapes, opts.input_types);
    for (size_t j = 0; j < job->input_tensors.size(); ++j) {
      job->input_types.emplace_back(job->input_tensors[j].scalar_type());
    }
    job->output_tensor = torch::empty_like(output_ref);
    job->job_id = i;
    job->start_time = std::chrono::high_resolution_clock::now();
    queue.push(job);
    std::this_thread::sleep_for(std::chrono::milliseconds(opts.delay_ms));
  }
  queue.shutdown();
}

void
server_worker(
    InferenceQueue& queue, torch::jit::script::Module& module,
    StarPUSetup& starpu, const ProgramOptions& opts,
    std::vector<InferenceResult>& results, std::mutex& results_mutex,
    std::atomic<int>& completed_jobs, std::condition_variable& all_done_cv)
{
  while (true) {
    std::shared_ptr<InferenceJob> job;
    queue.wait_and_pop(job);

    if (job->is_shutdown())
      break;

    job->on_complete = [id = job->job_id, inputs = job->input_tensors, &results,
                        &results_mutex, &completed_jobs, &all_done_cv](
                           torch::Tensor result, int64_t latency_us) {
      {
        std::lock_guard<std::mutex> lock(results_mutex);
        results.push_back({id, inputs, result, latency_us});
      }

      completed_jobs.fetch_add(1);
      all_done_cv.notify_one();
    };

    auto fail_job = [&](const std::string& error_msg) {
      std::cerr << error_msg << std::endl;
      if (job->on_complete) {
        job->on_complete(torch::Tensor(), -1);
      }
    };

    try {
      submit_inference_task(starpu, job, module, opts);
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

std::pair<torch::jit::script::Module, torch::Tensor>
load_model_and_reference_output(const ProgramOptions& opts)
{
  torch::jit::script::Module module;
  torch::Tensor output_ref;

  try {
    module = torch::jit::load(opts.model_path);

    auto inputs = generate_random_inputs(opts.input_shapes, opts.input_types);
    std::vector<torch::IValue> input_ivalues(inputs.begin(), inputs.end());

    output_ref = module.forward(input_ivalues).toTensor();
  }
  catch (const c10::Error& e) {
    std::cerr << "[Error] Failed to load model or run reference inference: "
              << e.what() << std::endl;
    throw;
  }

  return {module, output_ref};
}

void
run_inference_loop(const ProgramOptions& opts, StarPUSetup& starpu)
{
  torch::jit::script::Module module;
  torch::Tensor output_ref;

  try {
    std::tie(module, output_ref) = load_model_and_reference_output(opts);
  }
  catch (...) {
    return;
  }

  InferenceQueue queue;
  std::vector<InferenceResult> results;
  std::mutex results_mutex;

  results.reserve(opts.iterations);
  std::atomic<int> completed_jobs = 0;
  std::condition_variable all_done_cv;
  std::mutex all_done_mutex;

  std::thread server([&]() {
    server_worker(
        queue, module, starpu, opts, results, results_mutex, completed_jobs,
        all_done_cv);
  });

  std::thread client(
      [&]() { client_worker(queue, opts, output_ref, opts.iterations); });

  client.join();

  {
    std::unique_lock<std::mutex> lock(all_done_mutex);
    all_done_cv.wait(
        lock, [&]() { return completed_jobs.load() >= opts.iterations; });
  }

  server.join();
  std::lock_guard<std::mutex> lock(results_mutex);

  for (const auto& r : results) {
    if (!r.result.defined()) {
      std::cerr << "[Client] Job " << r.job_id << " failed.\n";
    } else {
      std::cout << "[Client] Job " << r.job_id
                << " done. Latency = " << r.latency_us << " µs\n";
      validate_inference_result(r, module);
    }
  }
}