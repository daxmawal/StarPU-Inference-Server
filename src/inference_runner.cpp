#include "inference_runner.hpp"

#include <torch/script.h>

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "inference_task.hpp"
#include "inference_validator.hpp"

struct InferenceResult {
  int job_id;
  torch::Tensor result;
  int64_t latency;
};

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

 private:
  std::queue<std::shared_ptr<InferenceJob>> queue_;
  std::mutex mutex_;
  std::condition_variable cv_;
};

void
client_thread(
    InferenceQueue& queue, const torch::Tensor& input_ref,
    const torch::Tensor& output_ref, int iterations)
{
  for (int i = 0; i < iterations; ++i) {
    auto job = std::make_shared<InferenceJob>();
    job->input_tensor = input_ref.clone();
    job->output_tensor = torch::empty_like(output_ref);
    job->job_id = i;
    queue.push(job);
    std::this_thread::sleep_for(std::chrono::milliseconds(0));
  }
  auto shutdown_job = std::make_shared<InferenceJob>();
  shutdown_job->is_shutdown_signal = true;
  queue.push(shutdown_job);
}

void
server_thread(
    InferenceQueue& queue, torch::jit::script::Module& module,
    StarPUSetup& starpu, const ProgramOptions& opts,
    const torch::Tensor& output_ref, std::vector<InferenceResult>& results,
    std::mutex& results_mutex, std::atomic<int>& completed_jobs,
    std::condition_variable& all_done_cv, std::mutex& all_done_mutex)
{
  while (true) {
    std::shared_ptr<InferenceJob> job;
    queue.wait_and_pop(job);

    if (job->is_shutdown_signal)
      break;

    job->on_complete = [id = job->job_id, &results, &results_mutex,
                        &completed_jobs,
                        &all_done_cv](torch::Tensor result, int64_t latency) {
      {
        std::lock_guard<std::mutex> lock(results_mutex);
        results.push_back({id, result, latency});
      }

      completed_jobs.fetch_add(1);
      all_done_cv.notify_one();
    };

    auto start_time = std::chrono::high_resolution_clock::now();
    submit_inference_task(starpu, job, module, opts, output_ref, start_time);
  }
}

void
run_inference_loop(const ProgramOptions& opts, StarPUSetup& starpu)
{
  torch::jit::script::Module module = torch::jit::load(opts.model_path);

  if (opts.input_shape.empty()) {
    std::cerr << "Error: you must provide --shape for the input tensor.\n";
    return;
  }

  torch::Tensor input_ref = torch::rand(opts.input_shape);
  torch::Tensor output_ref = module.forward({input_ref}).toTensor();

  InferenceQueue queue;
  std::vector<InferenceResult> results;
  std::mutex results_mutex;

  std::atomic<int> completed_jobs = 0;
  std::condition_variable all_done_cv;
  std::mutex all_done_mutex;

  std::thread server([&]() {
    server_thread(
        queue, module, starpu, opts, output_ref, results, results_mutex,
        completed_jobs, all_done_cv, all_done_mutex);
  });

  std::thread client(
      [&]() { client_thread(queue, input_ref, output_ref, opts.iterations); });

  client.join();
  server.join();

  {
    std::unique_lock<std::mutex> lock(all_done_mutex);
    all_done_cv.wait(
        lock, [&]() { return completed_jobs.load() >= opts.iterations; });
  }

  std::lock_guard<std::mutex> lock(results_mutex);
  std::cout << "results size : " << results.size() << std::endl;

  for (const auto& r : results) {
    std::cout << "[Client] Job " << r.job_id << " done. Latency = " << r.latency
              << " µs\n";
    std::cout << "[Client] Result (first 10): "
              << r.result.flatten().slice(0, 0, 10) << "\n";
    validate_outputs(output_ref, r.result);
  }
}