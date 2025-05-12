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

struct InferenceJob {
  torch::Tensor input_tensor;
  torch::Tensor output_tensor;
  int job_id;
  bool is_shutdown_signal = false;
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
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  auto shutdown_job = std::make_shared<InferenceJob>();
  shutdown_job->is_shutdown_signal = true;
  queue.push(shutdown_job);
}

void
server_thread(
    InferenceQueue& queue, torch::jit::script::Module& module,
    StarPUSetup& starpu, const ProgramOptions& opts,
    const torch::Tensor& output_ref)
{
  while (true) {
    std::shared_ptr<InferenceJob> job;
    queue.wait_and_pop(job);

    if (job->is_shutdown_signal)
      break;

    auto start_time = std::chrono::high_resolution_clock::now();
    submit_inference_task(
        starpu, job->input_tensor, job->output_tensor, module, opts, output_ref,
        job->job_id, start_time);
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

  std::thread server(
      [&]() { server_thread(queue, module, starpu, opts, output_ref); });

  std::thread client(
      [&]() { client_thread(queue, input_ref, output_ref, opts.iterations); });

  client.join();
  server.join();
}
