#include "warmup.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "Inference_queue.hpp"
#include "inference_runner.hpp"
#include "input_generator.hpp"
#include "logger.hpp"
#include "runtime_config.hpp"
#include "server_worker.hpp"
#include "starpu_setup.hpp"

constexpr int NUM_PREGENERATED_INPUTS = 2;

WarmupRunner::WarmupRunner(
    const RuntimeConfig& opts, StarPUSetup& starpu,
    torch::jit::script::Module& model_cpu,
    std::vector<torch::jit::script::Module>& models_gpu,
    const std::vector<torch::Tensor>& outputs_ref)
    : opts_(opts), starpu_(starpu), model_cpu_(model_cpu),
      models_gpu_(models_gpu), outputs_ref_(outputs_ref),
      dummy_completed_jobs_(0)
{
}

void
WarmupRunner::client_worker(
    const std::map<unsigned int, std::vector<int32_t>> device_workers,
    InferenceQueue& queue, const int iterations_per_worker)
{
  // Pre-generate random input tensors
  std::vector<std::vector<torch::Tensor>> pregen_inputs;
  pregen_inputs.reserve(NUM_PREGENERATED_INPUTS);

  std::generate_n(
      std::back_inserter(pregen_inputs), NUM_PREGENERATED_INPUTS, [&]() {
        return generate_random_inputs(opts_.input_shapes, opts_.input_types);
      });

  // RNG setup to randomly pick pre-generated inputs for warmup jobs
  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<> dist(0, NUM_PREGENERATED_INPUTS - 1);

  unsigned int job_id = 0;

  // Iterate over each device ID and its associated StarPU worker IDs
  for (const auto& [device_id, worker_ids] : device_workers) {
    for (const int worker_id : worker_ids) {
      for (unsigned int i = 0; i < iterations_per_worker; ++i) {
        auto job = std::make_shared<InferenceJob>();

        const auto& chosen_inputs =
            pregen_inputs[static_cast<std::size_t>(dist(rng))];
        job->set_input_tensors(chosen_inputs);

        std::vector<at::ScalarType> types;
        types.reserve(chosen_inputs.size());
        for (const auto& tensor : chosen_inputs) {
          types.emplace_back(tensor.scalar_type());
        }
        job->set_input_types(types);

        std::vector<torch::Tensor> outputs;
        outputs.reserve(outputs_ref_.size());
        for (const auto& ref : outputs_ref_) {
          outputs.emplace_back(torch::empty_like(ref));
        }
        job->set_outputs_tensors(outputs);

        job->set_job_id(job_id++);
        job->set_fixed_worker_id(worker_id);

        auto now = std::chrono::high_resolution_clock::now();
        job->set_start_time(now);
        job->timing_info().enqueued_time = now;

        log_trace(
            opts_.verbosity, "[Warmup] Job ID " +
                                 std::to_string(job->get_job_id()) +
                                 ", Iteration " + std::to_string(i + 1) + "/" +
                                 std::to_string(iterations_per_worker) +
                                 ", device ID " + std::to_string(device_id) +
                                 ", worker ID " + std::to_string(worker_id));

        queue.push(job);
      }
    }
  }

  queue.shutdown();
}

void
WarmupRunner::run(const int iterations_per_worker)
{
  if (!opts_.use_cuda) {
    return;
  }

  // Dummy resources (since warmup results aren't needed)
  InferenceQueue queue;
  std::atomic<unsigned int> dummy_completed_jobs = 0;
  std::mutex dummy_mutex;
  std::mutex dummy_results_mutex;
  std::condition_variable dummy_cv;
  std::vector<InferenceResult> dummy_results;

  // Launch server thread
  ServerWorker worker(
      &queue, &model_cpu_, &models_gpu_, &starpu_, &opts_, &dummy_results,
      &dummy_results_mutex, &dummy_completed_jobs, &dummy_cv);

  std::thread server(&ServerWorker::run, &worker);

  const auto device_workers =
      StarPUSetup::get_cuda_workers_by_device(opts_.device_ids);

  // Launch client thread to feed warmup jobs
  std::thread client(
      [&]() { client_worker(device_workers, queue, iterations_per_worker); });

  // Wait for client thread to finish pushing jobs
  client.join();

  // Count the total number of CUDA workers
  size_t total_worker_count = 0;
  for (const auto& [device_id, worker_list] : device_workers) {
    total_worker_count += worker_list.size();
  }

  // Wait until all warmup jobs are marked completed
  {
    std::unique_lock<std::mutex> lock(dummy_mutex);
    const size_t total_jobs =
        static_cast<size_t>(iterations_per_worker) * total_worker_count;
    dummy_cv.wait(
        lock, [&]() { return dummy_completed_jobs.load() >= total_jobs; });
  }

  // Join server thread
  server.join();
}