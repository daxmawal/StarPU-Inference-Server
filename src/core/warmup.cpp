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
#include "client_utils.hpp"
#include "inference_runner.hpp"
#include "input_generator.hpp"
#include "logger.hpp"
#include "runtime_config.hpp"
#include "server_worker.hpp"
#include "starpu_setup.hpp"

constexpr size_t NUM_PREGENERATED_INPUTS = 2;

// =============================================================================
// Constructor
// =============================================================================

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

// =============================================================================
// Client thread: generate and enqueue jobs for warmup
// =============================================================================

void
WarmupRunner::client_worker(
    const std::map<unsigned int, std::vector<int32_t>>& device_workers,
    InferenceQueue& queue, unsigned int iterations_per_worker) const
{
  // Pre-generates a small set of random inputs for reuse
  auto pregen_inputs =
      client_utils::pre_generate_inputs(opts_, NUM_PREGENERATED_INPUTS);
  std::mt19937 rng(std::random_device{}());

  unsigned int job_id = 0;
  const std::size_t total =
      std::accumulate(
          device_workers.begin(), device_workers.end(), std::size_t{0},
          [](std::size_t sum, const auto& pair) {
            return sum + pair.second.size();
          }) *
      iterations_per_worker;

  // Sends jobs to the queue, fixing the targeted workers
  for (const auto& [device_id, worker_ids] : device_workers) {
    for (const int worker_id : worker_ids) {
      for (unsigned int iteration = 0; iteration < iterations_per_worker;
           ++iteration) {
        const auto& inputs =
            client_utils::pick_random_input(pregen_inputs, rng);
        auto job = client_utils::create_job(inputs, outputs_ref_, job_id);
        job->set_fixed_worker_id(worker_id);

        client_utils::log_job_enqueued(
            opts_, job_id, total, job->timing_info().enqueued_time);

        queue.push(job);
        job_id++;
      }
    }
  }

  queue.shutdown();  // Tells the server that there are no more jobs
}

// =============================================================================
// Warmup execution: launch server and client threads and wait for completion
// =============================================================================

void
WarmupRunner::run(unsigned int iterations_per_worker)
{
  if (!opts_.use_cuda) {
    return;
  }

  InferenceQueue queue;
  std::atomic<unsigned int> dummy_completed_jobs = 0;
  std::mutex dummy_mutex;
  std::mutex dummy_results_mutex;
  std::condition_variable dummy_cv;
  std::vector<InferenceResult> dummy_results;

  // Start the server (consumes jobs in the queue)
  ServerWorker worker(
      &queue, &model_cpu_, &models_gpu_, &starpu_, &opts_, &dummy_results,
      &dummy_results_mutex, &dummy_completed_jobs, &dummy_cv);

  std::thread server(&ServerWorker::run, &worker);

  const auto device_workers =
      StarPUSetup::get_cuda_workers_by_device(opts_.device_ids);

  // Launch the client (generates the jobs to be run)
  std::thread client(
      [&]() { client_worker(device_workers, queue, iterations_per_worker); });

  client.join();  // wait for jobs to finish sending

  // Calculation of the total number of jobs to expect
  size_t total_worker_count = 0;
  for (const auto& [device_id, worker_list] : device_workers) {
    total_worker_count += worker_list.size();
  }

  {
    std::unique_lock<std::mutex> lock(dummy_mutex);
    const size_t total_jobs =
        static_cast<size_t>(iterations_per_worker) * total_worker_count;
    dummy_cv.wait(
        lock, [&]() { return dummy_completed_jobs.load() >= total_jobs; });
  }

  server.join();
}