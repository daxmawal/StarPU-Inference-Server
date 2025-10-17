#include "warmup.hpp"

#include <torch/torch.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <format>
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#include "client_utils.hpp"
#include "exceptions.hpp"
#include "inference_queue.hpp"
#include "inference_runner.hpp"
#include "input_generator.hpp"
#include "logger.hpp"
#include "runtime_config.hpp"
#include "starpu_setup.hpp"
#include "starpu_task_worker.hpp"

namespace starpu_server {
// =============================================================================
// Constructor
// =============================================================================

WarmupRunner::WarmupRunner(
    const RuntimeConfig& opts, StarPUSetup& starpu,
    torch::jit::script::Module& model_cpu,
    std::vector<torch::jit::script::Module>& models_gpu,
    const std::vector<torch::Tensor>& outputs_ref,
    CompletionObserver completion_observer)
    : opts_(opts), starpu_(starpu), model_cpu_(model_cpu),
      models_gpu_(models_gpu), outputs_ref_(outputs_ref),
      completion_observer_(std::move(completion_observer))
{
}

// =============================================================================
// Client thread: generate and enqueue jobs for warmup
// =============================================================================

void
WarmupRunner::client_worker(
    const std::map<int, std::vector<int32_t>>& device_workers,
    InferenceQueue& queue, int request_nb_per_worker) const
{
  thread_local std::mt19937 rng;
  if (opts_.seed.has_value()) {
    rng.seed(*opts_.seed);
    torch::manual_seed(*opts_.seed);
  } else {
    rng.seed(std::random_device{}());
  }

  auto pregen_inputs =
      client_utils::pre_generate_inputs(opts_, opts_.warmup_pregen_inputs);

  if (request_nb_per_worker < 0) {
    throw std::invalid_argument("request_nb_per_worker must be non-negative");
  }

  const size_t worker_count = std::accumulate(
      device_workers.begin(), device_workers.end(), std::size_t{0},
      [](std::size_t sum, const auto& pair) {
        return sum + pair.second.size();
      });
  const size_t total_size_t =
      worker_count * static_cast<std::size_t>(request_nb_per_worker);

  if (total_size_t > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("Total exceeds int capacity");
  }

  const auto total = static_cast<int>(total_size_t);
  int request_id = 0;

  std::vector<int> flat_worker_ids;
  flat_worker_ids.reserve(worker_count);
  for ([[maybe_unused]] const auto& [device_id, worker_ids] : device_workers) {
    flat_worker_ids.insert(
        flat_worker_ids.end(), worker_ids.begin(), worker_ids.end());
  }

  for (const int worker_id : flat_worker_ids) {
    for (auto request_nb = 0; request_nb < request_nb_per_worker;
         ++request_nb) {
      const auto& inputs = client_utils::pick_random_input(pregen_inputs, rng);
      auto job = client_utils::create_job(inputs, outputs_ref_, request_id);
      job->set_fixed_worker_id(worker_id);

      client_utils::log_job_enqueued(
          opts_, request_id, total, job->timing_info().enqueued_time);

      if (!queue.push(job)) {
        log_warning(std::format(
            "[Warmup] Failed to enqueue job {}: queue shutting down",
            request_id));
        queue.shutdown();
        return;
      }
      request_id++;
    }
  }

  queue.shutdown();
}

// =============================================================================
// Warmup execution: launch server and client threads and wait for completion
// =============================================================================

void
WarmupRunner::run(int request_nb_per_worker)
{
  if (request_nb_per_worker < 0) {
    throw std::invalid_argument("request_nb_per_worker must be non-negative");
  }

  if (!opts_.use_cuda) {
    return;
  }

  InferenceQueue queue;
  std::atomic dummy_completed_jobs = 0;
  std::mutex dummy_mutex;
  std::mutex dummy_results_mutex;
  std::condition_variable dummy_cv;
  std::vector<InferenceResult> dummy_results;

  StarPUTaskRunnerConfig config{};
  config.queue = &queue;
  config.model_cpu = &model_cpu_;
  config.models_gpu = &models_gpu_;
  config.starpu = &starpu_;
  config.opts = &opts_;
  config.results = &dummy_results;
  config.results_mutex = &dummy_results_mutex;
  config.completed_jobs = &dummy_completed_jobs;
  config.all_done_cv = &dummy_cv;
  StarPUTaskRunner worker(config);

  const std::jthread server(&StarPUTaskRunner::run, &worker);

  const auto device_workers =
      StarPUSetup::get_cuda_workers_by_device(opts_.device_ids);

  const std::jthread client(
      [&]() { client_worker(device_workers, queue, request_nb_per_worker); });

  size_t total_worker_count = 0;
  for (const auto& [device_id, worker_list] : device_workers) {
    total_worker_count += worker_list.size();
  }

  {
    std::unique_lock lock(dummy_mutex);
    const size_t total_jobs =
        static_cast<size_t>(request_nb_per_worker) * total_worker_count;
    dummy_cv.wait(lock, [this, total_jobs, &dummy_completed_jobs]() {
      if (completion_observer_) {
        completion_observer_(dummy_completed_jobs);
      }
      int count = dummy_completed_jobs.load();
      if (count < 0) {
        throw InferenceExecutionException(
            "dummy_completed_jobs became negative, which should not happen.");
      }
      return static_cast<size_t>(count) >= total_jobs;
    });
  }
}
}  // namespace starpu_server
