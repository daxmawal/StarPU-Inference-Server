#include "client_utils.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <format>
#include <iomanip>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "inference_queue.hpp"
#include "inference_runner.hpp"
#include "input_generator.hpp"
#include "logger.hpp"
#include "runtime_config.hpp"
#include "starpu_setup.hpp"
#include "starpu_task_worker.hpp"
#include "time_utils.hpp"

namespace starpu_server::client_utils {

// =============================================================================
// Time Utilities: Format timestamp for logging/debugging
// =============================================================================

static auto
current_time_formatted(const std::chrono::high_resolution_clock::time_point&
                           time_point) -> std::string
{
  return time_utils::format_timestamp(time_point);
}

// =============================================================================
// Input Preparation: Pre-generate random inputs and select random sample
// =============================================================================

auto
pre_generate_inputs(const RuntimeConfig& opts, size_t num_inputs)
    -> std::vector<std::vector<torch::Tensor>>
{
  std::vector<std::vector<torch::Tensor>> inputs;
  inputs.reserve(num_inputs);
  std::generate_n(std::back_inserter(inputs), num_inputs, [&]() {
    return input_generator::generate_random_inputs(opts.inputs);
  });
  return inputs;
}

auto
pick_random_input(
    const std::vector<std::vector<torch::Tensor>>& pool,
    std::mt19937& rng) -> const std::vector<torch::Tensor>&
{
  if (pool.empty()) {
    throw std::invalid_argument(
        "Input pool is empty. Cannot pick random input.");
  }
  std::uniform_int_distribution<std::size_t> dist(0, pool.size() - 1);
  const auto idx = static_cast<size_t>(dist(rng));
  TORCH_CHECK(idx < pool.size(), "Random index out of bounds.");
  return pool[idx];
}

// =============================================================================
// Logging: Track job enqueueing for debugging or performance tracking
// =============================================================================

void
log_job_enqueued(
    const RuntimeConfig& opts, int job_id, int iterations,
    std::chrono::high_resolution_clock::time_point now)
{
  log_trace(
      opts.verbosity,
      std::format(
          "[Inference] Job ID {} Iteration {}/{} Enqueued at {}", job_id,
          job_id + 1, iterations, current_time_formatted(now)));
}

// =============================================================================
// Job Creation: Build InferenceJob with input tensors and allocated outputs
// =============================================================================

auto
create_job(
    const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& outputs_ref,
    int job_id) -> std::shared_ptr<InferenceJob>
{
  auto job = std::make_shared<InferenceJob>();
  job->set_input_tensors(inputs);

  std::vector<at::ScalarType> types;
  types.reserve(inputs.size());
  std::ranges::transform(
      inputs, std::back_inserter(types),
      [](const auto& tensor) { return tensor.scalar_type(); });
  job->set_input_types(types);

  std::vector<torch::Tensor> outputs;
  outputs.reserve(outputs_ref.size());
  std::ranges::transform(
      outputs_ref, std::back_inserter(outputs),
      [](const auto& ref) { return torch::empty_like(ref); });
  job->set_outputs_tensors(outputs);

  job->set_job_id(job_id);

  auto now = std::chrono::high_resolution_clock::now();
  job->set_start_time(now);
  job->timing_info().enqueued_time = now;

  return job;
}

}  // namespace starpu_server::client_utils
