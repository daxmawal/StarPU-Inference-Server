#include "client_utils.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
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
#include "server_worker.hpp"
#include "starpu_setup.hpp"
#include "time_utils.hpp"

namespace client_utils {

// =============================================================================
// Time Utilities: Format timestamp for logging/debugging
// =============================================================================

auto
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
    return input_generator::generate_random_inputs(
        opts.input_shapes, opts.input_types);
  });
  return inputs;
}

auto
pick_random_input(
    const std::vector<std::vector<torch::Tensor>>& pool,
    std::mt19937& rng) -> const std::vector<torch::Tensor>&
{
  std::uniform_int_distribution<> dist(0, static_cast<int>(pool.size()) - 1);
  const auto idx = static_cast<size_t>(dist(rng));
  TORCH_CHECK(idx < pool.size(), "Random index out of bounds.");
  return pool[idx];
}

// =============================================================================
// Logging: Track job enqueueing for debugging or performance tracking
// =============================================================================

void
log_job_enqueued(
    const RuntimeConfig& opts, unsigned int job_id, size_t iterations,
    std::chrono::high_resolution_clock::time_point now)
{
  log_trace(
      opts.verbosity, "[Inference] Job ID " + std::to_string(job_id) +
                          ", Iteration " + std::to_string(job_id + 1) + "/" +
                          std::to_string(iterations) + ", Enqueued at " +
                          current_time_formatted(now));
}

// =============================================================================
// Job Creation: Build InferenceJob with input tensors and allocated outputs
// =============================================================================

auto
create_job(
    const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& outputs_ref,
    unsigned int job_id) -> std::shared_ptr<InferenceJob>
{
  auto job = std::make_shared<InferenceJob>();
  job->set_input_tensors(inputs);

  std::vector<at::ScalarType> types;
  types.reserve(inputs.size());
  std::transform(
      inputs.begin(), inputs.end(), std::back_inserter(types),
      [](const auto& tensor) { return tensor.scalar_type(); });
  job->set_input_types(types);

  std::vector<torch::Tensor> outputs;
  outputs.reserve(outputs_ref.size());
  std::transform(
      outputs_ref.begin(), outputs_ref.end(), std::back_inserter(outputs),
      [](const auto& ref) { return torch::empty_like(ref); });
  job->set_outputs_tensors(outputs);

  job->set_job_id(job_id);

  auto now = std::chrono::high_resolution_clock::now();
  job->set_start_time(now);
  job->timing_info().enqueued_time = now;

  return job;
}

}  // namespace client_utils