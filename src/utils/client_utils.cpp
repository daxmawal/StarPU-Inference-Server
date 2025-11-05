#include "client_utils.hpp"

#include <starpu.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <format>
#include <iterator>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "logger.hpp"
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
  const auto& tensors =
      opts.models.empty() ? std::vector<TensorConfig>{} : opts.models[0].inputs;
  std::generate_n(std::back_inserter(inputs), num_inputs, [&]() {
    return input_generator::generate_random_inputs(tensors);
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
  const auto idx = dist(rng);
  STARPU_ASSERT(idx < pool.size());
  return pool[idx];
}

// =============================================================================
// Logging: Track job enqueueing for debugging or performance tracking
// =============================================================================

void
log_job_enqueued(
    const RuntimeConfig& opts, int request_id, int request_nb,
    std::chrono::high_resolution_clock::time_point now)
{
  if (should_log(VerbosityLevel::Trace, opts.verbosity)) {
    log_trace(
        opts.verbosity,
        std::format(
            "[Inference] Request ID {} Iteration {}/{} Enqueued at {}",
            request_id, request_id + 1, request_nb,
            current_time_formatted(now)));
  }
}

// =============================================================================
// Job Creation: Build InferenceJob with input tensors and allocated outputs
// =============================================================================

auto
create_job(
    const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& outputs_ref, int request_id,
    std::vector<std::shared_ptr<const void>> input_lifetimes,
    std::chrono::high_resolution_clock::time_point start_time_arg,
    std::string model_name) -> std::shared_ptr<InferenceJob>
{
  auto job = std::make_shared<InferenceJob>();
  job->set_input_tensors(inputs);
  job->set_input_memory_holders(std::move(input_lifetimes));

  std::vector<at::ScalarType> types;
  types.reserve(inputs.size());
  std::ranges::transform(
      inputs, std::back_inserter(types),
      [](const auto& tensor) { return tensor.scalar_type(); });
  job->set_input_types(types);

  int64_t requested_batch = 1;
  if (!inputs.empty() && inputs[0].dim() >= 1) {
    requested_batch = inputs[0].size(0);
  }

  std::vector<torch::Tensor> outputs;
  outputs.reserve(outputs_ref.size());
  for (const auto& ref : outputs_ref) {
    const auto dtype = ref.scalar_type();
    const auto options = ref.options().dtype(dtype);

    std::vector<int64_t> shape(ref.sizes().begin(), ref.sizes().end());
    if (!shape.empty() && requested_batch > 0) {
      shape[0] = requested_batch;
    }
    outputs.emplace_back(torch::empty(shape, options));
  }
  job->set_output_tensors(outputs);

  job->set_request_id(request_id);
  job->set_model_name(std::move(model_name));

  auto start_time = start_time_arg;
  const auto enqueued_time = std::chrono::high_resolution_clock::now();
  if (start_time == std::chrono::high_resolution_clock::time_point{}) {
    start_time = enqueued_time;
  }
  job->set_start_time(start_time);
  job->timing_info().enqueued_time = enqueued_time;

  return job;
}

}  // namespace starpu_server::client_utils
