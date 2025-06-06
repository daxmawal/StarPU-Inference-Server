#pragma once

#include <chrono>
#include <random>
#include <string>
#include <vector>

#include "Inference_queue.hpp"
#include "client_utils.hpp"
#include "inference_runner.hpp"
#include "input_generator.hpp"
#include "logger.hpp"
#include "runtime_config.hpp"
#include "server_worker.hpp"
#include "starpu_setup.hpp"

// =============================================================================
// client_utils: Helper utilities for inference input preparation and job setup
// =============================================================================

namespace client_utils {
auto pre_generate_inputs(const RuntimeConfig& opts, size_t num_inputs)
    -> std::vector<std::vector<torch::Tensor>>;

auto pick_random_input(
    const std::vector<std::vector<torch::Tensor>>& pool,
    std::mt19937& rng) -> const std::vector<torch::Tensor>&;

void log_job_enqueued(
    const RuntimeConfig& opts, unsigned int job_id, size_t iterations,
    std::chrono::high_resolution_clock::time_point now);

auto create_job(
    const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& outputs_ref,
    unsigned int job_id) -> std::shared_ptr<InferenceJob>;

}  // namespace client_utils