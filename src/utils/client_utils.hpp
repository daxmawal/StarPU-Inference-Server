#pragma once

#include <chrono>
#include <random>
#include <string>
#include <vector>

#include "inference_queue.hpp"
#include "inference_runner.hpp"
#include "input_generator.hpp"
#include "logger.hpp"
#include "runtime_config.hpp"
#include "starpu_setup.hpp"
#include "starpu_task_worker.hpp"

namespace starpu_server::client_utils {

// =============================================================================
// client_utils: Helper utilities for inference input preparation and job setup
// =============================================================================

auto pre_generate_inputs(const RuntimeConfig& opts, size_t num_inputs)
    -> std::vector<std::vector<torch::Tensor>>;

auto pick_random_input(
    const std::vector<std::vector<torch::Tensor>>& pool,
    std::mt19937& rng) -> const std::vector<torch::Tensor>&;

void log_job_enqueued(
    const RuntimeConfig& opts, int job_id, int iterations,
    std::chrono::high_resolution_clock::time_point now);

auto create_job(
    const std::vector<torch::Tensor>& inputs,
    const std::vector<torch::Tensor>& outputs_ref,
    int job_id) -> std::shared_ptr<InferenceJob>;

}  // namespace starpu_server::client_utils
