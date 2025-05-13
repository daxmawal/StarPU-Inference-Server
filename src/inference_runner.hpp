#pragma once
#include "args_parser.hpp"
#include "starpu_setup.hpp"

struct InferenceResult {
  int job_id;
  torch::Tensor input;
  torch::Tensor result;
  int64_t latency;
};

struct InferenceJob {
  torch::Tensor input_tensor;
  torch::Tensor output_tensor;
  int job_id;
  bool is_shutdown_signal = false;
  std::chrono::high_resolution_clock::time_point start_time;

  std::function<void(torch::Tensor, int64_t)> on_complete;
};

void run_inference_loop(const ProgramOptions& opts, StarPUSetup& starpu);

bool validate_inference_result(
    const InferenceResult& r, torch::jit::script::Module& module);