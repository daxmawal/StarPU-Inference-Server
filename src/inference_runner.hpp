#pragma once
#include "args_parser.hpp"
#include "starpu_setup.hpp"

struct InferenceJob {
  torch::Tensor input_tensor;
  torch::Tensor output_tensor;
  int job_id;
  bool is_shutdown_signal = false;

  std::function<void(torch::Tensor, int64_t)> on_complete;
};

void run_inference_loop(const ProgramOptions& opts, StarPUSetup& starpu);