#pragma once
#include "args_parser.hpp"
#include "starpu_setup.hpp"

struct InferenceJob {
  torch::Tensor input_tensor;
  torch::Tensor output_tensor;
  int job_id;
  bool is_shutdown_signal = false;
};

void run_inference_loop(const ProgramOptions& opts, StarPUSetup& starpu);