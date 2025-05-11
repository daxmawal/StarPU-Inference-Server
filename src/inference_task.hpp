#pragma once
#include <torch/torch.h>
#include <string>
#include <chrono>
#include "args_parser.hpp"
#include "starpu_setup.hpp"

void output_tensor_ready_callback(void* arg);

void submit_inference_task(StarPUSetup&                starpu,
                           const torch::Tensor&        input_tensor,
                           torch::Tensor&              output_tensor,
                           torch::jit::script::Module& module,
                           const ProgramOptions&       opts,
                           const torch::Tensor&        output_direct,
                           int                         iteration,
                           std::chrono::high_resolution_clock::time_point start_time);