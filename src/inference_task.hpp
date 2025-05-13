#pragma once
#include <torch/torch.h>

#include <chrono>
#include <string>

#include "args_parser.hpp"
#include "inference_runner.hpp"
#include "starpu_setup.hpp"

void output_tensor_ready_callback(void* arg);

void submit_inference_task(
    StarPUSetup& starpu, std::shared_ptr<InferenceJob> job,
    torch::jit::script::Module& module, const ProgramOptions& opts);