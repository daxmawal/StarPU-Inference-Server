#pragma once
#include "starpu_setup.hpp"

void run_warmup_phase(
    const ProgramOptions& opts, StarPUSetup& starpu,
    const torch::jit::script::Module& model_cpu,
    const torch::jit::script::Module& model_gpu,
    const torch::Tensor& output_ref, const unsigned int warmup_iterations);