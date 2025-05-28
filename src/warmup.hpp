#pragma once
#include "starpu_setup.hpp"

void run_warmup_phase(
    const ProgramOptions& opts, StarPUSetup& starpu,
    torch::jit::script::Module& model_cpu,
    std::vector<torch::jit::script::Module>& models_gpu,
    const torch::Tensor& output_ref, const unsigned int warmup_iterations);