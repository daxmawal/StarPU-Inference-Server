#pragma once

#include "starpu_setup.hpp"

// =============================================================================
// run_warmup_phase: Executes a warmup phase before timed inference
// =============================================================================
/// This function launches a short phase of inference jobs to "warm up" the
/// StarPU scheduler and CUDA context. It launches a server thread that runs
/// inference tasks and a client thread that pushes warmup jobs into the queue.
// =============================================================================
void run_warmup_phase(
    const ProgramOptions& opts, StarPUSetup& starpu,
    torch::jit::script::Module& model_cpu,
    std::vector<torch::jit::script::Module>& models_gpu,
    const std::vector<torch::Tensor>& outputs_ref,
    unsigned int iterations_per_worker);