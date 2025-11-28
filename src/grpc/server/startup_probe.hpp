#pragma once

#include <torch/torch.h>

#include <vector>

namespace starpu_server {

struct RuntimeConfig;
class StarPUSetup;

auto run_startup_throughput_probe(
    const RuntimeConfig& opts, StarPUSetup& starpu,
    torch::jit::script::Module& model_cpu,
    std::vector<torch::jit::script::Module>& models_gpu,
    const std::vector<torch::Tensor>& reference_outputs) -> double;

auto run_startup_throughput_probe_cpu(
    const RuntimeConfig& opts, StarPUSetup& starpu,
    torch::jit::script::Module& model_cpu,
    std::vector<torch::jit::script::Module>& models_gpu,
    const std::vector<torch::Tensor>& reference_outputs) -> double;

}  // namespace starpu_server
