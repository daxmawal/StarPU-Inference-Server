#pragma once

#include "inference_queue.hpp"
#include "starpu_setup.hpp"

namespace starpu_server {
// =============================================================================
// Runs a warmup phase by simulating inference jobs across StarPU workers
// =============================================================================

class WarmupRunner {
 public:
  WarmupRunner(
      const RuntimeConfig& opts, StarPUSetup& starpu,
      torch::jit::script::Module& model_cpu,
      std::vector<torch::jit::script::Module>& models_gpu,
      const std::vector<torch::Tensor>& outputs_ref);
  ~WarmupRunner() = default;
  WarmupRunner(const WarmupRunner&) = delete;
  auto operator=(const WarmupRunner&) -> WarmupRunner& = delete;
  WarmupRunner(WarmupRunner&&) = delete;
  auto operator=(WarmupRunner&&) -> WarmupRunner& = delete;

  void run(int iterations_per_worker);

 private:
  void client_worker(
      const std::map<int, std::vector<int32_t>>& device_workers,
      InferenceQueue& queue, int iterations_per_worker) const;

  // *****************************************************************************
  // Configuration and model references (owned externally)
  // *****************************************************************************
  const RuntimeConfig& opts_;              // Runtime options
  StarPUSetup& starpu_;                    // StarPU configuration and codelet
  torch::jit::script::Module& model_cpu_;  // CPU model (TorchScript)
  std::vector<torch::jit::script::Module>&
      models_gpu_;  // GPU models (TorchScript)
  const std::vector<torch::Tensor>&
      outputs_ref_;  // Reference outputs to validate warmup jobs
};
}  // namespace starpu_server
