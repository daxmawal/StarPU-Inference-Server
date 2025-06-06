#pragma once

#include "Inference_queue.hpp"
#include "starpu_setup.hpp"

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

  void run(unsigned int iterations_per_worker);

 private:
  void client_worker(
      const std::map<unsigned int, std::vector<int32_t>>& device_workers,
      InferenceQueue& queue, unsigned int iterations_per_worker) const;

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

  // *****************************************************************************
  // Dummy result collection and synchronization primitives
  // *****************************************************************************
  std::atomic<unsigned int>
      dummy_completed_jobs_;          // Tracks how many jobs completed
  std::mutex dummy_mutex_;            // Protects dummy CV
  std::mutex dummy_results_mutex_;    // Protects dummy result list
  std::condition_variable dummy_cv_;  // Used to wait for all warmup jobs
  std::vector<InferenceResult>
      dummy_results_;  // Results discarded after warmup
};