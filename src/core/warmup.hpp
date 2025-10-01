#pragma once

#include <atomic>
#include <functional>

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

  using WarmupRunnerTestHook =
      std::function<void(std::atomic<int>& dummy_completed_jobs)>;
  static void set_test_hook(WarmupRunnerTestHook hook);
  static void clear_test_hook();

 private:
  void client_worker(
      const std::map<int, std::vector<int32_t>>& device_workers,
      InferenceQueue& queue, int iterations_per_worker) const;

  // *****************************************************************************
  // Configuration and model references (owned externally)
  // *****************************************************************************
  const RuntimeConfig& opts_;
  StarPUSetup& starpu_;
  torch::jit::script::Module& model_cpu_;
  std::vector<torch::jit::script::Module>& models_gpu_;
  const std::vector<torch::Tensor>& outputs_ref_;
};
}  // namespace starpu_server
