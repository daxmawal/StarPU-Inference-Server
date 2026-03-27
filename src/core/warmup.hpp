#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>

#include "inference_queue.hpp"
#include "starpu_setup.hpp"

namespace starpu_server {
struct RuntimeObservability;
struct WarmupRunnerTestHelper;
// =============================================================================
// Runs a warmup phase by simulating inference jobs across StarPU workers
// =============================================================================

class WarmupRunner {
 public:
  using CompletionObserver =
      std::function<void(std::atomic<std::size_t>& dummy_completed_jobs)>;

  WarmupRunner(
      const RuntimeConfig& opts, StarPUSetup& starpu,
      torch::jit::script::Module& model_cpu,
      std::vector<torch::jit::script::Module>& models_gpu,
      const std::vector<torch::Tensor>& outputs_ref,
      CompletionObserver completion_observer = {},
      std::shared_ptr<RuntimeObservability> observability = {});
  ~WarmupRunner() = default;
  WarmupRunner(const WarmupRunner&) = delete;
  auto operator=(const WarmupRunner&) -> WarmupRunner& = delete;
  WarmupRunner(WarmupRunner&&) = delete;
  auto operator=(WarmupRunner&&) -> WarmupRunner& = delete;

  void run(int request_nb_per_worker);

 private:
  friend struct WarmupRunnerTestHelper;
  auto client_worker(
      const std::map<int, std::vector<int>>& device_workers,
      InferenceQueue& queue, int request_nb_per_worker) const -> std::size_t;

  // *****************************************************************************
  // Configuration and model references (owned externally)
  // *****************************************************************************
  const RuntimeConfig& opts_;
  StarPUSetup& starpu_;
  torch::jit::script::Module& model_cpu_;
  std::vector<torch::jit::script::Module>& models_gpu_;
  const std::vector<torch::Tensor>& outputs_ref_;
  CompletionObserver completion_observer_;
  std::shared_ptr<RuntimeObservability> observability_;
};
}  // namespace starpu_server
