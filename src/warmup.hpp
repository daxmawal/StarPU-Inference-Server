#pragma once

#include "Inference_queue.hpp"
#include "starpu_setup.hpp"

// =============================================================================
// run_warmup_phase: Executes a warmup phase before timed inference
// =============================================================================
class WarmupRunner {
 public:
  WarmupRunner(
      const RuntimeConfig& opts, StarPUSetup& starpu,
      torch::jit::script::Module& model_cpu,
      std::vector<torch::jit::script::Module>& models_gpu,
      const std::vector<torch::Tensor>& outputs_ref);

  void run(const int iterations_per_worker);

 private:
  void client_worker(
      const std::map<unsigned int, std::vector<int32_t>> device_workers,
      InferenceQueue& queue, const int iterations_per_worker);

  const RuntimeConfig& opts_;
  StarPUSetup& starpu_;
  torch::jit::script::Module& model_cpu_;
  std::vector<torch::jit::script::Module>& models_gpu_;
  const std::vector<torch::Tensor>& outputs_ref_;

  InferenceQueue queue_;
  std::atomic<unsigned int> dummy_completed_jobs_;
  std::mutex dummy_mutex_;
  std::mutex dummy_results_mutex_;
  std::condition_variable dummy_cv_;
  std::vector<InferenceResult> dummy_results_;
};