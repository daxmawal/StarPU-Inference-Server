#pragma once

#include <torch/script.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <vector>

#include "inference_queue.hpp"
#include "inference_runner.hpp"
#include "runtime_config.hpp"
#include "starpu_setup.hpp"

// ============================================================================
// StarPUTaskRunner
// ----------------------------------------------------------------------------
// Threaded worker responsible for:
//  - Pulling jobs from the inference queue
//  - Submitting them to StarPU
//  - Collecting and storing results
// ============================================================================
class StarPUTaskRunner {
 public:
  StarPUTaskRunner(
      InferenceQueue* queue, torch::jit::script::Module* model_cpu,
      std::vector<torch::jit::script::Module>* models_gpu, StarPUSetup* starpu,
      const RuntimeConfig* opts, std::vector<InferenceResult>* results,
      std::mutex* results_mutex, std::atomic<int>* completed_jobs,
      std::condition_variable* all_done_cv);

  void run();
  auto wait_for_next_job() -> std::shared_ptr<InferenceJob>;
  [[nodiscard]] auto should_shutdown(
      const std::shared_ptr<InferenceJob>& job) const -> bool;
  void prepare_job_completion_callback(
      const std::shared_ptr<InferenceJob>& job);
  void submit_inference_task(const std::shared_ptr<InferenceJob>& job);
  static void handle_job_exception(
      const std::shared_ptr<InferenceJob>& job,
      const std::exception& exception);
  void log_job_timings(
      int job_id, double latency_ms,
      const detail::TimingInfo& timing_info) const;

 private:
  InferenceQueue* queue_;
  torch::jit::script::Module* model_cpu_;
  std::vector<torch::jit::script::Module>* models_gpu_;
  StarPUSetup* starpu_;
  const RuntimeConfig* opts_;

  std::vector<InferenceResult>* results_;
  std::mutex* results_mutex_;
  std::atomic<int>* completed_jobs_;
  std::condition_variable* all_done_cv_;
};
