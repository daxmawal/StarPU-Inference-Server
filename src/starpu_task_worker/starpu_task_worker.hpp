#pragma once

#include <torch/script.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <vector>

#include "inference_queue.hpp"
#include "inference_runner.hpp"
#include "runtime_config.hpp"
#include "starpu_setup.hpp"

namespace starpu_server {
class InferenceTask;
struct InferenceCallbackContext;
// ============================================================================
// StarPUTaskRunner
// ----------------------------------------------------------------------------
// Threaded worker responsible for:
//  - Pulling jobs from the inference queue
//  - Submitting them to StarPU
//  - Collecting and storing results
// ============================================================================
struct InferenceTaskDependencies;

struct StarPUTaskRunnerConfig {
  InferenceQueue* queue{};
  torch::jit::script::Module* model_cpu{};
  std::vector<torch::jit::script::Module>* models_gpu{};
  StarPUSetup* starpu{};
  const RuntimeConfig* opts{};
  std::vector<InferenceResult>* results{};
  std::mutex* results_mutex{};
  std::atomic<int>* completed_jobs{};
  std::condition_variable* all_done_cv{};
  const InferenceTaskDependencies* dependencies{};
};

class StarPUTaskRunner {
 public:
  explicit StarPUTaskRunner(const StarPUTaskRunnerConfig& config);

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
  friend class StarPUTaskRunnerTestAdapter;

  struct PoolResources {
    InputSlotPool* input_pool = nullptr;
    OutputSlotPool* output_pool = nullptr;
    int input_slot = -1;
    int output_slot = -1;

    [[nodiscard]] bool has_input() const { return input_pool != nullptr; }
    [[nodiscard]] bool has_output() const { return output_pool != nullptr; }
  };

  [[nodiscard]] auto acquire_pools() -> PoolResources;
  auto validate_batch_and_copy_inputs(
      const std::shared_ptr<InferenceJob>& job,
      const PoolResources& pools) -> int64_t;
  [[nodiscard]] auto collect_batch(
      const std::shared_ptr<InferenceJob>& first_job)
      -> std::vector<std::shared_ptr<InferenceJob>>;
  auto maybe_build_batched_job(std::vector<std::shared_ptr<InferenceJob>>& jobs)
      -> std::shared_ptr<InferenceJob>;
  static auto can_merge_jobs(
      const std::shared_ptr<InferenceJob>& lhs,
      const std::shared_ptr<InferenceJob>& rhs) -> bool;
  static auto merge_input_tensors(
      const std::vector<std::shared_ptr<InferenceJob>>& jobs)
      -> std::vector<torch::Tensor>;
  static auto merge_input_memory_holders(
      const std::vector<std::shared_ptr<InferenceJob>>& jobs)
      -> std::vector<std::shared_ptr<const void>>;
  static void propagate_completion_to_sub_jobs(
      const std::shared_ptr<InferenceJob>& aggregated_job,
      const std::vector<torch::Tensor>& aggregated_outputs, double latency_ms);
  static auto configure_task_context(
      InferenceTask& task, const PoolResources& pools,
      const std::vector<starpu_data_handle_t>& input_handles,
      const std::vector<starpu_data_handle_t>& output_handles,
      int64_t batch_size) -> std::shared_ptr<InferenceCallbackContext>;
  [[noreturn]] static void handle_submission_failure(
      const PoolResources& pools,
      const std::shared_ptr<InferenceCallbackContext>& ctx, int submit_code);

  InferenceQueue* queue_;
  torch::jit::script::Module* model_cpu_;
  std::vector<torch::jit::script::Module>* models_gpu_;
  StarPUSetup* starpu_;
  const RuntimeConfig* opts_;

  std::vector<InferenceResult>* results_;
  std::mutex* results_mutex_;
  std::atomic<int>* completed_jobs_;
  std::condition_variable* all_done_cv_;
  const InferenceTaskDependencies* dependencies_;
  std::shared_ptr<InferenceJob> pending_job_;
  std::atomic<int> next_submission_id_{0};
};
}  // namespace starpu_server
