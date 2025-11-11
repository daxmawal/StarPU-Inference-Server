#pragma once

#include <torch/script.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "inference_queue.hpp"
#include "inference_runner.hpp"
#include "runtime_config.hpp"
#include "starpu_setup.hpp"

namespace starpu_server {
class BatchCollector;
class SlotManager;
class ResultDispatcher;

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
  ~StarPUTaskRunner();

  using DurationMs = std::chrono::duration<double, std::milli>;

  void run();
  auto wait_for_next_job() -> std::shared_ptr<InferenceJob>;
  [[nodiscard]] auto should_shutdown(
      const std::shared_ptr<InferenceJob>& job) const -> bool;
  void prepare_job_completion_callback(
      const std::shared_ptr<InferenceJob>& job);
  void submit_inference_task(const std::shared_ptr<InferenceJob>& job);
  static auto handle_job_exception(
      const std::shared_ptr<InferenceJob>& job,
      const std::exception& exception) -> bool;
  void log_job_timings(
      int request_id, DurationMs latency,
      const detail::TimingInfo& timing_info) const;

 private:
  friend class StarPUTaskRunnerTestAdapter;
  friend class SlotManager;
  friend class ResultDispatcher;
  friend class BatchCollector;

  struct PoolResources {
    InputSlotPool* input_pool = nullptr;
    OutputSlotPool* output_pool = nullptr;
    int input_slot = -1;
    int output_slot = -1;

    [[nodiscard]] auto has_input() const -> bool
    {
      return input_pool != nullptr;
    }
    [[nodiscard]] auto has_output() const -> bool
    {
      return output_pool != nullptr;
    }
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
  void batching_loop();
  void enqueue_prepared_job(const std::shared_ptr<InferenceJob>& job);
  auto wait_for_prepared_job() -> std::shared_ptr<InferenceJob>;
  static auto can_merge_jobs(
      const std::shared_ptr<InferenceJob>& lhs,
      const std::shared_ptr<InferenceJob>& rhs) -> bool;
  static auto merge_input_tensors(
      const std::vector<std::shared_ptr<InferenceJob>>& jobs,
      int64_t total_samples) -> std::vector<torch::Tensor>;
  static auto merge_input_memory_holders(
      const std::vector<std::shared_ptr<InferenceJob>>& jobs)
      -> std::vector<std::shared_ptr<const void>>;
  static void propagate_completion_to_sub_jobs(
      const std::shared_ptr<InferenceJob>& aggregated_job,
      std::vector<torch::Tensor> aggregated_outputs, double latency_ms);
  static auto configure_task_context(
      InferenceTask& task, const PoolResources& pools,
      const std::vector<starpu_data_handle_t>& input_handles,
      const std::vector<starpu_data_handle_t>& output_handles,
      int64_t batch_size) -> std::shared_ptr<InferenceCallbackContext>;
  [[noreturn]] static void handle_submission_failure(
      const PoolResources& pools,
      const std::shared_ptr<InferenceCallbackContext>& ctx, int submit_code);
  [[nodiscard]] auto resolve_batch_size(
      const std::shared_ptr<InferenceJob>& job) const -> int64_t;
  void release_pending_jobs(
      const std::shared_ptr<InferenceJob>& job,
      std::vector<std::shared_ptr<InferenceJob>>& pending_jobs) const;
  void store_completed_job_result(
      const std::shared_ptr<InferenceJob>& job,
      const std::vector<torch::Tensor>& results, double latency_ms) const;
  void ensure_callback_timing(detail::TimingInfo& timing) const;
  void record_job_metrics(
      const std::shared_ptr<InferenceJob>& job, DurationMs latency,
      std::size_t batch_size) const;
  void finalize_job_completion(const std::shared_ptr<InferenceJob>& job) const;

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
  std::jthread batching_thread_;
  std::mutex prepared_mutex_;
  std::condition_variable prepared_cv_;
  std::deque<std::shared_ptr<InferenceJob>> prepared_jobs_;
  bool batching_done_ = false;

  std::unique_ptr<BatchCollector> batch_collector_;
  std::unique_ptr<SlotManager> slot_manager_;
  std::unique_ptr<ResultDispatcher> result_dispatcher_;
};
}  // namespace starpu_server
