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
#include <string_view>
#include <thread>
#include <vector>

#include "inference_queue.hpp"
#include "inference_runner.hpp"
#include "inference_task.hpp"
#include "runtime_config.hpp"
#include "starpu_setup.hpp"

namespace starpu_server {
class BatchCollector;
class SlotManager;
class ResultDispatcher;
class StarPUTaskRunner;

// GCOVR_EXCL_START
#if defined(STARPU_TESTING)
namespace task_runner_helpers {
void ensure_callback_timing(detail::TimingInfo& timing);
void record_job_metrics(
    StarPUTaskRunner& runner, const std::shared_ptr<InferenceJob>& job,
    std::chrono::duration<double, std::milli> latency, std::size_t batch_size);
void finalize_job_completion(
    StarPUTaskRunner& runner, const std::shared_ptr<InferenceJob>& job);
}  // namespace task_runner_helpers
#endif
// GCOVR_EXCL_STOP

// ============================================================================
// StarPUTaskRunner
// ----------------------------------------------------------------------------
// Threaded worker responsible for:
//  - Pulling jobs from the inference queue
//  - Submitting them to StarPU
//  - Collecting and storing results
// ============================================================================
struct StarPUTaskRunnerConfig {
  InferenceQueue* queue{};
  torch::jit::script::Module* model_cpu{};
  std::vector<torch::jit::script::Module>* models_gpu{};
  StarPUSetup* starpu{};
  const RuntimeConfig* opts{};
  std::atomic<int>* completed_jobs{};
  std::condition_variable* all_done_cv{};
  const InferenceTaskDependencies* dependencies{};
};

class StarPUTaskRunner {
 public:
  explicit StarPUTaskRunner(const StarPUTaskRunnerConfig& config);
  ~StarPUTaskRunner();
  StarPUTaskRunner(const StarPUTaskRunner&) = delete;
  auto operator=(const StarPUTaskRunner&) -> StarPUTaskRunner& = delete;
  StarPUTaskRunner(StarPUTaskRunner&&) = delete;
  auto operator=(StarPUTaskRunner&&) -> StarPUTaskRunner& = delete;

  using DurationMs = std::chrono::duration<double, std::milli>;

  void run();
// GCOVR_EXCL_START
#if defined(STARPU_TESTING)
  auto wait_for_next_job() -> std::shared_ptr<InferenceJob>;
#endif
  // GCOVR_EXCL_STOP
  void prepare_job_completion_callback(
      const std::shared_ptr<InferenceJob>& job);
  void submit_inference_task(const std::shared_ptr<InferenceJob>& job);
  static auto handle_job_exception(
      const std::shared_ptr<InferenceJob>& job,
      const std::exception& exception) -> bool;
// GCOVR_EXCL_START
#if defined(STARPU_TESTING)
  void log_job_timings(
      int request_id, DurationMs latency,
      const detail::TimingInfo& timing_info) const;
#endif
  // GCOVR_EXCL_STOP

 private:
// GCOVR_EXCL_START
#if defined(STARPU_TESTING)
  friend void task_runner_helpers::record_job_metrics(
      StarPUTaskRunner& runner, const std::shared_ptr<InferenceJob>& job,
      std::chrono::duration<double, std::milli> latency,
      std::size_t batch_size);
  friend void task_runner_helpers::finalize_job_completion(
      StarPUTaskRunner& runner, const std::shared_ptr<InferenceJob>& job);
#endif
  // GCOVR_EXCL_STOP
  friend struct InflightReleaseGuard;

// GCOVR_EXCL_START
#if defined(STARPU_TESTING)
  friend class StarPUTaskRunnerTestAdapter;
#endif
  // GCOVR_EXCL_STOP
  friend class SlotManager;
  friend class ResultDispatcher;
  friend class BatchCollector;

  struct SubmissionInfo {
    int submission_id;
    int job_id;
  };

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
// GCOVR_EXCL_START
#if defined(STARPU_TESTING)
  [[nodiscard]] auto collect_batch(
      const std::shared_ptr<InferenceJob>& first_job)
      -> std::vector<std::shared_ptr<InferenceJob>>;
  auto maybe_build_batched_job(std::vector<std::shared_ptr<InferenceJob>>& jobs)
      -> std::shared_ptr<InferenceJob>;
#endif
  // GCOVR_EXCL_STOP
  void batching_loop();
// GCOVR_EXCL_START
#if defined(STARPU_TESTING)
  void enqueue_prepared_job(const std::shared_ptr<InferenceJob>& job);
#endif
  // GCOVR_EXCL_STOP
  auto wait_for_prepared_job() -> std::shared_ptr<InferenceJob>;
// GCOVR_EXCL_START
#if defined(STARPU_TESTING)
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
      const std::vector<torch::Tensor>& aggregated_outputs, double latency_ms);
#endif
  // GCOVR_EXCL_STOP
  static auto configure_task_context(
      InferenceTask& task, const PoolResources& pools,
      std::vector<starpu_data_handle_t> input_handles,
      std::vector<starpu_data_handle_t> output_handles,
      int64_t batch_size) -> std::shared_ptr<InferenceCallbackContext>;
  [[noreturn]] static void handle_submission_failure(
      const PoolResources& pools,
      const std::shared_ptr<InferenceCallbackContext>& ctx, int submit_code);
  [[nodiscard]] auto resolve_batch_size(
      const std::shared_ptr<InferenceJob>& job) const -> int64_t;
// GCOVR_EXCL_START
#if defined(STARPU_TESTING)
  static void release_pending_jobs(
      const std::shared_ptr<InferenceJob>& job,
      std::vector<std::shared_ptr<InferenceJob>>& pending_jobs);
#endif
  // GCOVR_EXCL_STOP
  void trace_batch_if_enabled(
      const std::shared_ptr<InferenceJob>& job, bool warmup_job,
      int submission_id) const;
  void submit_job_or_handle_failure(
      const std::shared_ptr<InferenceJob>& job, SubmissionInfo submission_info);
  void finalize_job_after_exception(
      const std::shared_ptr<InferenceJob>& job, const std::exception& exception,
      std::string_view log_prefix, int job_id);
// GCOVR_EXCL_START
#if defined(STARPU_TESTING)
  void reserve_inflight_slot();
#endif
  // GCOVR_EXCL_STOP
  void release_inflight_slot();
  [[nodiscard]] auto has_inflight_limit() const -> bool
  {
    return inflight_state_.max_tasks > 0;
  }

  struct InflightState {
    std::atomic<std::size_t> tasks{0};
    std::size_t max_tasks{0};
    std::mutex mutex;
    std::condition_variable cv;
  };

  struct PreparedState {
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::shared_ptr<InferenceJob>> jobs;
    bool batching_done = false;
  };

  InferenceQueue* queue_;
  torch::jit::script::Module* model_cpu_;
  std::vector<torch::jit::script::Module>* models_gpu_;
  StarPUSetup* starpu_;
  const RuntimeConfig* opts_;

  std::atomic<int>* completed_jobs_;
  std::condition_variable* all_done_cv_;
  InferenceTaskDependencies dependencies_;
  std::shared_ptr<InferenceJob> pending_job_;
  std::atomic<int> next_submission_id_{0};
  std::jthread batching_thread_;

  InflightState inflight_state_;
  PreparedState prepared_state_;

  std::unique_ptr<BatchCollector> batch_collector_;
  std::unique_ptr<SlotManager> slot_manager_;
  std::unique_ptr<ResultDispatcher> result_dispatcher_;
};
}  // namespace starpu_server
