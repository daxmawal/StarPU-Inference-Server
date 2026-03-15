#pragma once

#include <torch/torch.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "inference_queue.hpp"
#include "starpu_setup.hpp"
#include "task_runner_internal.hpp"

namespace starpu_server {

class ResultDispatcher;

struct PreparedBatchingContext {
  std::mutex* prepared_mutex{};
  std::condition_variable* prepared_cv{};
  std::deque<std::shared_ptr<InferenceJob>>* prepared_jobs{};
  bool* batching_done{};
};

struct InflightContext {
  std::atomic<std::size_t>* inflight_tasks{};
  std::condition_variable* inflight_cv{};
  std::mutex* inflight_mutex{};
  std::size_t max_inflight_tasks{0};
};

class BatchCollector {
 public:
  BatchCollector(
      InferenceQueue* queue, const RuntimeConfig* opts, StarPUSetup* starpu,
      std::shared_ptr<InferenceJob>* pending_job,
      const PreparedBatchingContext& prepared, const InflightContext& inflight);

  auto wait_for_next_job() -> std::shared_ptr<InferenceJob>;
  auto collect_batch(const std::shared_ptr<InferenceJob>& first_job)
      -> std::vector<std::shared_ptr<InferenceJob>>;
  auto maybe_build_batched_job(std::vector<std::shared_ptr<InferenceJob>>& jobs)
      -> std::shared_ptr<InferenceJob>;
  void enqueue_prepared_job(const std::shared_ptr<InferenceJob>& job);
  auto wait_for_prepared_job() -> std::shared_ptr<InferenceJob>;
  void batching_loop();

  static auto can_merge_jobs(
      const std::shared_ptr<InferenceJob>& lhs,
      const std::shared_ptr<InferenceJob>& rhs) -> bool;
  static auto merge_input_tensors(
      const std::vector<std::shared_ptr<InferenceJob>>& jobs,
      int64_t total_samples) -> std::vector<torch::Tensor>;
  static auto merge_input_memory_holders(
      const std::vector<std::shared_ptr<InferenceJob>>& jobs)
      -> std::vector<std::shared_ptr<const void>>;

 private:
  struct BatchPressureState {
    bool congested = false;
    bool high = false;
    bool low = false;
    bool severe = false;
  };
  struct BatchPressureSample {
    BatchPressureState state{};
    std::optional<task_runner_internal::Clock::time_point> monitor_tick{};
  };

  [[nodiscard]] auto job_sample_size(
      const std::shared_ptr<InferenceJob>& job) const -> int64_t;
  [[nodiscard]] auto sample_limit_per_batch() const -> int;
  [[nodiscard]] auto effective_batch_limit() -> int;
  void update_adaptive_batch_target(int batch_limit);
  [[nodiscard]] auto should_refresh_adaptive_target(
      const BatchPressureSample& pressure) -> bool;
  [[nodiscard]] auto high_pressure_step(int batch_limit, bool severe) const
      -> int;
  [[nodiscard]] auto low_pressure_streak_threshold() const -> int;
  [[nodiscard]] auto sample_batch_pressure() const -> BatchPressureSample;
  [[nodiscard]] auto try_acquire_next_job(
      bool enable_wait,
      task_runner_internal::Clock::time_point coalesce_deadline)
      -> std::shared_ptr<InferenceJob>;
  void store_pending_job(const std::shared_ptr<InferenceJob>& job);
  [[nodiscard]] auto is_batching_done() const -> bool;
  [[nodiscard]] auto should_abort_inflight_wait() const -> bool;
  [[nodiscard]] static auto should_hold_job(
      const std::shared_ptr<InferenceJob>& candidate,
      const std::shared_ptr<InferenceJob>& reference,
      const std::optional<int>& target_worker) -> bool;
  [[nodiscard]] auto exceeds_sample_limit(
      int64_t accumulated_samples, const std::shared_ptr<InferenceJob>& job,
      int64_t max_samples_cap) const -> bool;

// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  friend auto task_runner_internal::testing::batch_collector_job_sample_size(
      const BatchCollector* collector,
      const std::shared_ptr<InferenceJob>& job) -> int64_t;
  friend auto
  task_runner_internal::testing::batch_collector_exceeds_sample_limit(
      const BatchCollector* collector, int64_t accumulated_samples,
      const std::shared_ptr<InferenceJob>& job,
      int64_t max_samples_cap) -> bool;
  friend auto
  task_runner_internal::testing::batch_collector_try_acquire_next_job(
      BatchCollector* collector, bool enable_wait,
      task_runner_internal::Clock::time_point coalesce_deadline)
      -> std::shared_ptr<InferenceJob>;
  friend auto task_runner_internal::testing::batch_collector_should_hold_job(
      const std::shared_ptr<InferenceJob>& candidate,
      const std::shared_ptr<InferenceJob>& reference,
      const std::optional<int>& target_worker) -> bool;
  friend auto task_runner_internal::testing::batch_collector_is_batching_done(
      const BatchCollector* collector) -> bool;
  friend auto
  task_runner_internal::testing::batch_collector_should_abort_inflight_wait(
      const BatchCollector* collector) -> bool;
  friend void
  task_runner_internal::testing::batch_collector_disable_prepared_job_sync(
      BatchCollector* collector);
  friend void task_runner_internal::testing::batch_collector_set_queue(
      BatchCollector* collector, InferenceQueue* queue);
  friend auto task_runner_internal::testing::batch_collector_get_queue(
      const BatchCollector* collector) -> InferenceQueue*;
  friend void
  task_runner_internal::testing::batch_collector_set_batching_done_ptr(
      BatchCollector* collector, bool* batching_done);
  friend void
  task_runner_internal::testing::batch_collector_set_batching_done_value(
      BatchCollector* collector, bool batching_done);
  friend void task_runner_internal::testing::batch_collector_set_pending_job(
      BatchCollector* collector, const std::shared_ptr<InferenceJob>& job);
#endif  // SONAR_IGNORE_END
  // GCOVR_EXCL_STOP

  InferenceQueue* queue_;
  const RuntimeConfig* opts_;
  StarPUSetup* starpu_;
  std::shared_ptr<InferenceJob>* pending_job_;
  std::atomic<std::size_t>* inflight_tasks_;
  std::condition_variable* inflight_cv_;
  std::mutex* inflight_mutex_;
  std::size_t max_inflight_tasks_;
  std::mutex* prepared_mutex_;
  std::condition_variable* prepared_cv_;
  std::deque<std::shared_ptr<InferenceJob>>* prepared_jobs_;
  bool* batching_done_;
  int adaptive_target_batch_size_ = 1;
  bool adaptive_target_initialized_ = false;
  int low_pressure_streak_ = 0;
  std::optional<task_runner_internal::Clock::time_point>
      last_adaptive_update_marker_{};
};

}  // namespace starpu_server
