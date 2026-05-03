#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "batch_capacity_policy.hpp"
#include "batch_composition_policy.hpp"
#include "batching_strategy.hpp"
#include "batching_strategy_input_provider.hpp"
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

struct BatchCollectorRuntimeContext {
  InferenceQueue* queue{};
  const RuntimeConfig* opts{};
  StarPUSetup* starpu{};
  std::shared_ptr<InferenceJob>* pending_job{};
  std::shared_ptr<RuntimeObservability> observability;
};

struct BatchCollectorBatchingDependencies {
  std::unique_ptr<BatchCapacityPolicy> capacity_policy;
  std::unique_ptr<BatchCompositionPolicy> composition_policy;
  std::unique_ptr<BatchingStrategyInputProvider> strategy_input_provider;
  std::unique_ptr<BatchingStrategy> strategy;
};

class BatchCollector {
 public:
  BatchCollector(
      BatchCollectorRuntimeContext runtime,
      const PreparedBatchingContext& prepared, const InflightContext& inflight,
      BatchCollectorBatchingDependencies batching_dependencies);

  auto wait_for_next_job() -> std::shared_ptr<InferenceJob>;
  auto collect_batch(const std::shared_ptr<InferenceJob>& first_job)
      -> std::vector<std::shared_ptr<InferenceJob>>;
  auto maybe_build_batched_job(std::vector<std::shared_ptr<InferenceJob>>& jobs)
      -> std::shared_ptr<InferenceJob>;
  void reset_prepared_queue_state();
  void abort_prepared_queue();
  void enqueue_prepared_job(const std::shared_ptr<InferenceJob>& job);
  auto wait_for_prepared_job() -> std::shared_ptr<InferenceJob>;
  void batching_loop();

 private:
  [[nodiscard]] auto try_acquire_next_job(
      bool enable_wait,
      task_runner_internal::Clock::time_point coalesce_deadline)
      -> std::shared_ptr<InferenceJob>;
  void store_pending_job(const std::shared_ptr<InferenceJob>& job);
  [[nodiscard]] auto is_batching_done() const -> bool;
  [[nodiscard]] auto should_abort_inflight_wait() const -> bool;

// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  friend class task_runner_internal::testing::BatchCollectorTestAdapter;
#endif  // SONAR_IGNORE_END
  // GCOVR_EXCL_STOP

  InferenceQueue* queue_;
  const RuntimeConfig* opts_;
  StarPUSetup* starpu_;
  std::shared_ptr<InferenceJob>* pending_job_;
  std::shared_ptr<RuntimeObservability> observability_;
  std::atomic<std::size_t>* inflight_tasks_;
  std::condition_variable* inflight_cv_;
  std::mutex* inflight_mutex_;
  std::size_t max_inflight_tasks_;
  std::mutex* prepared_mutex_;
  std::condition_variable* prepared_cv_;
  std::deque<std::shared_ptr<InferenceJob>>* prepared_jobs_;
  bool* batching_done_;
  std::unique_ptr<BatchCapacityPolicy> batch_capacity_policy_;
  std::unique_ptr<BatchCompositionPolicy> batch_composition_policy_;
  std::unique_ptr<BatchingStrategyInputProvider>
      batching_strategy_input_provider_;
  std::unique_ptr<BatchingStrategy> batching_strategy_;
};

}  // namespace starpu_server
