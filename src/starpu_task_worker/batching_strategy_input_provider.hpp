#pragma once

#include <atomic>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>

#include "batching_strategy.hpp"

namespace starpu_server {

class InferenceJob;
class InferenceQueue;
struct RuntimeObservability;

class BatchingStrategyInputProvider {
 public:
  BatchingStrategyInputProvider() = default;
  virtual ~BatchingStrategyInputProvider() = default;
  BatchingStrategyInputProvider(const BatchingStrategyInputProvider&) = default;
  auto operator=(const BatchingStrategyInputProvider&)
      -> BatchingStrategyInputProvider& = default;
  BatchingStrategyInputProvider(BatchingStrategyInputProvider&&) = default;
  auto operator=(BatchingStrategyInputProvider&&)
      -> BatchingStrategyInputProvider& = default;

  [[nodiscard]] virtual auto sample() const -> BatchingStrategyInput = 0;
};

class RuntimeBatchingStrategyInputProvider final
    : public BatchingStrategyInputProvider {
 public:
  RuntimeBatchingStrategyInputProvider(
      const RuntimeConfig* opts, InferenceQueue* queue,
      std::shared_ptr<RuntimeObservability> observability,
      const std::deque<std::shared_ptr<InferenceJob>>* prepared_jobs,
      std::mutex* prepared_mutex,
      const std::atomic<std::size_t>* inflight_tasks,
      std::size_t max_inflight_tasks);

  [[nodiscard]] auto sample() const -> BatchingStrategyInput override;

 private:
  const RuntimeConfig* opts_;
  InferenceQueue* queue_;
  std::shared_ptr<RuntimeObservability> observability_;
  const std::deque<std::shared_ptr<InferenceJob>>* prepared_jobs_;
  std::mutex* prepared_mutex_;
  const std::atomic<std::size_t>* inflight_tasks_;
  std::size_t max_inflight_tasks_;
};

}  // namespace starpu_server
