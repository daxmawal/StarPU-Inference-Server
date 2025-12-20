#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <utility>

#include "monitoring/metrics.hpp"
#include "utils/batching_trace_logger.hpp"
#include "utils/runtime_config.hpp"

namespace starpu_server {
class InferenceJob;
// =============================================================================
// Thread-safe job queue for asynchronous inference execution
// =============================================================================

class InferenceQueue {
 public:
  explicit InferenceQueue(std::size_t max_size = kDefaultMaxQueueSize)
      : max_size_(max_size)
  {
    if (max_size_ == 0) {
      throw std::invalid_argument("max_queue_size must be > 0");
    }
    set_queue_capacity(max_size_);
  }

  [[nodiscard]] auto push(
      std::shared_ptr<InferenceJob> job, bool* queue_full = nullptr) -> bool
  {
    if (queue_full != nullptr) {
      *queue_full = false;
    }
    if (job == nullptr) {
      return false;
    }
    std::size_t size = 0;
    {
      const std::scoped_lock lock(mutex_);
      if (shutdown_) {
        return false;
      }
      if (queue_.size() >= max_size_) {
        if (queue_full != nullptr) {
          *queue_full = true;
        }
        return false;
      }
      queue_.push(std::move(job));
      size = queue_.size();
    }
    update_queue_metrics(size);
    cv_.notify_one();
    return true;
  }
  [[nodiscard]] auto wait_and_pop(std::shared_ptr<InferenceJob>& job) -> bool
  {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this] { return !queue_.empty() || shutdown_; });
    if (queue_.empty()) {
      return false;
    }
    job = std::move(queue_.front());
    queue_.pop();
    const auto size = queue_.size();
    lock.unlock();
    update_queue_metrics(size);
    return true;
  }
  [[nodiscard]] auto try_pop(std::shared_ptr<InferenceJob>& job) -> bool
  {
    std::unique_lock lock(mutex_);
    if (queue_.empty()) {
      return false;
    }
    job = std::move(queue_.front());
    queue_.pop();
    const auto size = queue_.size();
    lock.unlock();
    update_queue_metrics(size);
    return true;
  }
  template <typename Rep, typename Period>
  [[nodiscard]] auto wait_for_and_pop(
      std::shared_ptr<InferenceJob>& job,
      const std::chrono::duration<Rep, Period>& timeout) -> bool
  {
    if (std::unique_lock lock(mutex_); cv_.wait_for(
            lock, timeout, [this] { return !queue_.empty() || shutdown_; })) {
      if (queue_.empty()) {
        return false;
      }
      job = std::move(queue_.front());
      queue_.pop();
      const auto size = queue_.size();
      lock.unlock();
      update_queue_metrics(size);
      return true;
    }
    return false;
  }


  void shutdown()
  {
    {
      const std::scoped_lock lock(mutex_);
      shutdown_ = true;
    }
    cv_.notify_all();
  }

  [[nodiscard]] auto size() const -> std::size_t
  {
    const std::scoped_lock lock(mutex_);
    return queue_.size();
  }

 private:
  static void update_queue_metrics(std::size_t size)
  {
    set_queue_size(size);
    auto& tracer = BatchingTraceLogger::instance();
    tracer.log_queue_size(size);
  }

  const std::size_t max_size_;
  mutable std::mutex mutex_;
  std::queue<std::shared_ptr<InferenceJob>> queue_;
  bool shutdown_ = false;
  std::condition_variable cv_;
};
}  // namespace starpu_server
