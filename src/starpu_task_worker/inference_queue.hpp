#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <memory>
#include <mutex>
#include <queue>
#include <utility>

#include "monitoring/metrics.hpp"

namespace starpu_server {
class InferenceJob;
// =============================================================================
// Thread-safe job queue for asynchronous inference execution
// =============================================================================

class InferenceQueue {
 public:
  [[nodiscard]] auto push(std::shared_ptr<InferenceJob> job) -> bool
  {
    if (job == nullptr) {
      return false;
    }
    {
      const std::scoped_lock lock(mutex_);
      if (shutdown_) {
        return false;
      }
      queue_.push(std::move(job));
      set_queue_size(queue_.size());
    }
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
    set_queue_size(queue_.size());
    return true;
  }
  [[nodiscard]] auto try_pop(std::shared_ptr<InferenceJob>& job) -> bool
  {
    const std::scoped_lock lock(mutex_);
    if (queue_.empty()) {
      return false;
    }
    job = std::move(queue_.front());
    queue_.pop();
    set_queue_size(queue_.size());
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
      set_queue_size(queue_.size());
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
  mutable std::mutex mutex_;
  std::queue<std::shared_ptr<InferenceJob>> queue_;
  bool shutdown_ = false;
  std::condition_variable cv_;
};
}  // namespace starpu_server
