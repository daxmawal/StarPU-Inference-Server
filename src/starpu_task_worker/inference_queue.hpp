#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>

#include "monitoring/metrics.hpp"

namespace starpu_server {
class InferenceJob;
// =============================================================================
// Thread-safe job queue for asynchronous inference execution
// =============================================================================

class InferenceQueue {
 public:
  [[nodiscard]] bool push(const std::shared_ptr<InferenceJob>& job)
  {
    if (job == nullptr) {
      return false;
    }
    {
      const std::scoped_lock lock(mutex_);
      if (shutdown_) {
        return false;
      }
      queue_.push(job);
      set_queue_size(queue_.size());
    }
    cv_.notify_one();
    return true;
  }
  [[nodiscard]] bool wait_and_pop(std::shared_ptr<InferenceJob>& job)
  {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this] { return !queue_.empty() || shutdown_; });
    if (queue_.empty()) {
      return false;
    }
    job = queue_.front();
    queue_.pop();
    set_queue_size(queue_.size());
    return true;
  }

  void shutdown()
  {
    {
      const std::scoped_lock lock(mutex_);
      shutdown_ = true;
    }
    cv_.notify_all();
  }

  size_t size()
  {
    const std::scoped_lock lock(mutex_);
    return queue_.size();
  }

 private:
  std::queue<std::shared_ptr<InferenceJob>> queue_;
  bool shutdown_ = false;
  std::mutex mutex_;
  std::condition_variable cv_;
};
}  // namespace starpu_server
