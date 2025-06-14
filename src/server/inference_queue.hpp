#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>

#include "inference_runner.hpp"
#include "utils/logger.hpp"

// =============================================================================
// Thread-safe job queue for asynchronous inference execution
// =============================================================================

class InferenceQueue {
 public:
  explicit InferenceQueue(size_t max_size = 0) : max_size_(max_size) {}

  // Enqueue a new inference job
  auto push(const std::shared_ptr<InferenceJob>& job) -> bool
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (max_size_ > 0 && queue_.size() >= max_size_) {
      log_warning("Inference queue is full. Dropping job.");
      return false;
    }
    queue_.push(job);
    cv_.notify_one();
    return true;
  }

  // Wait until a job is available, then dequeue it
  void wait_and_pop(std::shared_ptr<InferenceJob>& job)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return !queue_.empty(); });
    job = queue_.front();
    queue_.pop();
  }

  // Gracefully shutdown by pushing a special "shutdown" job
  void shutdown() { push(InferenceJob::make_shutdown_job()); }

 private:
  std::queue<std::shared_ptr<InferenceJob>> queue_;
  size_t max_size_;
  std::mutex mutex_;
  std::condition_variable cv_;
};
