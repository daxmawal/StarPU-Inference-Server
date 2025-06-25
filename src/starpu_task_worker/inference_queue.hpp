#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>

#include "inference_runner.hpp"

namespace starpu_server {
// =============================================================================
// Thread-safe job queue for asynchronous inference execution
// =============================================================================

class InferenceQueue {
 public:
  // Enqueue a new inference job
  void push(const std::shared_ptr<InferenceJob>& job)
  {
    const std::scoped_lock lock(mutex_);
    queue_.push(job);
    cv_.notify_one();
  }

  // Wait until a job is available, then dequeue it
  void wait_and_pop(std::shared_ptr<InferenceJob>& job)
  {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this] { return !queue_.empty(); });
    job = queue_.front();
    queue_.pop();
  }

  // Gracefully shutdown by pushing a special "shutdown" job
  void shutdown() { push(InferenceJob::make_shutdown_job()); }

 private:
  std::queue<std::shared_ptr<InferenceJob>> queue_;
  std::mutex mutex_;
  std::condition_variable cv_;
};
}  // namespace starpu_server