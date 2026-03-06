#pragma once

#include <torch/torch.h>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <vector>

#include "starpu_task_worker.hpp"

namespace starpu_server {

class ResultDispatcher {
 public:
  ResultDispatcher(
      const RuntimeConfig* opts, std::atomic<std::size_t>* completed_jobs,
      std::condition_variable* all_done_cv);

  void prepare_job_completion_callback(
      StarPUTaskRunner& runner, const std::shared_ptr<InferenceJob>& job) const;

  static void ensure_callback_timing(detail::TimingInfo& timing);

  void record_job_metrics(
      const std::shared_ptr<InferenceJob>& job,
      StarPUTaskRunner::DurationMs latency, std::size_t batch_size) const;

  void log_job_timings(
      int request_id, StarPUTaskRunner::DurationMs latency,
      const detail::TimingInfo& timing_info) const;

  void finalize_job_completion(const std::shared_ptr<InferenceJob>& job) const;

  static auto resolve_batch_size(
      StarPUTaskRunner& runner,
      const std::shared_ptr<InferenceJob>& job) -> std::size_t;

  static void emit_batch_traces(
      const std::shared_ptr<InferenceJob>& job, StarPUTaskRunner& runner);

  static void invoke_previous_callback(
      const std::function<void(std::vector<torch::Tensor>&&, double)>& previous,
      std::vector<torch::Tensor>& results, double latency_ms);

  static void propagate_completion_to_sub_jobs(
      const std::shared_ptr<InferenceJob>& aggregated_job,
      const std::vector<torch::Tensor>& aggregated_outputs, double latency_ms);

 private:
  void handle_job_completion(
      StarPUTaskRunner& runner, const std::shared_ptr<InferenceJob>& job,
      const std::function<void(std::vector<torch::Tensor>, double)>&
          prev_callback,
      std::vector<torch::Tensor>& results, double latency_ms) const;

  const RuntimeConfig* opts_;
  std::atomic<std::size_t>* completed_jobs_;
  std::condition_variable* all_done_cv_;
};

}  // namespace starpu_server
