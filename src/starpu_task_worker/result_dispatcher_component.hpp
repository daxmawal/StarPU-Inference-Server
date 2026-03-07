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

  static void prepare_job_completion_callback(
      StarPUTaskRunner& runner, const std::shared_ptr<InferenceJob>& job);

  static void ensure_callback_timing(detail::TimingInfo& timing);

  void record_job_metrics(
      const std::shared_ptr<InferenceJob>& job,
      StarPUTaskRunner::DurationMs latency, std::size_t batch_size) const;

  void log_job_timings(
      int request_id, StarPUTaskRunner::DurationMs latency,
      const detail::TimingInfo& timing_info) const;

  void finalize_job_completion(const std::shared_ptr<InferenceJob>& job) const;

  static auto resolve_batch_size(
      const RuntimeConfig* opts,
      const std::shared_ptr<InferenceJob>& job) -> std::size_t;

  static void emit_batch_traces(
      const RuntimeConfig* opts, const std::shared_ptr<InferenceJob>& job);

  // Invoke the callback captured before dispatcher wrapping.
  // Terminal callback contract: success/error/cancel must converge to one
  // callback execution per job.
  static void invoke_previous_callback(
      const InferenceJob::CompletionCallback& previous,
      std::vector<torch::Tensor>& results, double latency_ms);

  static void clear_pending_sub_job_callbacks(
      const std::shared_ptr<InferenceJob>& job);

  static void clear_batching_state(const std::shared_ptr<InferenceJob>& job);

  static void cleanup_terminal_job_payload(
      const std::shared_ptr<InferenceJob>& job);

  static void propagate_completion_to_sub_jobs(
      const std::shared_ptr<InferenceJob>& aggregated_job,
      const std::vector<torch::Tensor>& aggregated_outputs, double latency_ms);

 private:
  void handle_job_completion(
      const std::shared_ptr<InferenceJob>& job,
      const InferenceJob::CompletionCallback& prev_callback,
      std::vector<torch::Tensor>& results, double latency_ms) const;

  static void release_inflight_slot(
      const std::shared_ptr<StarPUTaskRunner::InflightState>& inflight_state);

  const RuntimeConfig* opts_;
  std::atomic<std::size_t>* completed_jobs_;
  std::condition_variable* all_done_cv_;
};

}  // namespace starpu_server
