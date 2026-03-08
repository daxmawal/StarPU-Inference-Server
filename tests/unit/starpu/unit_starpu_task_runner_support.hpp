#pragma once

#include <cuda_runtime_api.h>
#include <prometheus/client_metric.h>
#include <prometheus/metric_family.h>
#include <starpu.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "core/inference_task.hpp"
#include "exceptions.hpp"
#include "monitoring/metrics.hpp"
#include "starpu_task_worker/task_runner_internal.hpp"
#include "support/starpu_data_interface_override.hpp"
#include "support/starpu_task_submit_override.hpp"
#include "test_starpu_task_runner.hpp"
#include "utils/batching_trace_logger.hpp"
#include "utils/perf_observer.hpp"

using starpu_server::CaptureStream;
using starpu_server::ErrorLevel;
using starpu_server::expected_log_line;
namespace test_api = starpu_server::task_runner_internal::testing;

struct SlotHandleLeaseAcquireContext {
  std::vector<starpu_data_handle_t>* calls = nullptr;
  std::optional<starpu_data_handle_t> fail_handle;
  int fail_code = 0;
};

struct SlotHandleLeaseReleaseContext {
  std::vector<starpu_data_handle_t>* calls = nullptr;
};

class ScopedCudaStreamSyncFailure {
 public:
  explicit ScopedCudaStreamSyncFailure(cudaStream_t target);

  ScopedCudaStreamSyncFailure(const ScopedCudaStreamSyncFailure&) = delete;
  auto operator=(const ScopedCudaStreamSyncFailure&)
      -> ScopedCudaStreamSyncFailure& = delete;

  ~ScopedCudaStreamSyncFailure();
};

auto cuda_stream_sync_failure_count() -> int;

class ScopedCudaStreamCreateFailure {
 public:
  ScopedCudaStreamCreateFailure();

  ScopedCudaStreamCreateFailure(const ScopedCudaStreamCreateFailure&) = delete;
  auto operator=(const ScopedCudaStreamCreateFailure&)
      -> ScopedCudaStreamCreateFailure& = delete;

  ~ScopedCudaStreamCreateFailure();
};

auto cuda_stream_create_failure_count() -> int;

class ScopedCudaMemcpyAsyncFailure {
 public:
  explicit ScopedCudaMemcpyAsyncFailure(cudaStream_t target);

  ScopedCudaMemcpyAsyncFailure(const ScopedCudaMemcpyAsyncFailure&) = delete;
  auto operator=(const ScopedCudaMemcpyAsyncFailure&)
      -> ScopedCudaMemcpyAsyncFailure& = delete;

  ~ScopedCudaMemcpyAsyncFailure();
};

auto cuda_memcpy_failure_count() -> int;

struct ScopedSlotHandleLeaseAcquireContext {
  SlotHandleLeaseAcquireContext* previous = nullptr;
  SlotHandleLeaseAcquireContext* current = nullptr;
  starpu_test::ScopedStarpuDataAcquireOverride guard;

  explicit ScopedSlotHandleLeaseAcquireContext(
      SlotHandleLeaseAcquireContext& ctx);

  ScopedSlotHandleLeaseAcquireContext(
      const ScopedSlotHandleLeaseAcquireContext&) = delete;
  auto operator=(const ScopedSlotHandleLeaseAcquireContext&)
      -> ScopedSlotHandleLeaseAcquireContext& = delete;

  ~ScopedSlotHandleLeaseAcquireContext();
};

struct ScopedSlotHandleLeaseReleaseContext {
  SlotHandleLeaseReleaseContext* previous = nullptr;
  SlotHandleLeaseReleaseContext* current = nullptr;
  starpu_test::ScopedStarpuDataReleaseOverride guard;

  explicit ScopedSlotHandleLeaseReleaseContext(
      SlotHandleLeaseReleaseContext& ctx);

  ScopedSlotHandleLeaseReleaseContext(
      const ScopedSlotHandleLeaseReleaseContext&) = delete;
  auto operator=(const ScopedSlotHandleLeaseReleaseContext&)
      -> ScopedSlotHandleLeaseReleaseContext& = delete;

  ~ScopedSlotHandleLeaseReleaseContext();
};

class TraceLoggerSession {
 public:
  TraceLoggerSession();

  TraceLoggerSession(const TraceLoggerSession&) = delete;
  auto operator=(const TraceLoggerSession&) -> TraceLoggerSession& = delete;

  ~TraceLoggerSession();

  void close();

  [[nodiscard]] auto path() const -> const std::filesystem::path&;

 private:
  std::filesystem::path path_;
  bool closed_{false};
};

auto read_trace_file(const std::filesystem::path& path) -> std::string;

void populate_trace_timing(starpu_server::InferenceJob& job);

auto make_aggregated_sub_job(
    const std::shared_ptr<starpu_server::InferenceJob>& job,
    int request_id) -> starpu_server::InferenceJob::AggregatedSubJob;

struct VectorInterfaceSnapshot {
  starpu_vector_interface* iface = nullptr;
  std::size_t elemsize = 0;
  std::size_t allocsize = 0;
  std::size_t nx = 0;
};

auto snapshot_vector_interfaces(starpu_data_handle_t handle)
    -> std::vector<VectorInterfaceSnapshot>;

void restore_vector_interfaces(
    const std::vector<VectorInterfaceSnapshot>& snapshots);

extern std::atomic<int> submit_override_calls;

extern std::atomic<int> missing_interface_override_hits;
extern starpu_data_handle_t missing_interface_override_handle;

auto missing_interface_override(starpu_data_handle_t handle, unsigned node)
    -> void*;

auto two_memory_nodes_override() -> unsigned;

auto AlwaysFailStarpuSubmit(starpu_task*) -> int;

auto NoOpStarpuDataAcquire(starpu_data_handle_t, starpu_data_access_mode)
    -> int;

void NoOpStarpuDataRelease(starpu_data_handle_t);

namespace starpu_server {
class StarPUTaskRunnerTestAdapter {
 public:
  static void handle_submission_failure(
      InputSlotPool* input_pool, int input_slot, OutputSlotPool* output_pool,
      int output_slot, const std::shared_ptr<InferenceCallbackContext>& ctx,
      int submit_code)
  {
    StarPUTaskRunner::PoolResources pools{};
    pools.input_pool = input_pool;
    pools.input_slot = input_slot;
    pools.output_pool = output_pool;
    pools.output_slot = output_slot;
    StarPUTaskRunner::handle_submission_failure(pools, ctx, submit_code);
  }

  static void propagate_completion_to_sub_jobs(
      const std::shared_ptr<InferenceJob>& aggregated_job,
      const std::vector<torch::Tensor>& aggregated_outputs, double latency_ms)
  {
    StarPUTaskRunner::propagate_completion_to_sub_jobs(
        aggregated_job, aggregated_outputs, latency_ms);
  }

  static auto maybe_build_batched_job(
      StarPUTaskRunner* runner,
      std::vector<std::shared_ptr<InferenceJob>>& jobs)
      -> std::shared_ptr<InferenceJob>
  {
    return runner->maybe_build_batched_job(jobs);
  }

  static auto can_merge_jobs(
      const std::shared_ptr<InferenceJob>& lhs,
      const std::shared_ptr<InferenceJob>& rhs) -> bool
  {
    return StarPUTaskRunner::can_merge_jobs(lhs, rhs);
  }

  static auto collect_batch(
      StarPUTaskRunner* runner, const std::shared_ptr<InferenceJob>& first_job)
      -> std::vector<std::shared_ptr<InferenceJob>>
  {
    return runner->collect_batch(first_job);
  }

  static auto merge_input_tensors(
      const std::vector<std::shared_ptr<InferenceJob>>& jobs,
      int64_t total_samples) -> std::vector<torch::Tensor>
  {
    return StarPUTaskRunner::merge_input_tensors(jobs, total_samples);
  }

  static auto merge_input_memory_holders(
      const std::vector<std::shared_ptr<InferenceJob>>& jobs)
      -> std::vector<std::shared_ptr<const void>>
  {
    return StarPUTaskRunner::merge_input_memory_holders(jobs);
  }

  static void set_submit_hook(std::function<void()> hook)
  {
    task_runner_internal::set_submit_inference_task_hook(std::move(hook));
  }

  static void reset_submit_hook()
  {
    task_runner_internal::reset_submit_inference_task_hook();
  }

  static auto validate_batch_and_copy_inputs(
      StarPUTaskRunner* runner, const std::shared_ptr<InferenceJob>& job,
      InputSlotPool* input_pool, int input_slot) -> int64_t
  {
    StarPUTaskRunner::PoolResources pools{};
    pools.input_pool = input_pool;
    pools.input_slot = input_slot;
    return runner->validate_batch_and_copy_inputs(job, pools);
  }

  static auto validate_batch_and_copy_inputs_with_pools(
      StarPUTaskRunner* runner, const std::shared_ptr<InferenceJob>& job,
      int64_t batch, const StarPUTaskRunner::PoolResources& pools) -> int64_t
  {
    if (runner == nullptr || runner->slot_manager_ == nullptr) {
      return -1;
    }
    return test_api::slot_manager_validate_batch_and_copy_inputs(
        runner->slot_manager_.get(), job, batch, pools.input_pool,
        pools.input_slot, pools.output_pool, pools.output_slot);
  }

  static auto validate_batch_and_copy_inputs_custom(
      StarPUTaskRunner* runner, const std::shared_ptr<InferenceJob>& job,
      int64_t batch, InputSlotPool* input_pool, int input_slot) -> int64_t
  {
    StarPUTaskRunner::PoolResources pools{};
    pools.input_pool = input_pool;
    pools.input_slot = input_slot;
    return validate_batch_and_copy_inputs_with_pools(runner, job, batch, pools);
  }

  static auto configure_task_context(
      InferenceTask& task, InputSlotPool* input_pool, int input_slot,
      OutputSlotPool* output_pool, int output_slot,
      std::vector<starpu_data_handle_t> input_handles,
      std::vector<starpu_data_handle_t> output_handles,
      int64_t batch_size) -> std::shared_ptr<InferenceCallbackContext>
  {
    StarPUTaskRunner::PoolResources pools{};
    pools.input_pool = input_pool;
    pools.input_slot = input_slot;
    pools.output_pool = output_pool;
    pools.output_slot = output_slot;
    return StarPUTaskRunner::configure_task_context(
        task, pools, std::move(input_handles), std::move(output_handles),
        batch_size);
  }

  static void trace_batch_if_enabled(
      StarPUTaskRunner* runner, const std::shared_ptr<InferenceJob>& job,
      bool warmup_job, int submission_id)
  {
    runner->trace_batch_if_enabled(job, warmup_job, submission_id);
  }

  static auto job_sample_size(
      StarPUTaskRunner* runner,
      const std::shared_ptr<InferenceJob>& job) -> int64_t
  {
    if (runner == nullptr || runner->batch_collector_ == nullptr) {
      return -1;
    }
    return test_api::batch_collector_job_sample_size(
        runner->batch_collector_.get(), job);
  }

  static void enqueue_prepared_job(
      StarPUTaskRunner* runner, const std::shared_ptr<InferenceJob>& job)
  {
    runner->enqueue_prepared_job(job);
  }

  static auto resolve_batch_size(
      StarPUTaskRunner* runner,
      const std::shared_ptr<InferenceJob>& job) -> int64_t
  {
    return runner->resolve_batch_size(job);
  }

  static void finalize_job_after_exception(
      StarPUTaskRunner* runner, const std::shared_ptr<InferenceJob>& job,
      const std::exception& exception, std::string_view log_prefix, int job_id)
  {
    runner->finalize_job_after_exception(job, exception, log_prefix, job_id);
  }

  static void finalize_job_after_unknown_exception(
      StarPUTaskRunner* runner, const std::shared_ptr<InferenceJob>& job,
      std::string_view log_prefix, int job_id)
  {
    runner->finalize_job_after_unknown_exception(job, log_prefix, job_id);
  }

  static auto wait_for_prepared_job(StarPUTaskRunner* runner)
      -> std::shared_ptr<InferenceJob>
  {
    return runner->wait_for_prepared_job();
  }

  static void run_batching_loop(StarPUTaskRunner* runner)
  {
    if (runner != nullptr) {
      runner->batching_loop();
    }
  }

  static auto batching_done(StarPUTaskRunner* runner) -> bool
  {
    if (runner == nullptr) {
      return false;
    }
    auto& state = runner->prepared_state_;
    const std::scoped_lock lock(state.mutex);
    return state.batching_done;
  }

  static void disable_prepared_job_sync(StarPUTaskRunner* runner)
  {
    if (runner == nullptr || runner->batch_collector_ == nullptr) {
      return;
    }
    test_api::batch_collector_disable_prepared_job_sync(
        runner->batch_collector_.get());
  }

  static auto is_batch_collector_batching_done(StarPUTaskRunner* runner) -> bool
  {
    if (runner == nullptr || runner->batch_collector_ == nullptr) {
      return false;
    }
    return test_api::batch_collector_is_batching_done(
        runner->batch_collector_.get());
  }

  static auto should_abort_batch_collector_inflight_wait(
      StarPUTaskRunner* runner) -> bool
  {
    if (runner == nullptr || runner->batch_collector_ == nullptr) {
      return false;
    }
    return test_api::batch_collector_should_abort_inflight_wait(
        runner->batch_collector_.get());
  }

  static void set_batch_collector_batching_done_ptr(
      StarPUTaskRunner* runner, bool* batching_done)
  {
    if (runner == nullptr || runner->batch_collector_ == nullptr) {
      return;
    }
    test_api::batch_collector_set_batching_done_ptr(
        runner->batch_collector_.get(), batching_done);
  }

  static void set_batch_collector_batching_done_value(
      StarPUTaskRunner* runner, bool batching_done)
  {
    if (runner == nullptr || runner->batch_collector_ == nullptr) {
      return;
    }
    test_api::batch_collector_set_batching_done_value(
        runner->batch_collector_.get(), batching_done);
  }

  static void set_batch_collector_queue_to_null(StarPUTaskRunner* runner)
  {
    if (runner == nullptr || runner->batch_collector_ == nullptr) {
      return;
    }
    test_api::batch_collector_set_queue(
        runner->batch_collector_.get(), nullptr);
  }

  static auto get_batch_collector_queue(StarPUTaskRunner* runner)
      -> starpu_server::InferenceQueue*
  {
    if (runner == nullptr || runner->batch_collector_ == nullptr) {
      return nullptr;
    }
    return test_api::batch_collector_get_queue(runner->batch_collector_.get());
  }

  static void set_batch_collector_pending_job(
      StarPUTaskRunner* runner,
      const std::shared_ptr<starpu_server::InferenceJob>& job)
  {
    if (runner == nullptr || runner->batch_collector_ == nullptr) {
      return;
    }
    test_api::batch_collector_set_pending_job(
        runner->batch_collector_.get(), job);
  }

  static auto try_acquire_next_job(
      StarPUTaskRunner* runner, bool enable_wait,
      task_runner_internal::Clock::time_point coalesce_deadline)
      -> std::shared_ptr<InferenceJob>
  {
    if (runner == nullptr || runner->batch_collector_ == nullptr) {
      return nullptr;
    }
    return test_api::batch_collector_try_acquire_next_job(
        runner->batch_collector_.get(), enable_wait, coalesce_deadline);
  }

  static void release_pending_jobs(
      const std::shared_ptr<InferenceJob>& job,
      std::vector<std::shared_ptr<InferenceJob>>& pending_jobs)
  {
    StarPUTaskRunner::release_pending_jobs(job, pending_jobs);
  }

  static void ensure_callback_timing(detail::TimingInfo& timing)
  {
    task_runner_helpers::ensure_callback_timing(timing);
  }

  static void record_job_metrics(
      StarPUTaskRunner* runner, const std::shared_ptr<InferenceJob>& job,
      StarPUTaskRunner::DurationMs latency, std::size_t batch_size)
  {
    if (runner != nullptr) {
      task_runner_helpers::record_job_metrics(
          *runner, job, latency, batch_size);
    }
  }

  static void finalize_job_completion(
      StarPUTaskRunner* runner, const std::shared_ptr<InferenceJob>& job)
  {
    if (runner != nullptr) {
      task_runner_helpers::finalize_job_completion(*runner, job);
    }
  }

  static void handle_cancelled_job(
      StarPUTaskRunner* runner, const std::shared_ptr<InferenceJob>& job)
  {
    if (runner != nullptr) {
      runner->handle_cancelled_job(job);
    }
  }

  static auto should_hold_job(
      const std::shared_ptr<InferenceJob>& candidate,
      const std::shared_ptr<InferenceJob>& reference,
      const std::optional<int>& target_worker) -> bool
  {
    return test_api::batch_collector_should_hold_job(
        candidate, reference, target_worker);
  }

  static auto exceeds_sample_limit(
      StarPUTaskRunner* runner, int64_t accumulated_samples,
      const std::shared_ptr<InferenceJob>& job, int64_t max_samples_cap) -> bool
  {
    if (runner == nullptr || runner->batch_collector_ == nullptr) {
      return false;
    }
    return test_api::batch_collector_exceeds_sample_limit(
        runner->batch_collector_.get(), accumulated_samples, job,
        max_samples_cap);
  }

  static void reserve_inflight_slot(StarPUTaskRunner* runner)
  {
    runner->reserve_inflight_slot();
  }

  static void release_inflight_slot(StarPUTaskRunner* runner)
  {
    runner->release_inflight_slot();
  }

  static auto has_inflight_limit(const StarPUTaskRunner* runner) -> bool
  {
    return runner->has_inflight_limit();
  }

  static auto get_inflight_tasks(const StarPUTaskRunner* runner) -> std::size_t
  {
    if (runner == nullptr || runner->inflight_state_ == nullptr) {
      return 0;
    }
    return runner->inflight_state_->tasks.load(std::memory_order_acquire);
  }

  static auto get_max_inflight_tasks(const StarPUTaskRunner* runner)
      -> std::size_t
  {
    if (runner == nullptr || runner->inflight_state_ == nullptr) {
      return 0;
    }
    return runner->inflight_state_->max_tasks;
  }

  static void set_inflight_state_to_null(StarPUTaskRunner* runner)
  {
    if (runner != nullptr) {
      runner->inflight_state_.reset();
    }
  }

  static void set_result_dispatcher(
      StarPUTaskRunner* runner, std::shared_ptr<ResultDispatcher> dispatcher)
  {
    if (runner == nullptr) {
      return;
    }
    runner->result_dispatcher_ = std::move(dispatcher);
  }
};
}  // namespace starpu_server
