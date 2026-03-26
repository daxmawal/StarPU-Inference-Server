#pragma once

#include <torch/script.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "device_type.hpp"
#include "runtime_config.hpp"
#include "starpu_setup.hpp"
#include "utils/monotonic_clock.hpp"

namespace starpu_server {
// =============================================================================
// TimingInfo: precise timestamps for latency profiling
// =============================================================================

namespace detail {
struct TimingInfo {
  MonotonicClock::time_point enqueued_time;
  MonotonicClock::time_point last_enqueued_time;
  MonotonicClock::time_point dequeued_time;
  MonotonicClock::time_point batch_collect_start_time;
  MonotonicClock::time_point batch_collect_end_time;
  MonotonicClock::time_point before_starpu_submitted_time;
  MonotonicClock::time_point codelet_start_time;
  MonotonicClock::time_point codelet_end_time;
  MonotonicClock::time_point inference_start_time;
  MonotonicClock::time_point callback_start_time;
  MonotonicClock::time_point callback_end_time;
  int submission_id = -1;
};

struct BaseLatencyBreakdown {
  double queue_ms = 0.0;
  double batch_ms = 0.0;
  double submit_ms = 0.0;
  double scheduling_ms = 0.0;
  double codelet_ms = 0.0;
  double inference_ms = 0.0;
  double callback_ms = 0.0;
  double total_ms = 0.0;
};

// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
void set_cuda_device_count_override(std::optional<int> override_count);
#endif  // SONAR_IGNORE_END
// GCOVR_EXCL_STOP
auto get_cuda_device_count() -> int;
void validate_device_ids(std::span<const int> device_ids, int device_count);
struct GpuReplicaAssignment {
  int device_id = -1;
  int worker_id = -1;

  auto operator==(const GpuReplicaAssignment&) const -> bool = default;
};
auto build_gpu_replica_assignments(const RuntimeConfig& opts)
    -> std::vector<GpuReplicaAssignment>;
auto compute_latency_breakdown(
    const TimingInfo& timing, double total_latency_ms) -> BaseLatencyBreakdown;
}  // namespace detail

// =============================================================================
// InferenceJob: a job submitted to the inference engine
// =============================================================================

class InferenceJob;

class JobBatchState {
 public:
  struct AggregatedSubJob {
    std::weak_ptr<InferenceJob> job;
    std::function<void(const std::vector<torch::Tensor>&, double)> callback;
    int64_t batch_size = 1;
    int request_id = -1;
    MonotonicClock::time_point arrival_time;
  };

  void set_logical_job_count(int count) { logical_job_count_ = count; }
  [[nodiscard]] auto logical_job_count() const -> int
  {
    return logical_job_count_;
  }

  void set_aggregated_sub_jobs(std::vector<AggregatedSubJob> jobs)
  {
    aggregated_sub_jobs_ = std::move(jobs);
  }

  void set_effective_batch_size(int64_t batch)
  {
    effective_batch_size_ = batch;
  }

  void reset_effective_batch_size() { effective_batch_size_.reset(); }

  [[nodiscard]] auto effective_batch_size() const -> std::optional<int64_t>
  {
    return effective_batch_size_;
  }

  void set_pending_sub_jobs(std::vector<std::shared_ptr<InferenceJob>> jobs)
  {
    pending_sub_jobs_ = std::move(jobs);
  }

  void clear_pending_sub_jobs() { pending_sub_jobs_.clear(); }

  [[nodiscard]] auto pending_sub_jobs() const
      -> const std::vector<std::shared_ptr<InferenceJob>>&
  {
    return pending_sub_jobs_;
  }

  [[nodiscard]] auto has_pending_sub_jobs() const -> bool
  {
    return !pending_sub_jobs_.empty();
  }

  [[nodiscard]] auto take_pending_sub_jobs()
      -> std::vector<std::shared_ptr<InferenceJob>>
  {
    return std::exchange(pending_sub_jobs_, {});
  }

  [[nodiscard]] auto aggregated_sub_jobs() const
      -> const std::vector<AggregatedSubJob>&
  {
    return aggregated_sub_jobs_;
  }

  [[nodiscard]] auto has_aggregated_sub_jobs() const -> bool
  {
    return !aggregated_sub_jobs_.empty();
  }

 private:
  int logical_job_count_ = 1;
  std::vector<AggregatedSubJob> aggregated_sub_jobs_;
  std::optional<int64_t> effective_batch_size_;
  std::vector<std::shared_ptr<InferenceJob>> pending_sub_jobs_;
};

using InferenceJobCompletionCallback =
    std::function<void(std::vector<torch::Tensor>, double)>;

struct InferenceJobFailureInfo {
  std::string stage;
  std::string reason;
  std::string message;
  bool metrics_reported = false;
};

class InferenceJobPayloadState : public JobBatchState {
 public:
  InferenceJobPayloadState() noexcept = default;

  InferenceJobPayloadState(
      std::vector<torch::Tensor> inputs, std::vector<at::ScalarType> types,
      int request_identifier)
      : input_tensors_(std::move(inputs)), input_types_(std::move(types)),
        request_id_(request_identifier), start_time_(MonotonicClock::now())
  {
  }

  void set_request_id(int request_id) { request_id_ = request_id; }
  void set_fixed_worker_id(int worker_id) { fixed_worker_id_ = worker_id; }
  void set_input_tensors(const std::vector<torch::Tensor>& inputs)
  {
    reset_effective_batch_size();
    input_tensors_.clear();
    input_tensors_.reserve(inputs.size());
    for (const auto& tensor : inputs) {
      if (tensor.is_contiguous()) {
        input_tensors_.push_back(tensor);
      } else {
        input_tensors_.push_back(tensor.contiguous());
      }
    }
  }
  void set_input_types(const std::vector<at::ScalarType>& types)
  {
    input_types_ = types;
  }
  void set_output_tensors(const std::vector<torch::Tensor>& outputs)
  {
    output_tensors_.clear();
    output_tensors_.reserve(outputs.size());
    for (const auto& output_tensor : outputs) {
      output_tensors_.push_back(output_tensor.contiguous());
    }
  }
  void set_start_time(MonotonicClock::time_point time) { start_time_ = time; }

  void set_input_memory_holders(
      std::vector<std::shared_ptr<const void>> holders)
  {
    input_memory_holders_ = std::move(holders);
  }

  void release_input_memory_holders() { input_memory_holders_.clear(); }

  void set_cancelled_flag(std::shared_ptr<std::atomic<bool>> flag)
  {
    cancelled_flag_ = std::move(flag);
  }

  [[nodiscard]] auto is_cancelled() const -> bool
  {
    return cancelled_flag_ != nullptr &&
           cancelled_flag_->load(std::memory_order_acquire);
  }

  void set_submission_id(int submission_id) { submission_id_ = submission_id; }

  [[nodiscard]] auto submission_id() const -> int { return submission_id_; }

  [[nodiscard]] auto get_input_memory_holders() const
      -> const std::vector<std::shared_ptr<const void>>&
  {
    return input_memory_holders_;
  }

  [[nodiscard]] auto get_request_id() const -> int { return request_id_; }
  [[nodiscard]] auto get_input_tensors() const
      -> const std::vector<torch::Tensor>&
  {
    return input_tensors_;
  }
  [[nodiscard]] auto release_input_tensors() -> std::vector<torch::Tensor>
  {
    return std::exchange(input_tensors_, {});
  }
  [[nodiscard]] auto get_input_types() const
      -> const std::vector<at::ScalarType>&
  {
    return input_types_;
  }
  [[nodiscard]] auto get_output_tensors() const
      -> const std::vector<torch::Tensor>&
  {
    return output_tensors_;
  }
  [[nodiscard]] auto get_start_time() const -> const MonotonicClock::time_point&
  {
    return start_time_;
  }
  [[nodiscard]] auto get_fixed_worker_id() const -> const std::optional<int>&
  {
    return fixed_worker_id_;
  }

 private:
  std::vector<torch::Tensor> input_tensors_;
  std::vector<at::ScalarType> input_types_;
  std::vector<torch::Tensor> output_tensors_;
  std::vector<std::shared_ptr<const void>> input_memory_holders_;
  std::shared_ptr<std::atomic<bool>> cancelled_flag_;

  int request_id_ = 0;
  int submission_id_ = -1;
  std::optional<int> fixed_worker_id_;
  MonotonicClock::time_point start_time_;
};

class InferenceJobCompletionState {
 public:
  InferenceJobCompletionState() noexcept = default;

  explicit InferenceJobCompletionState(InferenceJobCompletionCallback callback)
      : on_complete_(std::move(callback))
  {
  }

  void set_on_complete(InferenceJobCompletionCallback call_back)
  {
    const std::scoped_lock lock(on_complete_mutex_);
    on_complete_ = std::move(call_back);
  }
  void set_model_name(std::string model_name)
  {
    model_name_ = std::move(model_name);
  }

  void set_failure_info(InferenceJobFailureInfo info)
  {
    const std::scoped_lock lock(failure_info_mutex_);
    failure_info_ = std::move(info);
  }
  void clear_failure_info()
  {
    const std::scoped_lock lock(failure_info_mutex_);
    failure_info_.reset();
  }
  [[nodiscard]] auto failure_info() const
      -> std::optional<InferenceJobFailureInfo>
  {
    const std::scoped_lock lock(failure_info_mutex_);
    return failure_info_;
  }
  [[nodiscard]] auto take_failure_info()
      -> std::optional<InferenceJobFailureInfo>
  {
    const std::scoped_lock lock(failure_info_mutex_);
    return std::exchange(failure_info_, std::nullopt);
  }
  [[nodiscard]] auto model_name() const -> std::string_view
  {
    return model_name_;
  }

  [[nodiscard]] auto get_on_complete() const -> InferenceJobCompletionCallback
  {
    const std::scoped_lock lock(on_complete_mutex_);
    return on_complete_;
  }
  [[nodiscard]] auto take_on_complete() -> InferenceJobCompletionCallback
  {
    const std::scoped_lock lock(on_complete_mutex_);
    return std::exchange(on_complete_, InferenceJobCompletionCallback{});
  }
  [[nodiscard]] auto has_on_complete() const -> bool
  {
    const std::scoped_lock lock(on_complete_mutex_);
    return static_cast<bool>(on_complete_);
  }
  [[nodiscard]] auto try_mark_terminal_handled() -> bool
  {
    bool expected = false;
    return terminal_handled_.compare_exchange_strong(
        expected, true, std::memory_order_acq_rel, std::memory_order_acquire);
  }
  [[nodiscard]] auto terminal_handled() const -> bool
  {
    return terminal_handled_.load(std::memory_order_acquire);
  }

 private:
  mutable std::mutex on_complete_mutex_;
  mutable std::mutex failure_info_mutex_;
  InferenceJobCompletionCallback on_complete_;
  std::string model_name_;
  std::optional<InferenceJobFailureInfo> failure_info_;
  std::atomic<bool> terminal_handled_{false};
};

class InferenceJobRuntimeState {
 public:
  void set_device_id(int device_id)
  {
    const std::scoped_lock lock(runtime_device_state_.mutex);
    runtime_device_state_.device_id = device_id;
  }

  void set_worker_id(int worker_id)
  {
    const std::scoped_lock lock(runtime_device_state_.mutex);
    runtime_device_state_.worker_id = worker_id;
  }

  void set_executed_on(DeviceType executed_on)
  {
    const std::scoped_lock lock(runtime_device_state_.mutex);
    runtime_device_state_.executed_on = executed_on;
  }

  [[nodiscard]] auto get_device_id() const -> int
  {
    const std::scoped_lock lock(runtime_device_state_.mutex);
    return runtime_device_state_.device_id;
  }

  [[nodiscard]] auto get_worker_id() const -> int
  {
    const std::scoped_lock lock(runtime_device_state_.mutex);
    return runtime_device_state_.worker_id;
  }

  [[nodiscard]] auto get_executed_on() const -> DeviceType
  {
    const std::scoped_lock lock(runtime_device_state_.mutex);
    return runtime_device_state_.executed_on;
  }

  // Timing writer contract:
  // - gRPC ingress thread writes: enqueued_time, last_enqueued_time
  // - batching thread writes: dequeued_time, batch_collect_* times
  // - runner submission thread writes: before_starpu_submitted_time,
  //   submission_id
  // - StarPU codelet worker writes: codelet_start/end, inference_start,
  //   executed_on/device_id/worker_id
  // - StarPU output callback thread writes: callback_start/end
  //
  // For cross-thread reads/writes on host-side paths, use
  // update_timing_info()/timing_info_snapshot()/set_timing_info().
  // timing_info() remains test-only. Runtime paths should only use
  // update_timing_info()/timing_info_snapshot()/set_timing_info().
  template <typename Updater>
  void update_timing_info(Updater&& updater)
  {
    const std::scoped_lock lock(timing_info_mutex_);
    std::forward<Updater>(updater)(timing_info_);
  }

  [[nodiscard]] auto timing_info_snapshot() const -> detail::TimingInfo
  {
    const std::scoped_lock lock(timing_info_mutex_);
    return timing_info_;
  }

  void set_timing_info(const detail::TimingInfo& timing)
  {
    const std::scoped_lock lock(timing_info_mutex_);
    timing_info_ = timing;
  }

// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  auto timing_info() -> detail::TimingInfo& { return timing_info_; }
#endif  // SONAR_IGNORE_END
        // GCOVR_EXCL_STOP

 private:
  mutable std::mutex timing_info_mutex_;

  struct RuntimeDeviceState {
    mutable std::mutex mutex;
    DeviceType executed_on = DeviceType::Unknown;
    int device_id = -1;
    int worker_id = -1;
  };

  RuntimeDeviceState runtime_device_state_;
  detail::TimingInfo timing_info_;
};

class InferenceJob : public InferenceJobPayloadState,
                     public InferenceJobCompletionState,
                     public InferenceJobRuntimeState {
 public:
  using AggregatedSubJob = JobBatchState::AggregatedSubJob;
  using CompletionCallback = InferenceJobCompletionCallback;
  using FailureInfo = InferenceJobFailureInfo;

  InferenceJob() noexcept = default;

  InferenceJob(
      std::vector<torch::Tensor> inputs, std::vector<at::ScalarType> types,
      int request_identifier, CompletionCallback callback = nullptr);
};

// =============================================================================
// Entry point: model loading and warmup
// =============================================================================

namespace detail {
auto sanitize_cuda_device_count(long long raw_count) -> int;
}  // namespace detail

auto load_model_and_reference_output(const RuntimeConfig& opts)
    -> std::optional<std::tuple<
        torch::jit::script::Module, std::vector<torch::jit::script::Module>,
        std::vector<torch::Tensor>>>;

void run_warmup(
    const RuntimeConfig& opts, StarPUSetup& starpu,
    torch::jit::script::Module& model_cpu,
    std::vector<torch::jit::script::Module>& models_gpu,
    const std::vector<torch::Tensor>& outputs_ref);
}  // namespace starpu_server
