#pragma once

#include <torch/script.h>

#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "device_type.hpp"
#include "runtime_config.hpp"
#include "starpu_setup.hpp"

namespace starpu_server {
// =============================================================================
// TimingInfo: precise timestamps for latency profiling
// =============================================================================

namespace detail {
struct TimingInfo {
  std::chrono::high_resolution_clock::time_point enqueued_time;
  std::chrono::high_resolution_clock::time_point last_enqueued_time;
  std::chrono::high_resolution_clock::time_point dequeued_time;
  std::chrono::high_resolution_clock::time_point batch_collect_start_time;
  std::chrono::high_resolution_clock::time_point batch_collect_end_time;
  std::chrono::high_resolution_clock::time_point before_starpu_submitted_time;
  std::chrono::high_resolution_clock::time_point codelet_start_time;
  std::chrono::high_resolution_clock::time_point codelet_end_time;
  std::chrono::high_resolution_clock::time_point inference_start_time;
  std::chrono::high_resolution_clock::time_point callback_start_time;
  std::chrono::high_resolution_clock::time_point callback_end_time;
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

void set_cuda_device_count_override(std::optional<int> override_count);
auto get_cuda_device_count() -> int;
void validate_device_ids(std::span<const int> device_ids, int device_count);
auto compute_latency_breakdown(
    const TimingInfo& timing, double total_latency_ms) -> BaseLatencyBreakdown;
}  // namespace detail

// =============================================================================
// InferenceResult: output of a completed job, including diagnostics
// =============================================================================

struct InferenceResult {
  InferenceResult() noexcept = default;
  std::vector<torch::Tensor> inputs;
  std::vector<torch::Tensor> results;
  double latency_ms = 0.0;
  detail::TimingInfo timing_info;
  int request_id = -1;
  int submission_id = -1;
  int device_id = -1;
  int worker_id = -1;
  DeviceType executed_on = DeviceType::Unknown;
};

// =============================================================================
// InferenceJob: a job submitted to the inference engine
// =============================================================================

class InferenceQueue;
class InferenceJob;

class JobBatchState {
 public:
  struct AggregatedSubJob {
    std::weak_ptr<InferenceJob> job;
    std::function<void(const std::vector<torch::Tensor>&, double)> callback;
    int64_t batch_size = 1;
    int request_id = -1;
    std::chrono::high_resolution_clock::time_point arrival_time;
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

class InferenceJobIO {
 public:
  virtual ~InferenceJobIO() = default;
  virtual void set_input_tensors(const std::vector<torch::Tensor>& inputs)
  {
    input_tensors_.clear();
    input_tensors_.reserve(inputs.size());
    for (const auto& tensor : inputs) {
      input_tensors_.push_back(tensor.contiguous());
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

  void set_input_memory_holders(
      std::vector<std::shared_ptr<const void>> holders)
  {
    input_memory_holders_ = std::move(holders);
  }

  [[nodiscard]] auto get_input_memory_holders() const
      -> const std::vector<std::shared_ptr<const void>>&
  {
    return input_memory_holders_;
  }

 protected:
  void adopt_input_tensors(std::vector<torch::Tensor> inputs)
  {
    input_tensors_ = std::move(inputs);
  }

  void adopt_input_types(std::vector<at::ScalarType> types)
  {
    input_types_ = std::move(types);
  }

 private:
  std::vector<torch::Tensor> input_tensors_;
  std::vector<at::ScalarType> input_types_;
  std::vector<torch::Tensor> output_tensors_;
  std::vector<std::shared_ptr<const void>> input_memory_holders_;
};

class InferenceJob : public JobBatchState, public InferenceJobIO {
 public:
  using AggregatedSubJob = JobBatchState::AggregatedSubJob;

  InferenceJob() noexcept = default;

  InferenceJob(
      std::vector<torch::Tensor> inputs, std::vector<at::ScalarType> types,
      int request_identifier,
      std::function<void(const std::vector<torch::Tensor>&, double)> callback =
          nullptr);

  static auto make_shutdown_job() -> std::shared_ptr<InferenceJob>;

  [[nodiscard]] auto is_shutdown() const -> bool { return is_shutdown_signal_; }

  void set_request_id(int request_id) { request_id_ = request_id; }
  void set_fixed_worker_id(int worker_id) { fixed_worker_id_ = worker_id; }
  void set_gpu_only(bool enable) { gpu_only_ = enable; }
  void set_is_warmup_job(bool is_warmup) { is_warmup_job_ = is_warmup; }
  void set_input_tensors(const std::vector<torch::Tensor>& inputs) override
  {
    reset_effective_batch_size();
    InferenceJobIO::set_input_tensors(inputs);
  }
  void set_start_time(std::chrono::high_resolution_clock::time_point time)
  {
    start_time_ = time;
  }
  void set_on_complete(
      std::function<void(std::vector<torch::Tensor>, double)> call_back)
  {
    on_complete_ = std::move(call_back);
  }
  void set_model_name(std::string model_name)
  {
    model_name_ = std::move(model_name);
  }
  [[nodiscard]] auto model_name() const -> std::string_view
  {
    return model_name_;
  }

  void set_submission_id(int submission_id) { submission_id_ = submission_id; }

  [[nodiscard]] auto submission_id() const -> int { return submission_id_; }

  [[nodiscard]] auto is_gpu_only() const -> bool { return gpu_only_; }
  [[nodiscard]] auto is_warmup_job() const -> bool { return is_warmup_job_; }
  [[nodiscard]] auto is_probe_job() const -> bool { return is_probe_job_; }
  void set_is_probe_job(bool is_probe) { is_probe_job_ = is_probe; }

  [[nodiscard]] auto get_request_id() const -> int { return request_id_; }
  [[nodiscard]] auto get_start_time() const
      -> const std::chrono::high_resolution_clock::time_point&
  {
    return start_time_;
  }
  [[nodiscard]] auto get_fixed_worker_id() const -> const std::optional<int>&
  {
    return fixed_worker_id_;
  }
  [[nodiscard]] auto get_on_complete() const
      -> const std::function<void(std::vector<torch::Tensor>, double)>&
  {
    return on_complete_;
  }
  [[nodiscard]] auto has_on_complete() const -> bool
  {
    return static_cast<bool>(on_complete_);
  }

  auto get_device_id() -> int& { return device_id_; }
  auto get_worker_id() -> int& { return worker_id_; }
  auto get_executed_on() -> DeviceType& { return executed_on_; }

  auto timing_info() -> detail::TimingInfo& { return timing_info_; }

 private:
  int request_id_ = 0;
  int submission_id_ = -1;
  std::optional<int> fixed_worker_id_;
  std::function<void(std::vector<torch::Tensor>, double)> on_complete_;
  std::chrono::high_resolution_clock::time_point start_time_;
  std::string model_name_;

  DeviceType executed_on_ = DeviceType::Unknown;
  int device_id_ = -1;
  int worker_id_ = -1;

  bool gpu_only_ = false;
  bool is_warmup_job_ = false;
  bool is_probe_job_ = false;
  detail::TimingInfo timing_info_;

  bool is_shutdown_signal_ = false;
};

// =============================================================================
// Entry point: launches warmup and execution loop
// =============================================================================

class StarPUTaskRunner;
using WorkerThreadLauncher = std::function<std::jthread(StarPUTaskRunner&)>;
auto get_worker_thread_launcher() -> WorkerThreadLauncher;
void set_worker_thread_launcher(WorkerThreadLauncher launcher);

namespace detail {
void client_worker(
    InferenceQueue& queue, const RuntimeConfig& opts,
    const std::vector<torch::Tensor>& outputs_ref, int request_nb);

auto build_gpu_model_lookup(
    std::vector<torch::jit::script::Module>& models_gpu,
    const std::vector<int>& device_ids)
    -> std::unordered_map<int, torch::jit::script::Module*>;

auto resolve_validation_model(
    const InferenceResult& result, torch::jit::script::Module& cpu_model,
    const std::unordered_map<int, torch::jit::script::Module*>& gpu_lookup,
    bool validate_results) -> std::optional<torch::jit::script::Module*>;

void process_results(
    const std::vector<InferenceResult>& results,
    torch::jit::script::Module& model_cpu,
    std::vector<torch::jit::script::Module>& models_gpu,
    const RuntimeConfig& opts);
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

void run_inference_loop(const RuntimeConfig& opts, StarPUSetup& starpu);
}  // namespace starpu_server
