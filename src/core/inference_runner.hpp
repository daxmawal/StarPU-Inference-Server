#pragma once

#include <torch/script.h>

#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "device_type.hpp"
#include "runtime_config.hpp"
#include "starpu_setup.hpp"
#include "utils/logger.hpp"

namespace starpu_server {
// =============================================================================
// TimingInfo: precise timestamps for latency profiling
// =============================================================================

namespace detail {
struct TimingInfo {
  std::chrono::high_resolution_clock::time_point enqueued_time;
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
  int job_id;
  int device_id = -1;
  int worker_id = -1;
  DeviceType executed_on = DeviceType::Unknown;
};

// =============================================================================
// InferenceJob: a job submitted to the inference engine
// =============================================================================

class InferenceQueue;

class InferenceJob {
 public:
  struct AggregatedSubJob {
    std::weak_ptr<InferenceJob> job;
    std::function<void(const std::vector<torch::Tensor>&, double)> callback;
    int64_t batch_size = 1;
  };

  InferenceJob() = default;

  InferenceJob(
      std::vector<torch::Tensor> inputs, std::vector<at::ScalarType> types,
      int job_identifier,
      std::function<void(const std::vector<torch::Tensor>&, double)> callback =
          nullptr);

  static auto make_shutdown_job() -> std::shared_ptr<InferenceJob>;

  [[nodiscard]] auto is_shutdown() const -> bool { return is_shutdown_signal_; }

  void set_job_id(int job_id) { job_id_ = job_id; }
  void set_fixed_worker_id(int worker_id) { fixed_worker_id_ = worker_id; }
  void set_input_tensors(const std::vector<torch::Tensor>& inputs)
  {
    input_tensors_.clear();
    input_tensors_.reserve(inputs.size());
    for (const auto& t : inputs) {
      if (t.is_contiguous()) {
        input_tensors_.push_back(t);
      } else {
        input_tensors_.push_back(t.contiguous());
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
    for (const auto& t : outputs) {
      output_tensors_.push_back(t.contiguous());
    }
  }
  void set_start_time(std::chrono::high_resolution_clock::time_point time)
  {
    start_time_ = time;
  }
  void set_on_complete(
      std::function<void(const std::vector<torch::Tensor>&, double)> call_back)
  {
    on_complete_ = std::move(call_back);
  }

  void set_input_memory_holders(
      std::vector<std::shared_ptr<const void>> holders)
  {
    input_memory_holders_ = std::move(holders);
  }

  void set_submission_id(int submission_id) { submission_id_ = submission_id; }

  [[nodiscard]] auto submission_id() const -> int { return submission_id_; }

  [[nodiscard]] auto get_input_memory_holders() const
      -> const std::vector<std::shared_ptr<const void>>&
  {
    return input_memory_holders_;
  }

  [[nodiscard]] auto get_job_id() const -> int { return job_id_; }
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
      -> const std::function<void(const std::vector<torch::Tensor>&, double)>&
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

  void set_logical_job_count(int count) { logical_job_count_ = count; }
  [[nodiscard]] auto logical_job_count() const -> int
  {
    return logical_job_count_;
  }

  void set_aggregated_sub_jobs(std::vector<AggregatedSubJob> jobs)
  {
    aggregated_sub_jobs_ = std::move(jobs);
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
  std::vector<torch::Tensor> input_tensors_;
  std::vector<at::ScalarType> input_types_;
  std::vector<torch::Tensor> output_tensors_;
  std::vector<std::shared_ptr<const void>> input_memory_holders_;

  int job_id_ = 0;
  int submission_id_ = -1;
  std::optional<int> fixed_worker_id_;
  std::function<void(const std::vector<torch::Tensor>&, double)> on_complete_;
  std::chrono::high_resolution_clock::time_point start_time_;

  DeviceType executed_on_ = DeviceType::Unknown;
  int device_id_ = -1;
  int worker_id_ = -1;

  detail::TimingInfo timing_info_;

  bool is_shutdown_signal_ = false;
  int logical_job_count_ = 1;
  std::vector<AggregatedSubJob> aggregated_sub_jobs_;
};

// =============================================================================
// Entry point: launches warmup and execution loop
// =============================================================================

class StarPUTaskRunner;
using WorkerThreadLauncher = std::jthread (*)(StarPUTaskRunner&);
auto get_worker_thread_launcher() -> WorkerThreadLauncher;
void set_worker_thread_launcher(WorkerThreadLauncher launcher);

namespace detail {
void client_worker(
    InferenceQueue& queue, const RuntimeConfig& opts,
    const std::vector<torch::Tensor>& outputs_ref, int iterations);

auto build_gpu_model_lookup(
    std::vector<torch::jit::script::Module>& models_gpu,
    const std::vector<int>& device_ids)
    -> std::vector<torch::jit::script::Module*>;

auto resolve_validation_model(
    const InferenceResult& result, torch::jit::script::Module& cpu_model,
    const std::vector<torch::jit::script::Module*>& gpu_lookup,
    bool validate_results) -> std::optional<torch::jit::script::Module*>;

void process_results(
    const std::vector<InferenceResult>& results,
    torch::jit::script::Module& model_cpu,
    std::vector<torch::jit::script::Module>& models_gpu,
    const std::vector<int>& device_ids, bool validate_results,
    VerbosityLevel verbosity, double rtol, double atol);
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
