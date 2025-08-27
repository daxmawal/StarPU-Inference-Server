#pragma once

#include <torch/script.h>

#include <chrono>
#include <functional>
#include <optional>
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
  std::chrono::high_resolution_clock::time_point dequeued_time;
  std::chrono::high_resolution_clock::time_point before_starpu_submitted_time;
  std::chrono::high_resolution_clock::time_point codelet_start_time;
  std::chrono::high_resolution_clock::time_point codelet_end_time;
  std::chrono::high_resolution_clock::time_point inference_start_time;
  std::chrono::high_resolution_clock::time_point callback_start_time;
  std::chrono::high_resolution_clock::time_point callback_end_time;
};
}  // namespace detail

// =============================================================================
// InferenceResult: output of a completed job, including diagnostics
// =============================================================================

struct InferenceResult {
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

class InferenceJob {
 public:
  // Constructors
  InferenceJob() = default;

  InferenceJob(
      std::vector<torch::Tensor> inputs, std::vector<at::ScalarType> types,
      int job_identifier,
      std::function<void(std::vector<torch::Tensor>, double)> callback =
          nullptr);

  static auto make_shutdown_job() -> std::shared_ptr<InferenceJob>;

  // Check shutdown
  [[nodiscard]] auto is_shutdown() const -> bool { return is_shutdown_signal_; }

  // Setters
  void set_job_id(int job_id) { job_id_ = job_id; }
  void set_fixed_worker_id(int worker_id) { fixed_worker_id_ = worker_id; }
  void set_input_tensors(const std::vector<torch::Tensor>& inputs)
  {
    input_tensors_ = inputs;
  }
  void set_input_types(const std::vector<at::ScalarType>& types)
  {
    input_types_ = types;
  }
  void set_outputs_tensors(const std::vector<torch::Tensor>& outputs)
  {
    output_tensors_ = outputs;
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

  // Getters
  [[nodiscard]] auto get_job_id() const -> int { return job_id_; }
  [[nodiscard]] auto get_input_tensors() const
      -> const std::vector<torch::Tensor>&
  {
    return input_tensors_;
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
  std::vector<torch::Tensor> input_tensors_;
  std::vector<at::ScalarType> input_types_;
  std::vector<torch::Tensor> output_tensors_;

  int job_id_ = 0;
  std::optional<int> fixed_worker_id_;
  std::function<void(std::vector<torch::Tensor>, double)> on_complete_;
  std::chrono::high_resolution_clock::time_point start_time_;

  DeviceType executed_on_ = DeviceType::Unknown;
  int device_id_ = -1;
  int worker_id_ = -1;

  detail::TimingInfo timing_info_;

  bool is_shutdown_signal_ = false;
};

// =============================================================================
// Entry point: launches warmup and execution loop
// =============================================================================

class StarPUTaskRunner;
using WorkerThreadLauncher = std::jthread (*)(StarPUTaskRunner&);
extern WorkerThreadLauncher worker_thread_launcher;

auto load_model_and_reference_output(const RuntimeConfig& opts)
    -> std::tuple<
        torch::jit::script::Module, std::vector<torch::jit::script::Module>,
        std::vector<torch::Tensor>>;

void run_warmup(
    const RuntimeConfig& opts, StarPUSetup& starpu,
    torch::jit::script::Module& model_cpu,
    std::vector<torch::jit::script::Module>& models_gpu,
    const std::vector<torch::Tensor>& outputs_ref);

void run_inference_loop(const RuntimeConfig& opts, StarPUSetup& starpu);
}  // namespace starpu_server
