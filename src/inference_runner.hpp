#pragma once

#include <torch/script.h>

#include <chrono>
#include <functional>
#include <optional>
#include <vector>

#include "args_parser.hpp"
#include "device_type.hpp"
#include "starpu_setup.hpp"

// =============================================================================
// TimingInfo: precise timestamps for latency profiling
// =============================================================================
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

// =============================================================================
// InferenceResult: output of a completed job, including diagnostics
// =============================================================================
struct InferenceResult {
  unsigned int job_id;
  std::vector<torch::Tensor> inputs;
  torch::Tensor result;
  double latency_ms = 0.0;
  DeviceType executed_on = DeviceType::Unknown;
  int device_id = -1;
  int worker_id = -1;
  TimingInfo timing_info;
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
      unsigned int id,
      std::function<void(torch::Tensor, int64_t)> callback = nullptr);

  // Factory
  static std::shared_ptr<InferenceJob> make_shutdown_job();

  // Job properties
  bool is_shutdown() const;

  // Input and metadata
  std::vector<torch::Tensor> input_tensors;
  std::vector<at::ScalarType> input_types;

  unsigned int job_id = 0;

  // Callback for result handling
  std::function<void(torch::Tensor, double)> on_complete;
  std::chrono::high_resolution_clock::time_point start_time;

  // Output and execution
  torch::Tensor output_tensor;
  DeviceType executed_on = DeviceType::Unknown;
  int device_id = -1;
  int worker_id = -1;
  TimingInfo timing_info;

  // Optional scheduling hint
  std::optional<unsigned int> fixed_worker_id;

 private:
  bool is_shutdown_signal_ = false;
};

// =============================================================================
// Entry point: launches warmup and execution loop
// =============================================================================
void run_inference_loop(const ProgramOptions& opts, StarPUSetup& starpu);
