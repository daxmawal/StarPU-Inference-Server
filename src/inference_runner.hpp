#pragma once
#include "args_parser.hpp"
#include "device_type.hpp"
#include "starpu_setup.hpp"

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

struct InferenceResult {
  unsigned int job_id;
  std::vector<torch::Tensor> inputs;
  torch::Tensor result;
  double latency_ms;
  DeviceType executed_on = DeviceType::Unknown;
  TimingInfo timing_info;
  int device_id;
};

class InferenceJob {
 public:
  InferenceJob() = default;

  InferenceJob(
      std::vector<torch::Tensor> inputs, std::vector<at::ScalarType> types,
      unsigned int id,
      std::function<void(torch::Tensor, int64_t)> callback = nullptr);

  static std::shared_ptr<InferenceJob> make_shutdown_job();

  bool is_shutdown() const;

  std::vector<torch::Tensor> input_tensors;
  std::vector<at::ScalarType> input_types;
  unsigned int job_id = 0;
  std::function<void(torch::Tensor, double)> on_complete;
  std::chrono::high_resolution_clock::time_point start_time;
  torch::Tensor output_tensor;
  DeviceType executed_on = DeviceType::Unknown;
  int device_id = -1;
  TimingInfo timing_info;
  std::optional<unsigned int> fixed_worker_id;

 private:
  bool is_shutdown_signal_ = false;
};

void run_inference_loop(const ProgramOptions& opts, StarPUSetup& starpu);