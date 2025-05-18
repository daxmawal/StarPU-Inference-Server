#pragma once
#include "args_parser.hpp"
#include "starpu_setup.hpp"

struct InferenceResult {
  int job_id;
  std::vector<torch::Tensor> inputs;
  torch::Tensor result;
  int64_t latency_us;
};

class InferenceJob {
 public:
  InferenceJob() = default;

  InferenceJob(
      std::vector<torch::Tensor> inputs, std::vector<at::ScalarType> types,
      int id, std::function<void(torch::Tensor, int64_t)> callback = nullptr);

  static std::shared_ptr<InferenceJob> make_shutdown_job();

  bool is_shutdown() const;

  std::vector<torch::Tensor> input_tensors;
  std::vector<at::ScalarType> input_types;
  int job_id;
  std::function<void(torch::Tensor, int64_t)> on_complete;
  std::chrono::high_resolution_clock::time_point start_time;
  torch::Tensor output_tensor;

 private:
  bool is_shutdown_signal_ = false;
};

void run_inference_loop(const ProgramOptions& opts, StarPUSetup& starpu);

bool validate_inference_result(
    const InferenceResult& r, torch::jit::script::Module& module);