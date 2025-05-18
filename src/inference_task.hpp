#pragma once
#include <torch/torch.h>

#include <chrono>
#include <string>

#include "args_parser.hpp"
#include "exceptions.hpp"
#include "inference_runner.hpp"
#include "starpu_setup.hpp"

struct InferenceCallbackContext {
  std::shared_ptr<InferenceJob> job;
  std::shared_ptr<InferenceParams> inference_params;
  ProgramOptions opts;
  int id;
  std::vector<starpu_data_handle_t> input_handles;
  starpu_data_handle_t output_handle;
};

class InferenceTask {
 public:
  InferenceTask(
      StarPUSetup& starpu, std::shared_ptr<InferenceJob> job,
      torch::jit::script::Module& module, const ProgramOptions& opts);

  static void cleanup(InferenceCallbackContext* ctx);
  static void on_output_ready_and_cleanup(void* arg);
  static void log_exception(const std::string& context);
  static void starpu_output_callback(void* arg);
  std::vector<starpu_data_handle_t> register_input_handles(
      const std::vector<torch::Tensor>& input_tensors);
  starpu_data_handle_t register_output_handle(
      const torch::Tensor& output_tensor);
  starpu_task* create_task(
      std::vector<starpu_data_handle_t> input_handles,
      starpu_data_handle_t output_handle, InferenceCallbackContext* ctx);
  std::shared_ptr<InferenceParams> create_inference_params();
  starpu_data_handle_t safe_register_tensor_vector(
      const torch::Tensor& tensor, const std::string& label);
  void submit();

 private:
  StarPUSetup& starpu_;
  std::shared_ptr<InferenceJob> job_;
  torch::jit::script::Module& module_;
  ProgramOptions opts_;
};