#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "args_parser.hpp"
#include "inference_runner.hpp"

// =============================================================================
// InferenceCallbackContext: passed to StarPU callbacks
// =============================================================================
struct InferenceCallbackContext {
  std::shared_ptr<InferenceJob> job;  // Job associated with the callback
  std::shared_ptr<InferenceParams>
      inference_params;  // Parameters used in the StarPU codelet
  ProgramOptions opts;   // Program settings
  unsigned int id = 0;   // Task ID (for logging/debugging)
  std::vector<starpu_data_handle_t>
      input_handles;  // Registered input data handles
  starpu_data_handle_t output_handle = nullptr;  // Output data handle
};

// =============================================================================
// InferenceTask: prepares and submits a task to StarPU
// =============================================================================
class InferenceTask {
 public:
  // Constructor
  InferenceTask(
      StarPUSetup& starpu, std::shared_ptr<InferenceJob> job,
      torch::jit::script::Module& model_cpu,
      std::vector<torch::jit::script::Module>& models_gpu,
      const ProgramOptions& opts);

  // ---- Static utility methods for task lifecycle ----

  /// Cleans up after task completion
  static void cleanup(InferenceCallbackContext* ctx);

  /// Called after output is ready, triggers cleanup
  static void on_output_ready_and_cleanup(void* arg);

  /// Logs any caught exception during task execution
  static void log_exception(const std::string& context);

  /// StarPU-specific callback for when the output is ready
  static void starpu_output_callback(void* arg);

  // ---- Static data registration methods ----

  /// Registers a tensor with StarPU as a vector handle (safe wrapper)
  static starpu_data_handle_t safe_register_tensor_vector(
      const torch::Tensor& tensor, const std::string& label);

  /// Registers input tensors with StarPU and returns their handles
  static std::vector<starpu_data_handle_t> register_input_handles(
      const std::vector<torch::Tensor>& input_tensors);

  /// Registers output tensor with StarPU
  static starpu_data_handle_t register_output_handle(
      const torch::Tensor& output_tensor);

  // ---- Instance methods for preparing and submitting the task ----

  /// Creates a StarPU task with the given input/output and context
  starpu_task* create_task(
      const std::vector<starpu_data_handle_t>& input_handles,
      const starpu_data_handle_t& output_handle, InferenceCallbackContext* ctx);

  /// Prepares the inference parameters (model, layout, etc.)
  std::shared_ptr<InferenceParams> create_inference_params();

  /// Submits the task to StarPU
  void submit();

 private:
  // Internal state
  StarPUSetup& starpu_;
  std::shared_ptr<InferenceJob> job_;
  torch::jit::script::Module& model_cpu_;
  std::vector<torch::jit::script::Module>& models_gpu_;
  ProgramOptions opts_;
};
