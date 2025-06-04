#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "inference_runner.hpp"
#include "runtime_config.hpp"

// =============================================================================
// InferenceCallbackContext: passed to StarPU callbacks
// =============================================================================
struct InferenceCallbackContext {
  std::shared_ptr<InferenceJob> job;  // Job associated with the callback
  std::shared_ptr<InferenceParams>
      inference_params;       // Parameters used in the StarPU codelet
  const RuntimeConfig* opts;  // Program settings
  unsigned int id = 0;        // Task ID (for logging/debugging)
  std::vector<starpu_data_handle_t>
      inputs_handles;  // Registered input data handles
  std::vector<starpu_data_handle_t> outputs_handles;  // Output data handles
  std::atomic<int> remaining_outputs_to_acquire = 0;
  std::mutex mutex;

  starpu_data_handle_t* dyn_handles = nullptr;
  starpu_data_access_mode* dyn_modes = nullptr;

  std::shared_ptr<void> self_keep_alive;

  InferenceCallbackContext(
      std::shared_ptr<InferenceJob> job_,
      std::shared_ptr<InferenceParams> params_, const RuntimeConfig* opts_,
      unsigned int id_, std::vector<starpu_data_handle_t> inputs_,
      std::vector<starpu_data_handle_t> outputs_);
};

// =============================================================================
// InferenceTask: prepares and submits a task to StarPU
// =============================================================================
class InferenceTask {
 public:
  // Constructor
  InferenceTask(
      StarPUSetup* starpu, std::shared_ptr<InferenceJob> job,
      torch::jit::script::Module* model_cpu,
      std::vector<torch::jit::script::Module>* models_gpu,
      const RuntimeConfig* opts);

  // ---- Static utility methods for task lifecycle ----

  /// Cleans up after task completion
  static void cleanup(const std::shared_ptr<InferenceCallbackContext>& ctx);

  /// Called after output is ready, triggers cleanup
  static void on_output_ready_and_cleanup(void* arg);

  /// Logs any caught exception during task execution
  static void log_exception(const std::string& context);

  /// StarPU-specific callback for when the output is ready
  static void starpu_output_callback(void* arg);

  // ---- Static data registration methods ----

  /// Registers a tensor with StarPU as a vector handle (safe wrapper)
  static auto safe_register_tensor_vector(
      const torch::Tensor& tensor,
      const std::string& label) -> starpu_data_handle_t;

  /// Registers input tensors with StarPU and returns their handles
  static auto register_inputs_handles(
      const std::vector<torch::Tensor>& input_tensors)
      -> std::vector<starpu_data_handle_t>;

  /// Registers output tensor with StarPU
  static auto register_outputs_handles(
      const std::vector<torch::Tensor>& outputs_tensors)
      -> std::vector<starpu_data_handle_t>;

  // ---- Instance methods for preparing and submitting the task ----

  /// Creates a StarPU task with the given input/output and context
  auto create_task(
      const std::vector<starpu_data_handle_t>& inputs_handles,
      const std::vector<starpu_data_handle_t>& outputs_handles,
      const std::shared_ptr<InferenceCallbackContext>& ctx) -> starpu_task*;

  /// Prepares the inference parameters (model, layout, etc.)
  auto create_inference_params() -> std::shared_ptr<InferenceParams>;

  /// Submits the task to StarPU
  void submit();

 private:
  // Internal state
  StarPUSetup* starpu_;
  std::shared_ptr<InferenceJob> job_;
  torch::jit::script::Module* model_cpu_;
  std::vector<torch::jit::script::Module>* models_gpu_;
  const RuntimeConfig* opts_;
};
