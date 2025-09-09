#pragma once

#include <memory>
#include <functional>
#include <optional>
#include <vector>

#include "inference_runner.hpp"
#include "runtime_config.hpp"

namespace starpu_server {
// =============================================================================
// InferenceCallbackContext: passed to StarPU callbacks
// =============================================================================

class OutputSlotPool;  // forward declaration

struct InferenceCallbackContext {
  std::shared_ptr<InferenceJob> job;
  std::shared_ptr<InferenceParams> inference_params;
  std::shared_ptr<void> self_keep_alive;
  const RuntimeConfig* opts = nullptr;
  starpu_data_handle_t* dyn_handles = nullptr;
  starpu_data_access_mode* dyn_modes = nullptr;
  std::vector<starpu_data_handle_t> inputs_handles;
  std::vector<starpu_data_handle_t> outputs_handles;
  int id = 0;
  std::atomic<int> remaining_outputs_to_acquire{0};
  std::mutex mutex;
  // When using pooled input handles, skip unregister on cleanup and call
  // on_finished to release the slot back to the pool.
  bool keep_input_handles = false;
  // When using pooled output handles, skip unregister on cleanup and call
  // on_finished to release the slot back to the pool. Also carry pool context
  // to copy results back to job-owned output tensors.
  bool keep_output_handles = false;
  OutputSlotPool* output_pool = nullptr;  // fwd-declared pointer
  int output_slot_id = -1;
  std::function<void()> on_finished;

  InferenceCallbackContext(
      std::shared_ptr<InferenceJob> job_,
      std::shared_ptr<InferenceParams> params_, const RuntimeConfig* opts_,
      int id_, std::vector<starpu_data_handle_t> inputs_,
      std::vector<starpu_data_handle_t> outputs_) noexcept;
};

// =============================================================================
// InferenceTask
// Responsible for submitting a single inference job to StarPU,
// including input/output registration, StarPU task creation, and handling
// asynchronous completion callbacks.
// =============================================================================

class InferenceTask {
 public:
  InferenceTask(
      StarPUSetup* starpu, std::shared_ptr<InferenceJob> job,
      torch::jit::script::Module* model_cpu,
      std::vector<torch::jit::script::Module>* models_gpu,
      const RuntimeConfig* opts) noexcept;

  static auto safe_register_tensor_vector(
      const torch::Tensor& tensor,
      const std::string& label) -> starpu_data_handle_t;

  static auto register_inputs_handles(
      const std::vector<torch::Tensor>& input_tensors)
      -> std::vector<starpu_data_handle_t>;

  static auto register_outputs_handles(
      const std::vector<torch::Tensor>& outputs_tensors)
      -> std::vector<starpu_data_handle_t>;

  [[nodiscard]] auto prepare_input_handles() const
      -> std::vector<starpu_data_handle_t>;

  [[nodiscard]] auto prepare_output_handles() const
      -> std::vector<starpu_data_handle_t>;

  auto create_inference_params() -> std::shared_ptr<InferenceParams>;

  auto create_context(
      const std::vector<starpu_data_handle_t>& inputs,
      const std::vector<starpu_data_handle_t>& outputs)
      -> std::shared_ptr<InferenceCallbackContext>;

  void fill_model_pointers(
      const std::shared_ptr<InferenceParams>& params) const;
  void bind_runtime_job_info(
      const std::shared_ptr<InferenceParams>& params) const;
  void fill_input_layout(
      const std::shared_ptr<InferenceParams>& params, size_t num_inputs) const;
  void check_limits(size_t num_inputs) const;

  auto create_task(
      const std::vector<starpu_data_handle_t>& inputs_handles,
      const std::vector<starpu_data_handle_t>& outputs_handles,
      const std::shared_ptr<InferenceCallbackContext>& ctx) -> starpu_task*;

  void submit();
  void assign_fixed_worker_if_needed(starpu_task* task) const;

  static void allocate_task_buffers(
      starpu_task* task, size_t num_buffers,
      const std::shared_ptr<InferenceCallbackContext>& ctx);

  static void fill_task_buffers(
      starpu_task* task, const std::vector<starpu_data_handle_t>& inputs,
      const std::vector<starpu_data_handle_t>& outputs);

  static void starpu_output_callback(void* arg);
  static void acquire_output_handle(
      starpu_data_handle_t handle, InferenceCallbackContext* ctx);
  static void process_output_handle(
      starpu_data_handle_t handle, InferenceCallbackContext* ctx);

  static void cleanup(const std::shared_ptr<InferenceCallbackContext>& ctx);
  static void release_output_data(
      const std::vector<starpu_data_handle_t>& handles);
  static void finalize_context(
      const std::shared_ptr<InferenceCallbackContext>& ctx_sptr);

  static void record_and_run_completion_callback(
      InferenceCallbackContext* ctx,
      std::chrono::high_resolution_clock::time_point end_time);

  static void finalize_inference_task(void* arg);

  static void log_exception(
      const std::string& context, const std::exception& e);

 private:
  StarPUSetup* starpu_;
  std::shared_ptr<InferenceJob> job_;
  torch::jit::script::Module* model_cpu_;
  std::vector<torch::jit::script::Module>* models_gpu_;
  const RuntimeConfig* opts_;
};
}  // namespace starpu_server
