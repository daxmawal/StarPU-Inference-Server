#include "inference_task.hpp"

#include <iostream>

#include "exceptions.hpp"

struct InferenceCallbackContext {
  std::shared_ptr<InferenceJob> job;
  ProgramOptions opts;
  int iteration;
  std::vector<starpu_data_handle_t> input_handles;
  starpu_data_handle_t output_handle;
};

void
cleanup_inference_context(InferenceCallbackContext* ctx)
{
  if (ctx) {
    for (size_t i = 0; i < ctx->input_handles.size(); ++i) {
      starpu_data_unregister_submit(ctx->input_handles[i]);
    }
    starpu_data_unregister_submit(ctx->output_handle);
    delete ctx;
  }
}

// Callback called after data has been acquired
void
handle_output_tensor_after_acquire(void* arg)
{
  auto* ctx = static_cast<InferenceCallbackContext*>(arg);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto latency = std::chrono::duration_cast<std::chrono::microseconds>(
                     end_time - ctx->job->start_time)
                     .count();

  if (ctx->job->on_complete) {
    ctx->job->on_complete(ctx->job->output_tensor, latency);
  }

  cleanup_inference_context(ctx);
}

// Submit an inference task using StarPU and TorchScript
void
submit_inference_task(
    StarPUSetup& starpu, std::shared_ptr<InferenceJob> job,
    torch::jit::script::Module& module, const ProgramOptions& opts)
{
  size_t num_inputs = job->input_tensors.size();
  size_t num_outputs = 1;
  size_t num_buffers = num_outputs + num_inputs;

  if (!job) {
    throw InvalidInferenceJobException("Job is null.");
  }

  // Register input data with StarPU
  std::vector<starpu_data_handle_t> input_handles(num_inputs);

  for (size_t i = 0; i < num_inputs; ++i) {
    auto& tensor = job->input_tensors[i];
    int64_t numel = tensor.numel();
    auto dtype = tensor.dtype();

    void* data_ptr = tensor.data_ptr();

    int64_t element_size = tensor.element_size();

    starpu_vector_data_register(
        &input_handles[i], STARPU_MAIN_RAM,
        reinterpret_cast<uintptr_t>(data_ptr), static_cast<size_t>(numel),
        static_cast<size_t>(element_size));

    if (input_handles[i] == nullptr) {
      throw StarPURegistrationException(
          "Failed to register input handle with StarPU.");
    }
  }

  // Register output data with StarPU
  starpu_data_handle_t output_handle;
  starpu_vector_data_register(
      &output_handle, STARPU_MAIN_RAM,
      reinterpret_cast<uintptr_t>(job->output_tensor.data_ptr<float>()),
      static_cast<size_t>(job->output_tensor.numel()), sizeof(float));
  if (output_handle == nullptr) {
    throw StarPURegistrationException(
        "Failed to register output_handle handle with StarPU.");
  }

  // Set up inference parameters to pass to the codelet
  InferenceParams* args = new InferenceParams();
  args->num_inputs = num_inputs;
  for (size_t i = 0; i < job->input_types.size(); ++i) {
    args->input_types[i] = job->input_types[i];
  }
  args->num_outputs = num_outputs;
  args->output_size = job->output_tensor.numel();
  args->module = module;

  for (size_t i = 0; i < num_inputs; ++i) {
    const auto& tensor = job->input_tensors[i];
    size_t num_dims = tensor.sizes().size();

    args->num_dims[i] = num_dims;

    for (size_t j = 0; j < num_dims; ++j) {
      args->dims[i][j] = tensor.sizes()[j];
    }
  }

  auto* ctx = new InferenceCallbackContext{
      job, opts, job->job_id, input_handles, output_handle};

  // Create and configure the StarPU task
  struct starpu_task* task = starpu_task_create();
  task->nbuffers = static_cast<int>(num_buffers);
  task->cl = starpu.codelet();
  task->synchronous = opts.synchronous ? 1 : 0;
  task->cl_arg = args;
  task->cl_arg_size = sizeof(InferenceParams);
  task->cl_arg_free = 1;
  task->destroy = 1;
  task->dyn_handles =
      (starpu_data_handle_t*)malloc(num_buffers * sizeof(*task->dyn_handles));
  task->dyn_modes =
      (starpu_data_access_mode*)malloc(num_buffers * sizeof(*task->dyn_modes));
  if (!task->dyn_handles || !task->dyn_modes) {
    cleanup_inference_context(ctx);
    if (task->dyn_handles)
      free(task->dyn_handles);
    if (task->dyn_modes)
      free(task->dyn_modes);
    starpu_task_destroy(task);

    throw MemoryAllocationException(
        "Memory allocation failed for dyn_handles or dyn_modes.");
  }

  for (size_t i = 0; i < num_inputs; ++i) {
    task->dyn_handles[i] = input_handles[i];
    task->dyn_modes[i] = STARPU_R;
  }
  task->dyn_handles[num_inputs] = output_handle;
  task->dyn_modes[num_inputs] = STARPU_W;

  // Callback: asynchronous output data retrieval
  task->callback_func = [](void* arg) {
    auto* cb_ctx = static_cast<InferenceCallbackContext*>(arg);
    starpu_data_acquire_cb(
        cb_ctx->output_handle, STARPU_R, handle_output_tensor_after_acquire,
        cb_ctx);
  };
  task->callback_arg = ctx;

  int ret = starpu_task_submit(task);
  if (ret != 0) {
    cleanup_inference_context(ctx);
    throw StarPUTaskSubmissionException(
        "Failed to submit task to StarPU (code " + std::to_string(ret) + ").");
  }
}