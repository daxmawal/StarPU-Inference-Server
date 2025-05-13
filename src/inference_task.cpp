#include "inference_task.hpp"

#include <iostream>

struct InferenceCallbackContext {
  std::shared_ptr<InferenceJob> job;
  torch::Tensor output_direct;
  ProgramOptions opts;
  int iteration;
  starpu_data_handle_t input_handle;
  starpu_data_handle_t output_handle;
};

void
cleanup_inference_context(InferenceCallbackContext* ctx)
{
  if (ctx) {
    starpu_data_unregister_submit(ctx->input_handle);
    starpu_data_unregister_submit(ctx->output_handle);
    delete ctx;
  }
}

// Callback called after data has been acquired
void
output_tensor_ready_callback(void* arg)
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
    torch::jit::script::Module& module, const ProgramOptions& opts,
    const torch::Tensor& output_direct)
{
  int num_buffers = 2;

  // Register input and output data with StarPU
  starpu_data_handle_t input_handle, output_handle;
  starpu_vector_data_register(
      &input_handle, STARPU_MAIN_RAM,
      reinterpret_cast<uintptr_t>(job->input_tensor.data_ptr<float>()),
      job->input_tensor.numel(), sizeof(float));
  starpu_vector_data_register(
      &output_handle, STARPU_MAIN_RAM,
      reinterpret_cast<uintptr_t>(job->output_tensor.data_ptr<float>()),
      job->output_tensor.numel(), sizeof(float));

  // Set up inference parameters to pass to the codelet
  InferenceParams* args = new InferenceParams();
  args->input_size = job->input_tensor.numel();
  args->output_size = job->output_tensor.numel();
  args->ndims = job->input_tensor.sizes().size();
  args->module = module;

  // Store tensor dimensions
  for (int i = 0; i < args->ndims; ++i) {
    args->dims[i] = job->input_tensor.sizes()[i];
  }

  auto* ctx = new InferenceCallbackContext{
      job,          output_direct, opts,      job->job_id,
      input_handle, output_handle};

  // Create and configure the StarPU task
  struct starpu_task* task = starpu_task_create();
  task->handles[0] = input_handle;
  task->handles[1] = output_handle;
  task->nbuffers = num_buffers;
  task->cl = starpu.codelet();
  task->synchronous = opts.synchronous ? 1 : 0;
  task->cl_arg = args;
  task->cl_arg_size = sizeof(InferenceParams);
  task->cl_arg_free = 1;
  task->dyn_handles =
      (starpu_data_handle_t*)malloc(num_buffers * sizeof(*task->dyn_handles));
  task->dyn_modes =
      (starpu_data_access_mode*)malloc(num_buffers * sizeof(*task->dyn_modes));

  task->dyn_handles[0] = input_handle;
  task->dyn_modes[0] = STARPU_R;
  task->dyn_handles[1] = output_handle;
  task->dyn_modes[1] = STARPU_W;

  // Callback: asynchronous output data retrieval
  task->callback_func = [](void* arg) {
    auto* ctx = static_cast<InferenceCallbackContext*>(arg);
    starpu_data_acquire_cb(
        ctx->output_handle, STARPU_R, output_tensor_ready_callback, ctx);
  };
  task->callback_arg = ctx;

  int ret = starpu_task_submit(task);
  if (ret != 0) {
    std::cerr << "Task submission error: " << ret << std::endl;
    cleanup_inference_context(ctx);
  }
}