#include "inference_task.hpp"
#include <iostream>
#include "inference_validator.hpp"

struct InferenceCallbackContext {
  torch::Tensor                                  output_direct;
  torch::Tensor                                  output_tensor;
  ProgramOptions                                 opts;
  int                                            iteration;
  starpu_data_handle_t                           input_handle;
  starpu_data_handle_t                           output_handle;
  std::chrono::high_resolution_clock::time_point start_time;
};

void cleanup_inference_context(InferenceCallbackContext* ctx) {
  if (ctx) {
    starpu_data_unregister_submit(ctx->input_handle);
    starpu_data_unregister_submit(ctx->output_handle);
    delete ctx;
  }
}

// Callback called after data has been acquired
void output_tensor_ready_callback(void* arg) {
  auto* ctx = static_cast<InferenceCallbackContext*>(arg);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto latency =
      std::chrono::duration_cast<std::chrono::microseconds>(end_time - ctx->start_time).count();

  std::cout << "Latency (user-visible) for iteration " << ctx->iteration << ": " << latency << " µs"
            << std::endl;

  std::cout << "Output (first 10 values): " << ctx->output_tensor.flatten().slice(0, 0, 10)
            << std::endl;

  validate_outputs(ctx->output_direct, ctx->output_tensor);

  std::cout << "End of iteration " << ctx->iteration << std::endl;

  cleanup_inference_context(ctx);
}

// Submit an inference task using StarPU and TorchScript
void submit_inference_task(StarPUSetup&                                   starpu,
                           const torch::Tensor&                           input_tensor,
                           torch::Tensor&                                 output_tensor,
                           torch::jit::script::Module&                    module,
                           const ProgramOptions&                          opts,
                           const torch::Tensor&                           output_direct,
                           int                                            iteration,
                           std::chrono::high_resolution_clock::time_point start_time) {
  int num_buffers = 2;
  // Extract raw pointers and sizes for input and output tensors
  float*  input_ptr   = input_tensor.data_ptr<float>();
  float*  output_ptr  = output_tensor.data_ptr<float>();
  int64_t input_size  = input_tensor.numel();
  int64_t output_size = output_tensor.numel();

  // Register input and output data with StarPU
  starpu_data_handle_t input_handle, output_handle;
  starpu_vector_data_register(&input_handle,
                              STARPU_MAIN_RAM,
                              reinterpret_cast<uintptr_t>(input_ptr),
                              input_size,
                              sizeof(float));
  starpu_vector_data_register(&output_handle,
                              STARPU_MAIN_RAM,
                              reinterpret_cast<uintptr_t>(output_ptr),
                              output_size,
                              sizeof(float));

  // Set up inference parameters to pass to the codelet
  InferenceParams* args = new InferenceParams();
  args->input_size      = input_size;
  args->output_size     = output_size;
  args->ndims           = input_tensor.sizes().size();
  args->module          = module;

  // Store tensor dimensions
  for (int i = 0; i < args->ndims; ++i) {
    args->dims[i] = input_tensor.sizes()[i];
  }

  auto* ctx = new InferenceCallbackContext{output_direct.clone(),
                                           output_tensor,
                                           opts,
                                           iteration,
                                           input_handle,
                                           output_handle,
                                           start_time};

  // Create and configure the StarPU task
  struct starpu_task* task = starpu_task_create();
  task->handles[0]         = input_handle;
  task->handles[1]         = output_handle;
  task->nbuffers           = num_buffers;
  task->cl                 = starpu.codelet();
  task->synchronous        = opts.synchronous ? 1 : 0;
  task->cl_arg             = args;
  task->cl_arg_size        = sizeof(InferenceParams);
  task->cl_arg_free        = 1;
  task->dyn_handles = (starpu_data_handle_t*) malloc(num_buffers * sizeof(*task->dyn_handles));
  task->dyn_modes   = (starpu_data_access_mode*) malloc(num_buffers * sizeof(*task->dyn_modes));

  task->dyn_handles[0] = input_handle;
  task->dyn_modes[0]   = STARPU_R;
  task->dyn_handles[1] = output_handle;
  task->dyn_modes[1]   = STARPU_W;

  // Callback: asynchronous output data retrieval
  task->callback_func = [](void* arg) {
    auto* ctx = static_cast<InferenceCallbackContext*>(arg);
    starpu_data_acquire_cb(ctx->output_handle, STARPU_R, output_tensor_ready_callback, ctx);
  };
  task->callback_arg = ctx;

  int ret = starpu_task_submit(task);
  if (ret != 0) {
    std::cerr << "Task submission error: " << ret << std::endl;
    cleanup_inference_context(ctx);
  }
}