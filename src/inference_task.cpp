#include "inference_task.hpp"

// Submit an inference task using StarPU and TorchScript
void submit_inference_task(StarPUSetup& starpu,
                           const torch::Tensor& input_tensor,
                           torch::Tensor& output_tensor,
                           torch::jit::script::Module& module,
                           const ProgramOptions& opts)
{
  int num_buffers = 2;
  // Extract raw pointers and sizes for input and output tensors
  float* input_ptr = input_tensor.data_ptr<float>();
  float* output_ptr = output_tensor.data_ptr<float>();
  int64_t input_size = input_tensor.numel();
  int64_t output_size = output_tensor.numel();

  // Register input and output data with StarPU
  starpu_data_handle_t input_handle, output_handle;
  starpu_vector_data_register(&input_handle, STARPU_MAIN_RAM,
    reinterpret_cast<uintptr_t>(input_ptr), input_size, sizeof(float));
  starpu_vector_data_register(&output_handle, STARPU_MAIN_RAM,
    reinterpret_cast<uintptr_t>(output_ptr), output_size, sizeof(float));
  
  // Set up inference parameters to pass to the codelet
  InferenceParams* args = new InferenceParams();
  args->input_size = input_size;
  args->output_size = output_size;
  auto sizes = input_tensor.sizes();
  args->ndims = sizes.size();
  args->module = module;

  // Store tensor dimensions
  for (int i = 0; i < args->ndims; ++i)
  {
    args->dims[i] = sizes[i];
  }

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
  task->dyn_handles = (starpu_data_handle_t*)malloc(num_buffers * sizeof(*(task->dyn_handles)));
  task->dyn_modes = (starpu_data_access_mode*)malloc(num_buffers * sizeof(*(task->dyn_modes)));
  task->callback_func = &StarPUSetup::task_end_callback;
  
  task->dyn_handles[0] = input_handle;    
  task->dyn_modes[0] = STARPU_R;

  task->dyn_handles[1] = output_handle;
  task->dyn_modes[1] = STARPU_W;

  // Submit the task to StarPU runtime system
  int ret = starpu_task_submit(task);
  if (ret != 0) 
  {
    std::cerr << "Task submission error: " << ret << std::endl;
  }

  // Ensure all tasks are completed before proceeding
  starpu_task_wait_for_all();

  // Unregister StarPU data handles
  starpu_data_unregister_submit(input_handle);
  starpu_data_unregister_submit(output_handle);
}