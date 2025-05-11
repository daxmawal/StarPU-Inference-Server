#include "inference_task.hpp"

void submit_inference_task(StarPUSetup& starpu,
                           const torch::Tensor& input_tensor,
                           torch::Tensor& output_tensor,
                           torch::jit::script::Module& module,
                           const ProgramOptions& opts)
{
  float* input_ptr = input_tensor.data_ptr<float>();
  float* output_ptr = output_tensor.data_ptr<float>();
  int64_t input_size = input_tensor.numel();
  int64_t output_size = output_tensor.numel();

  starpu_data_handle_t input_handle, output_handle;
  starpu_variable_data_register(&input_handle, STARPU_MAIN_RAM,
    reinterpret_cast<uintptr_t>(input_ptr), input_size * sizeof(float));
  starpu_variable_data_register(&output_handle, STARPU_MAIN_RAM,
    reinterpret_cast<uintptr_t>(output_ptr), output_size * sizeof(float));

  InferenceParams* args = new InferenceParams();
  std::strncpy(args->model_path, opts.model_path.c_str(), sizeof(args->model_path));
  args->input_size = input_size;
  args->output_size = output_size;
  auto sizes = input_tensor.sizes();
  args->ndims = sizes.size();
  args->module = module;
  
  for (int i = 0; i < args->ndims; ++i)
  {
    args->dims[i] = sizes[i];
  }

  struct starpu_task* task = starpu_task_create();
  task->handles[0] = input_handle;
  task->handles[1] = output_handle;
  task->nbuffers = 2;
  task->cl = starpu.codelet();
  task->synchronous = opts.synchronous ? 1 : 0;
  task->cl_arg = args;
  task->cl_arg_size = sizeof(InferenceParams);
  task->cl_arg_free = 1;

  int ret = starpu_task_submit(task);
  if (ret != 0) 
  {
    std::cerr << "Task submission error: " << ret << std::endl;
  }

  starpu_task_wait_for_all();

  starpu_data_unregister(input_handle);
  starpu_data_unregister(output_handle);
}