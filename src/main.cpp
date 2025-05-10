#include "starpu_setup.hpp"
#include "args_parser.hpp"
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>

int main(int argc, char* argv[])
{
  ProgramOptions opts = parse_arguments(argc, argv);

  if (opts.show_help)
  {
    display_help("Inference Engine");
    return 0;
  }

  if (!opts.valid)
    return 1;

  std::cout << "Scheduler  : " << opts.scheduler << "\n";
  std::cout << "Iterations : " << opts.iterations << "\n";

  try
  {
    StarPUSetup starpu(opts.scheduler.c_str());

    // Load the model once for direct inference
    torch::jit::script::Module module_direct = torch::jit::load(opts.model_path);

    for (int i = 0; i < opts.iterations; ++i)
    {
      if (opts.input_shape.empty()) {
        std::cerr << "Error: you must provide --shape for the input tensor.\n";
        return 1;
      }

      torch::Tensor input_tensor = torch::rand(opts.input_shape);
      float* input_ptr = input_tensor.data_ptr<float>();
      int64_t input_size = input_tensor.numel();

      // Perform direct inference to get the actual output shape
      at::Tensor output_direct = module_direct.forward({input_tensor}).toTensor();

      // Reallocate output_tensor with the correct shape
      torch::Tensor output_tensor = torch::empty_like(output_direct);
      float* output_ptr = output_tensor.data_ptr<float>();
      int64_t output_size = output_tensor.numel();

      // StarPU buffer registration
      starpu_data_handle_t input_handle, output_handle;
      starpu_variable_data_register(&input_handle, STARPU_MAIN_RAM,
        reinterpret_cast<uintptr_t>(input_ptr), input_size * sizeof(float));
      starpu_variable_data_register(&output_handle, STARPU_MAIN_RAM,
        reinterpret_cast<uintptr_t>(output_ptr), output_size * sizeof(float));

      // Prepare arguments for the codelet
      InferenceParams* args = new InferenceParams();
      std::strncpy(args->model_path, opts.model_path.c_str(), sizeof(args->model_path));
      args->input_size = input_size;
      args->output_size = output_size;
      auto sizes = input_tensor.sizes();
      args->ndims = sizes.size();
      if (args->ndims > 8) {
        std::cerr << "Error: the tensor has more than 8 dimensions, which is not supported" << std::endl;
        return 1;
      }
      for (int i = 0; i < args->ndims; ++i) {
        args->dims[i] = sizes[i];
      }

      // Task creation and configuration
      struct starpu_task* task = starpu_task_create();
      task->handles[0] = input_handle;
      task->handles[1] = output_handle;
      task->nbuffers = 2;
      task->cl = starpu.codelet();
      task->synchronous = 1;
      task->cl_arg = args;
      task->cl_arg_size = sizeof(InferenceParams);
      task->cl_arg_free = 1;

      int ret = starpu_task_submit(task);
      if (ret != 0)
      {
        std::cerr << "Task submission error: " << ret << std::endl;
      }

      starpu_task_wait_for_all();

      // Read the results
      std::cout << "Output (first 10 values): "
                << output_tensor.flatten().slice(0, 0, 10) << std::endl;
      
      // Compute the absolute error
      at::Tensor diff = torch::abs(output_direct - output_tensor);
      double max_diff = diff.max().item<double>();
      std::cout << "Max difference with direct inference: " << max_diff << std::endl;

      const double tolerance = 1e-5;
      if (max_diff < tolerance) {
        std::cout << "StarPU output matches direct inference (within tolerance).\n";
      } else {
        std::cerr << "Mismatch between StarPU and direct inference! Max diff = " << max_diff << "\n";
      }

      // Cleanup
      starpu_data_unregister(input_handle);
      starpu_data_unregister(output_handle);
      std::cout << "End of iteration " << i << std::endl;
    }
  }
  catch (const std::exception& e)
  {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return 0;
}