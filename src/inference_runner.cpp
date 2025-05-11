#include "inference_runner.hpp"
#include "inference_task.hpp"
#include "inference_validator.hpp"

#include <torch/script.h>
#include <iostream>

void run_inference_loop(const ProgramOptions& opts, StarPUSetup& starpu)
{
  torch::jit::script::Module module_direct = torch::jit::load(opts.model_path);

  for (int i = 0; i < opts.iterations; ++i)
  {
    if (opts.input_shape.empty()) 
    {
      std::cerr << "Error: you must provide --shape for the input tensor.\n";
      return;
    }

    torch::Tensor input_tensor = torch::rand(opts.input_shape);
    at::Tensor output_direct = module_direct.forward({input_tensor}).toTensor();
    torch::Tensor output_tensor = torch::empty_like(output_direct);

    submit_inference_task(starpu, input_tensor, output_tensor, opts.model_path);

    std::cout << "Output (first 10 values): "
              << output_tensor.flatten().slice(0, 0, 10) << std::endl;

    validate_outputs(output_direct, output_tensor);

    std::cout << "End of iteration " << i << std::endl;
  }
}
