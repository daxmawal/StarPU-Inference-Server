#include "inference_runner.hpp"
#include "inference_task.hpp"
#include <torch/script.h>
#include <iostream>

// Main loop for running multiple inference iterations using StarPU and TorchScript
void run_inference_loop(const ProgramOptions& opts, StarPUSetup& starpu)
{
  torch::jit::script::Module module = torch::jit::load(opts.model_path);

  if (opts.input_shape.empty()) 
  {
    std::cerr << "Error: you must provide --shape for the input tensor.\n";
    return;
  }

  // Generate random input, run direct inference for reference, and allocate an output tensor for StarPU
  torch::Tensor input_tensor = torch::rand(opts.input_shape);
  at::Tensor output_direct = module.forward({input_tensor}).toTensor();
  torch::Tensor output_tensor = torch::empty_like(output_direct);

  for (int i = 0; i < opts.iterations; ++i)
  {
    submit_inference_task(starpu, input_tensor, output_tensor, module, opts, output_direct, i);
  }
}
