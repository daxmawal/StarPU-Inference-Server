#include "inference_runner.hpp"
#include <torch/script.h>
#include <iostream>
#include "inference_task.hpp"
#include <vector>

// Main loop for running multiple inference iterations using StarPU and TorchScript
void run_inference_loop(const ProgramOptions& opts, StarPUSetup& starpu) {
  torch::jit::script::Module module = torch::jit::load(opts.model_path);

  if (opts.input_shape.empty()) {
    std::cerr << "Error: you must provide --shape for the input tensor.\n";
    return;
  }

  // Run reference direct inference with one example
  torch::Tensor input_ref = torch::rand(opts.input_shape);
  at::Tensor output_direct = module.forward({input_ref}).toTensor();

  // Keep input/output tensors alive for all async tasks
  std::vector<torch::Tensor> input_tensors;
  std::vector<torch::Tensor> output_tensors;

  for (int i = 0; i < opts.iterations; ++i) {
    torch::Tensor input_tensor = input_ref;
    torch::Tensor output_tensor = torch::empty_like(output_direct);

    input_tensors.push_back(input_tensor);
    output_tensors.push_back(output_tensor);

    auto start_time = std::chrono::high_resolution_clock::now(); 
    submit_inference_task(starpu, input_tensor, output_tensor, module, opts, output_direct, i, start_time);
  }
}