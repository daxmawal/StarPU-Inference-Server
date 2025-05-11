#pragma once
#include "starpu_setup.hpp"
#include <torch/torch.h>
#include <string>

void submit_inference_task(StarPUSetup& starpu,
                           const torch::Tensor& input_tensor,
                           torch::Tensor& output_tensor,
                           const std::string& model_path);