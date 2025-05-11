#pragma once

#include <torch/torch.h>

bool validate_outputs(const at::Tensor& output_direct, const at::Tensor& output_starpu, double tolerance = 1e-5);