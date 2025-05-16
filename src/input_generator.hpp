#pragma once

#include <torch/torch.h>

#include <stdexcept>
#include <vector>

inline std::vector<torch::Tensor>
generate_random_inputs(
    const std::vector<std::vector<int64_t>>& shapes,
    const std::vector<at::ScalarType>& types)
{
  std::vector<torch::Tensor> inputs;

  for (size_t i = 0; i < shapes.size(); ++i) {
    const auto& shape = shapes[i];
    at::ScalarType type = at::kFloat;
    if (i < types.size()) {
      type = types[i];
    }

    torch::Tensor tensor;
    torch::TensorOptions options = torch::TensorOptions().dtype(type);

    if (type == at::kFloat || type == at::kDouble) {
      tensor = torch::rand(shape, options);
    } else if (
        type == at::kInt || type == at::kLong || type == at::kShort ||
        type == at::kChar) {
      int64_t low = 0;
      int64_t high = 10;
      if (i == 0 && shape.size() == 2 && shape[1] >= 64) {
        high = 30522;
      }
      tensor = torch::randint(low, high, shape, options);
    } else if (type == at::kBool) {
      tensor = torch::randint(0, 2, shape, options);
    } else {
      throw std::runtime_error("Unsupported input type");
    }

    inputs.push_back(tensor);
  }

  return inputs;
}
