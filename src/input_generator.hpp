#pragma once

// ----------------------------------------------------------------------------
// input_generator.hpp
// Generates random input tensors based on user-defined shapes and types.
// ----------------------------------------------------------------------------

#include <torch/torch.h>

#include <vector>

// =============================================================================
// generate_random_inputs
// Given a set of shapes and data types, generate a vector of random tensors.
// - Floating types use torch::rand
// - Integer types use torch::randint (defaults to [0,10), or [0,30522) for
// BERT-like embeddings)
// - Bool uses [0,2)
// Throws if an unsupported type is encountered.
// =============================================================================
inline std::vector<torch::Tensor>
generate_random_inputs(
    const std::vector<std::vector<int64_t>>& shapes,
    const std::vector<at::ScalarType>& types)
{
  std::vector<torch::Tensor> inputs;
  inputs.reserve(shapes.size());

  for (size_t i = 0; i < shapes.size(); ++i) {
    const auto& shape = shapes[i];
    at::ScalarType type = (i < types.size()) ? types[i] : at::kFloat;

    const torch::TensorOptions options = torch::TensorOptions().dtype(type);
    torch::Tensor tensor;

    switch (type) {
      case at::kFloat:
      case at::kDouble:
        tensor = torch::rand(shape, options);
        break;

      case at::kInt:
      case at::kLong:
      case at::kShort:
      case at::kChar: {
        int64_t low = 0;
        int64_t high = 10;

        // Special heuristic: use higher vocab-like range for first tensor with
        // >= 64 elements in dim 1
        if (i == 0 && shape.size() == 2 && shape[1] >= 64) {
          high = 30522;
        }

        tensor = torch::randint(low, high, shape, options);
        break;
      }

      case at::kBool:
        tensor = torch::randint(0, 2, shape, options);
        break;

      default:
        throw std::runtime_error("Unsupported input type");
    }

    inputs.push_back(tensor);
  }

  return inputs;
}