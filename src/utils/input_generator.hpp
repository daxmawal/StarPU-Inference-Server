#pragma once
#include <torch/torch.h>

#include <vector>

#include "exceptions.hpp"

namespace starpu_server::input_generator {

// =============================================================================
// input_generator
// -----------------------------------------------------------------------------
// Utility namespace for generating randomized input tensors for inference.
// Supports float, int, bool, and auto-adjusts bounds for BERT-style embeddings.
// Used to create synthetic input data for benchmarking or testing pipelines.
// =============================================================================

// -----------------------------------------------------------------------------
// Constants used for integer bounds
// -----------------------------------------------------------------------------
constexpr int64_t DEFAULT_INT_HIGH = 10;
constexpr int64_t BERT_VOCAB_SIZE = 30522;
constexpr int64_t EMBEDDING_THRESHOLD = 64;

// -----------------------------------------------------------------------------
// Internal utility: choose upper bound for integer input tensor
// - If the input looks like a [B, S] shape with S >= 64, assume it's token IDs
// -----------------------------------------------------------------------------
inline auto
get_integer_upper_bound(const std::vector<int64_t>& shape, size_t index)
    -> int64_t
{
  if (index == 0 && shape.size() == 2 && shape[1] >= EMBEDDING_THRESHOLD) {
    return BERT_VOCAB_SIZE;
  }
  return DEFAULT_INT_HIGH;
}


// -----------------------------------------------------------------------------
// Internal utility: generate a single random tensor given shape and type
// -----------------------------------------------------------------------------
inline auto
generate_random_tensor(
    const std::vector<int64_t>& shape, at::ScalarType type,
    size_t index) -> torch::Tensor
{
  const auto options = torch::TensorOptions().dtype(type);

  switch (type) {
    case at::kFloat:
    case at::kDouble:
      return torch::rand(shape, options);

    case at::kInt:
    case at::kLong:
    case at::kShort:
    case at::kChar: {
      const int64_t high = get_integer_upper_bound(shape, index);
      return torch::randint(0, high, shape, options);
    }

    case at::kBool:
      return torch::randint(0, 2, shape, options);

    default:
      throw UnsupportedDtypeException("Unsupported input type");
  }
}

// -----------------------------------------------------------------------------
// Public API: generate a list of random input tensors
// - Shapes and types must match model input signature
// -----------------------------------------------------------------------------
inline auto
generate_random_inputs(
    const std::vector<std::vector<int64_t>>& shapes,
    const std::vector<at::ScalarType>& types) -> std::vector<torch::Tensor>
{
  std::vector<torch::Tensor> inputs;
  inputs.reserve(shapes.size());

  for (size_t i = 0; i < shapes.size(); ++i) {
    const auto& shape = shapes[i];
    const at::ScalarType type = (i < types.size()) ? types[i] : at::kFloat;
    inputs.push_back(generate_random_tensor(shape, type, i));
  }

  return inputs;
}

}  // namespace starpu_server::input_generator
