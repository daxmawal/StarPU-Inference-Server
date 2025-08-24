#pragma once
#include <torch/torch.h>

#include <vector>

#include "exceptions.hpp"
#include "runtime_config.hpp"

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
// - If the input looks like a [B, S] dim with S >= 64, assume it's token IDs
// -----------------------------------------------------------------------------
inline auto
get_integer_upper_bound(const std::vector<int64_t>& dim, size_t index)
    -> int64_t
{
  if (index == 0 && dim.size() == 2 && dim[1] >= EMBEDDING_THRESHOLD) {
    return BERT_VOCAB_SIZE;
  }
  return DEFAULT_INT_HIGH;
}


// -----------------------------------------------------------------------------
// Internal utility: generate a single random tensor given dim and type
// -----------------------------------------------------------------------------
inline auto
generate_random_tensor(
    const std::vector<int64_t>& dim, at::ScalarType type,
    size_t index) -> torch::Tensor
{
  const auto options = torch::TensorOptions().dtype(type);

  switch (type) {
    case at::kFloat:
    case at::kDouble:
      return torch::rand(dim, options);
    case at::kInt:
    case at::kLong:
    case at::kShort:
    case at::kChar: {
      const int64_t high = get_integer_upper_bound(dim, index);
      return torch::randint(0, high, dim, options);
    }
    case at::kBool:
      return torch::randint(0, 2, dim, options);
    default:
      throw UnsupportedDtypeException("Unsupported input type");
  }
}

// -----------------------------------------------------------------------------
// Public API: generate a list of random input tensors
// - dims and types must match model input signature
// -----------------------------------------------------------------------------
inline auto
generate_random_inputs(const std::vector<TensorConfig>& tensors)
    -> std::vector<torch::Tensor>
{
  std::vector<torch::Tensor> inputs;
  inputs.reserve(tensors.size());

  for (size_t i = 0; i < tensors.size(); ++i) {
    const auto& t = tensors[i];
    inputs.push_back(generate_random_tensor(t.dims, t.type, i));
  }

  return inputs;
}

}  // namespace starpu_server::input_generator
