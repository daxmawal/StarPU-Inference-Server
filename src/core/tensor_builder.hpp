#pragma once
#include <starpu.h>

#include <cstddef>
#include <span>
#include <vector>

#include "inference_params.hpp"

namespace starpu_server {

using StarpuBufferInterface = starpu_variable_interface;
using StarpuBufferPtr = StarpuBufferInterface*;
using StarpuBufferSpan = std::span<StarpuBufferPtr const>;

// =============================================================================
// TensorBuilder: utility class for wrapping StarPU buffers as torch::Tensor
// =============================================================================
class TensorBuilder {
 public:
  TensorBuilder() = delete;

  [[nodiscard]] static auto from_starpu_buffers(
      InferenceParams* params, StarpuBufferSpan buffers,
      torch::Device device) -> std::vector<torch::Tensor>;

  [[nodiscard]] static auto prepare_input_tensors(
      InferenceParams* params, StarpuBufferSpan buffers,
      torch::Device device) -> const std::vector<torch::Tensor>&;

  [[nodiscard]] static auto prepare_input_ivalues(
      InferenceParams* params, StarpuBufferSpan buffers,
      torch::Device device) -> const std::vector<c10::IValue>&;

  static void copy_output_to_buffer(
      const at::Tensor& output, std::span<std::byte> buffer,
      int64_t expected_numel, at::ScalarType expected_type);

 private:
  static auto from_raw_ptr(
      uintptr_t ptr, at::ScalarType type, const std::vector<int64_t>& shape,
      torch::Device device) -> torch::Tensor;
};
}  // namespace starpu_server
