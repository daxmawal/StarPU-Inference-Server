#pragma once
#include <starpu.h>

#include "inference_params.hpp"

namespace starpu_server {
// =============================================================================
// TensorBuilder: utility class for wrapping StarPU buffers as torch::Tensor
// =============================================================================

class TensorBuilder {
 public:
  TensorBuilder() = delete;

  [[nodiscard]] static auto from_starpu_buffers(
      const InferenceParams* params, const std::vector<void*>& buffers,
      torch::Device device) -> std::vector<torch::Tensor>;

  static void copy_output_to_buffer(
      const at::Tensor& output, void* buffer_ptr, int64_t expected_numel);

 private:
  static auto from_raw_ptr(
      uintptr_t ptr, at::ScalarType type, const std::vector<int64_t>& shape,
      torch::Device device) -> torch::Tensor;
};
}  // namespace starpu_server