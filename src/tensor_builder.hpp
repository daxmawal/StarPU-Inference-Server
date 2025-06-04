#pragma once
#include <starpu.h>

#include "inference_params.hpp"

// =============================================================================
// TensorBuilder: utility class for managing Torch tensors and StarPU buffers
// =============================================================================
class TensorBuilder {
 public:
  TensorBuilder() = delete;

  // ---- Conversion and wrapping utilities ----

  /// Builds input tensors from StarPU buffers based on inference parameters.
  ///
  /// params   Pointer to inference parameters with layout/type info
  /// buffers  StarPU data buffer array
  /// device   Target device (CPU or CUDA)
  /// Vector of Torch tensors on the target device
  static auto from_starpu_buffers(
      const InferenceParams* params, const std::vector<void*>& buffers,
      torch::Device device) -> std::vector<torch::Tensor>;

  /// Copies a Torch tensor into a raw StarPU output buffer.
  ///
  /// output          The tensor to copy
  /// buffer_ptr      Raw pointer to destination memory
  /// expected_numel  Number of elements expected in the output
  static void copy_output_to_buffer(
      const at::Tensor& output, void* buffer_ptr, int64_t expected_numel);

 private:
  // ---- Internal helper ----

  /// Wraps a raw pointer into a Torch tensor.
  ///
  /// ptr     Raw memory pointer
  /// type    Scalar type (float, int, etc.)
  /// shape   Tensor shape
  /// device  Target device for the tensor
  /// Tensor referencing the raw memory
  static auto from_raw_ptr(
      uintptr_t ptr, at::ScalarType type, const std::vector<int64_t>& shape,
      torch::Device device) -> torch::Tensor;
};
