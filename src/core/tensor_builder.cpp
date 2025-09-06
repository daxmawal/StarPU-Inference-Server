#include "tensor_builder.hpp"

#include <ATen/core/ScalarType.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "exceptions.hpp"
#include "inference_params.hpp"

namespace starpu_server {
// =============================================================================
// Build input tensors from StarPU raw buffers and layout metadata
// =============================================================================

auto
TensorBuilder::from_starpu_buffers(
    const InferenceParams* params, std::span<void* const> buffers,
    torch::Device device) -> std::vector<torch::Tensor>
{
  if (params->num_inputs > params->limits.max_inputs ||
      params->num_inputs > params->layout.dims.size() ||
      params->num_inputs > params->layout.num_dims.size() ||
      params->num_inputs > params->layout.input_types.size()) {
    throw InferenceExecutionException("[ERROR] Too many input tensors");
  }

  if (buffers.size() < params->num_inputs) {
    throw InferenceExecutionException("[ERROR] Too few input buffers");
  }

  std::vector<torch::Tensor> inputs;
  inputs.reserve(params->num_inputs);

  for (size_t idx = 0; idx < params->num_inputs; ++idx) {
    auto* var_iface = static_cast<starpu_variable_interface*>(buffers[idx]);
    auto input_data = var_iface->ptr;

    const auto& dims = params->layout.dims.at(idx);
    const auto raw_ndim = params->layout.num_dims.at(idx);
    TORCH_CHECK(
        raw_ndim >= 0, "Invalid number of dimensions (must be non-negative)");
    const auto ndim = static_cast<size_t>(raw_ndim);
    if (dims.size() != ndim) {
      throw InferenceExecutionException("[ERROR] Tensor layout mismatch");
    }

    const at::ScalarType dtype = params->layout.input_types.at(idx);
    inputs.emplace_back(from_raw_ptr(input_data, dtype, dims, device));
  }

  return inputs;
}

// =============================================================================
// Copy output tensor to raw buffer, with size and type checks
// =============================================================================

void
TensorBuilder::copy_output_to_buffer(
    const at::Tensor& output, void* buffer_ptr, int64_t expected_numel,
    at::ScalarType expected_type)
{
  if (buffer_ptr == nullptr) {
    throw InferenceExecutionException("[ERROR] Output buffer pointer is null");
  }
  if (output.numel() != expected_numel) {
    throw InferenceExecutionException("[ERROR] Output size mismatch");
  }

  if (output.scalar_type() != expected_type) {
    throw InferenceExecutionException("[ERROR] Output type mismatch");
  }

  if (!output.is_contiguous()) {
    throw InferenceExecutionException(
        "[ERROR] Output tensor must be contiguous");
  }

  std::memcpy(buffer_ptr, output.data_ptr(), output.nbytes());
}

// =============================================================================
// Wrap raw memory into a tensor view (non-owning, no copy)
// =============================================================================

auto
TensorBuilder::from_raw_ptr(
    uintptr_t ptr, at::ScalarType type, const std::vector<int64_t>& shape,
    torch::Device device) -> torch::Tensor
{
  auto options = torch::TensorOptions().dtype(type).device(device);

  switch (type) {
    case at::kFloat:
      return torch::from_blob(std::bit_cast<float*>(ptr), shape, options);
    case at::kDouble:
      return torch::from_blob(std::bit_cast<double*>(ptr), shape, options);
    case at::kInt:
      return torch::from_blob(std::bit_cast<int32_t*>(ptr), shape, options);
    case at::kLong:
      return torch::from_blob(std::bit_cast<int64_t*>(ptr), shape, options);
    case at::kShort:
      return torch::from_blob(std::bit_cast<int16_t*>(ptr), shape, options);
    case at::kChar:
      return torch::from_blob(std::bit_cast<int8_t*>(ptr), shape, options);
    case at::kByte:
      return torch::from_blob(std::bit_cast<uint8_t*>(ptr), shape, options);
    case at::kBool:
      return torch::from_blob(std::bit_cast<bool*>(ptr), shape, options);
    default:
      throw InferenceExecutionException(
          "[ERROR] Unsupported input type for tensor");
  }
}
}  // namespace starpu_server
