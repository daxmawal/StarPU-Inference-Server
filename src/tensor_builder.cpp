#include "tensor_builder.hpp"

#include <ATen/core/ScalarType.h>
#include <c10/core/Device.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "inference_params.hpp"

// ============================================================================
// Converts StarPU buffers into Torch tensors on the specified device
// ============================================================================
auto
TensorBuilder::from_starpu_buffers(
    const InferenceParams* params, const std::vector<void*>& buffers,
    const torch::Device device) -> std::vector<torch::Tensor>
{
  if (params->num_inputs > InferLimits::MaxInputs) {
    throw std::runtime_error("[ERROR] Too many input tensors");
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
    const std::vector<int64_t> shape(dims.begin(), dims.begin() + ndim);

    const at::ScalarType dtype = params->layout.input_types.at(idx);
    inputs.emplace_back(from_raw_ptr(input_data, dtype, shape, device));
  }

  return inputs;
}

// ============================================================================
// Copies the inference output tensor into a raw StarPU buffer
// ============================================================================
void
TensorBuilder::copy_output_to_buffer(
    const at::Tensor& output, void* buffer_ptr, const int64_t expected_numel)
{
  if (output.numel() != expected_numel) {
    throw std::runtime_error("[ERROR] Output size mismatch");
  }

  if (output.scalar_type() != at::kFloat) {
    throw std::runtime_error("[ERROR] Expected float output tensor");
  }

  std::memcpy(
      buffer_ptr, output.data_ptr<float>(),
      static_cast<size_t>(output.numel()) * sizeof(float));
}

// ============================================================================
// Constructs a Torch tensor from a raw pointer with shape and type
// ============================================================================
auto
TensorBuilder::from_raw_ptr(
    uintptr_t ptr, const at::ScalarType type, const std::vector<int64_t>& shape,
    const torch::Device device) -> torch::Tensor
{
  auto options = torch::TensorOptions().dtype(type).device(device);

  switch (type) {
    case at::kFloat:
      return torch::from_blob(std::bit_cast<float*>(ptr), shape, options)
          .contiguous();
    case at::kInt:
      return torch::from_blob(std::bit_cast<int32_t*>(ptr), shape, options)
          .contiguous();
    case at::kLong:
      return torch::from_blob(std::bit_cast<int64_t*>(ptr), shape, options)
          .contiguous();
    case at::kBool:
      return torch::from_blob(std::bit_cast<bool*>(ptr), shape, options)
          .contiguous();
    default:
      throw std::runtime_error("[ERROR] Unsupported input type for tensor");
  }
}
