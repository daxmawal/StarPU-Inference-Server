#include "tensor_builder.hpp"

// ============================================================================
// Converts StarPU buffers into Torch tensors on the specified device
// ============================================================================
std::vector<torch::Tensor>
TensorBuilder::from_starpu_buffers(
    const InferenceParams* params, void* buffers[], torch::Device device)
{
  if (params->num_inputs > InferLimits::MaxInputs) {
    throw std::runtime_error("[ERROR] Too many input tensors");
  }

  std::vector<torch::Tensor> inputs;
  inputs.reserve(params->num_inputs);

  for (size_t i = 0; i < params->num_inputs; ++i) {
    void* input_data = reinterpret_cast<void*>(
        static_cast<uintptr_t>(STARPU_VARIABLE_GET_PTR(buffers[i])));

    if (!input_data) {
      throw std::runtime_error(
          "[ERROR] Null buffer pointer for input " + std::to_string(i));
    }

    const auto& dims = params->layout.dims[i];
    const auto raw_ndim = params->layout.num_dims[i];
    TORCH_CHECK(
        raw_ndim >= 0, "Invalid number of dimensions (must be non-negative)");
    const size_t ndim = static_cast<size_t>(raw_ndim);
    std::vector<int64_t> shape(dims.begin(), dims.begin() + ndim);

    at::ScalarType dtype = params->layout.input_types[i];
    inputs.emplace_back(from_raw_ptr(input_data, dtype, shape, device));
  }

  return inputs;
}

// ============================================================================
// Copies the inference output tensor into a raw StarPU buffer
// ============================================================================
void
TensorBuilder::copy_output_to_buffer(
    const at::Tensor& output, void* buffer_ptr, int64_t expected_numel)
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
torch::Tensor
TensorBuilder::from_raw_ptr(
    void* ptr, at::ScalarType type, const std::vector<int64_t>& shape,
    torch::Device device)
{
  auto options = torch::TensorOptions().dtype(type).device(device);

  switch (type) {
    case at::kFloat:
      return torch::from_blob(static_cast<float*>(ptr), shape, options)
          .contiguous();
    case at::kInt:
      return torch::from_blob(static_cast<int32_t*>(ptr), shape, options)
          .contiguous();
    case at::kLong:
      return torch::from_blob(static_cast<int64_t*>(ptr), shape, options)
          .contiguous();
    case at::kBool:
      return torch::from_blob(static_cast<bool*>(ptr), shape, options)
          .contiguous();
    default:
      throw std::runtime_error("[ERROR] Unsupported input type for tensor");
  }
}
