#include "tensor_builder.hpp"

std::vector<torch::Tensor>
TensorBuilder::from_starpu_buffers(
    const InferenceParams* params, void* buffers[])
{
  if (params->num_inputs > InferLimits::MaxInputs)
    throw std::runtime_error("[ERROR] Too many input tensors");

  std::vector<torch::Tensor> inputs;
  inputs.reserve(params->num_inputs);

  for (size_t i = 0; i < params->num_inputs; ++i) {
    void* raw_ptr = reinterpret_cast<void*>(
        static_cast<uintptr_t>(STARPU_VARIABLE_GET_PTR(buffers[i])));
    if (!raw_ptr)
      throw std::runtime_error("[ERROR] Null buffer pointer");

    std::vector<int64_t> shape(
        params->dims[i].begin(), params->dims[i].begin() + params->num_dims[i]);
    inputs.emplace_back(from_raw_ptr(raw_ptr, params->input_types[i], shape));
  }

  return inputs;
}

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

torch::Tensor
TensorBuilder::from_raw_ptr(
    void* ptr, at::ScalarType type, const std::vector<int64_t>& shape)
{
  switch (type) {
    case at::kFloat:
      return torch::from_blob(reinterpret_cast<float*>(ptr), shape, at::kFloat)
          .contiguous();
    case at::kInt:
      return torch::from_blob(reinterpret_cast<int32_t*>(ptr), shape, at::kInt)
          .contiguous();
    case at::kLong:
      return torch::from_blob(reinterpret_cast<int64_t*>(ptr), shape, at::kLong)
          .contiguous();
    case at::kBool:
      return torch::from_blob(reinterpret_cast<bool*>(ptr), shape, at::kBool)
          .contiguous();
    default:
      throw std::runtime_error("[ERROR] Unsupported input type");
  }
}