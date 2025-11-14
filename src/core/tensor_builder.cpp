#include "tensor_builder.hpp"

#include <ATen/core/ScalarType.h>
#include <c10/core/Device.h>
#include <c10/core/Storage.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <numeric>
#include <span>
#include <vector>

#include "exceptions.hpp"
#include "inference_params.hpp"
#include "utils/datatype_utils.hpp"

namespace starpu_server {
namespace {
void
noop_deleter(void* /*unused*/) noexcept
{
  // Storage is backed by StarPU pool allocations which remain owned elsewhere.
}

auto
compute_default_strides(const std::vector<int64_t>& dims)
    -> std::vector<int64_t>
{
  std::vector<int64_t> strides(dims.size(), 1);
  int64_t stride = 1;
  for (int64_t idx = static_cast<int64_t>(dims.size()) - 1; idx >= 0; --idx) {
    strides[static_cast<size_t>(idx)] = stride;
    stride *= dims[static_cast<size_t>(idx)];
  }
  return strides;
}

auto
compute_numel(const std::vector<int64_t>& dims) -> int64_t
{
  return std::accumulate(
      dims.begin(), dims.end(), static_cast<int64_t>(1), std::multiplies<>());
}

void
validate_input_layout(const InferenceParams* params, StarpuBufferSpan buffers)
{
  if (params == nullptr) {
    throw InferenceExecutionException("[ERROR] InferenceParams is null");
  }
  if (params->num_inputs > params->limits.max_inputs ||
      params->num_inputs > params->layout.dims.size() ||
      params->num_inputs > params->layout.num_dims.size() ||
      params->num_inputs > params->layout.input_types.size()) {
    throw InferenceExecutionException("[ERROR] Too many input tensors");
  }
  if (buffers.size() < params->num_inputs) {
    throw InferenceExecutionException("[ERROR] Too few input buffers");
  }
}

void
assign_tensor_view(
    torch::Tensor& tensor, uintptr_t raw_ptr, at::ScalarType dtype,
    const std::vector<int64_t>& dims, torch::Device device)
{
  const auto options = torch::TensorOptions().dtype(dtype).device(device);
  if (!tensor.defined()) {
    tensor = torch::from_blob(std::bit_cast<void*>(raw_ptr), dims, options);
    return;
  }

  auto strides = compute_default_strides(dims);
  const int64_t numel = compute_numel(dims);
  const size_t byte_size =
      static_cast<size_t>(std::max<int64_t>(numel, 0)) * element_size(dtype);

  c10::DataPtr data_ptr(
      std::bit_cast<void*>(raw_ptr), std::bit_cast<void*>(raw_ptr),
      &noop_deleter, device.type());
  auto storage = c10::Storage(
      c10::Storage::use_byte_size_t(), byte_size, std::move(data_ptr),
      /*allocator=*/nullptr, /*resizable=*/false);

  tensor.set_(storage, 0, dims, strides);
}

void
refresh_input_cache(
    InferenceParams* params, StarpuBufferSpan buffers, torch::Device device)
{
  validate_input_layout(params, buffers);

  auto& tensor_cache = params->cached_input_tensors;
  auto& ivalue_cache = params->cached_input_ivalues;
  tensor_cache.resize(params->num_inputs);
  ivalue_cache.resize(params->num_inputs);

  for (size_t idx = 0; idx < params->num_inputs; ++idx) {
    const auto* var_iface = buffers[idx];
    if (var_iface == nullptr) {
      throw InferenceExecutionException("[ERROR] StarPU buffer is null");
    }

    const auto& dims = params->layout.dims.at(idx);
    const auto raw_ndim = params->layout.num_dims.at(idx);
    TORCH_CHECK(
        raw_ndim >= 0, "Invalid number of dimensions (must be non-negative)");
    const auto ndim = static_cast<size_t>(raw_ndim);
    if (dims.size() != ndim) {
      throw InferenceExecutionException("[ERROR] Tensor layout mismatch");
    }

    const at::ScalarType dtype = params->layout.input_types.at(idx);
    assign_tensor_view(tensor_cache[idx], var_iface->ptr, dtype, dims, device);
    ivalue_cache[idx] = tensor_cache[idx];
  }
}
}  // namespace

// =============================================================================
// Build input tensors from StarPU raw buffers and layout metadata
// =============================================================================

auto
TensorBuilder::from_starpu_buffers(
    InferenceParams* params, StarpuBufferSpan buffers,
    torch::Device device) -> std::vector<torch::Tensor>
{
  const auto& cached = prepare_input_tensors(params, buffers, device);
  return {cached.begin(), cached.end()};
}

auto
TensorBuilder::prepare_input_tensors(
    InferenceParams* params, StarpuBufferSpan buffers,
    torch::Device device) -> const std::vector<torch::Tensor>&
{
  refresh_input_cache(params, buffers, device);
  return params->cached_input_tensors;
}

auto
TensorBuilder::prepare_input_ivalues(
    InferenceParams* params, StarpuBufferSpan buffers,
    torch::Device device) -> const std::vector<c10::IValue>&
{
  refresh_input_cache(params, buffers, device);
  return params->cached_input_ivalues;
}

// =============================================================================
// Copy output tensor to raw buffer, with size and type checks
// =============================================================================

void
TensorBuilder::copy_output_to_buffer(
    const at::Tensor& output, std::span<std::byte> buffer,
    int64_t expected_numel, at::ScalarType expected_type)
{
  if (buffer.data() == nullptr) {
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

  const auto bytes_required = output.nbytes();
  if (buffer.size() != bytes_required) {
    throw InferenceExecutionException(
        "[ERROR] Output buffer size mismatch in bytes");
  }

  std::memcpy(buffer.data(), output.data_ptr(), bytes_required);
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
