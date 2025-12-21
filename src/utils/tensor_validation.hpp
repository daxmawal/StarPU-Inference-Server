#pragma once

#include <torch/torch.h>

#include <optional>
#include <string>
#include <string_view>

namespace starpu_server::tensor_validation {

enum class Failure : std::uint8_t {
  Undefined,
  NullData,
  NotCpu,
  NotContiguous,
};

inline auto
format_failure(std::string_view label, Failure failure) -> std::string
{
  switch (failure) {
    case Failure::Undefined:
      return std::string("Tensor '") + std::string(label) + "' is undefined.";
    case Failure::NullData:
      return std::string("Tensor '") + std::string(label) + "' is invalid.";
    case Failure::NotCpu:
      return std::string("Tensor '") + std::string(label) +
             "' must reside on CPU";
    case Failure::NotContiguous:
      return std::string("Tensor '") + std::string(label) +
             "' must be contiguous.";
  }
  return std::string("Tensor '") + std::string(label) + "' validation failed.";
}

inline auto
validate_cpu_contiguous_tensor(
    const torch::Tensor& tensor, std::string_view label,
    bool check_data_ptr = false) -> std::optional<std::string>
{
  if (!tensor.defined()) {
    return format_failure(label, Failure::Undefined);
  }
  if (check_data_ptr && tensor.data_ptr() == nullptr) {
    return format_failure(label, Failure::NullData);
  }
  if (!tensor.device().is_cpu()) {
    return format_failure(label, Failure::NotCpu);
  }
  if (!tensor.is_contiguous()) {
    return format_failure(label, Failure::NotContiguous);
  }
  return std::nullopt;
}

}  // namespace starpu_server::tensor_validation
