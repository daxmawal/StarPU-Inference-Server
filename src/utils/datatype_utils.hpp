#pragma once
#include <ATen/core/ScalarType.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace starpu_server {
namespace detail {
struct ScalarTypeAlias {
  std::string_view name;
  at::ScalarType type;
};

inline constexpr std::array<ScalarTypeAlias, 10> kDatatypeAliases = {{
    {"fp32", at::kFloat},
    {"fp64", at::kDouble},
    {"fp16", at::kHalf},
    {"bf16", at::kBFloat16},
    {"int32", at::kInt},
    {"int64", at::kLong},
    {"int16", at::kShort},
    {"int8", at::kChar},
    {"uint8", at::kByte},
    {"bool", at::kBool},
}};

inline constexpr std::array<ScalarTypeAlias, 12> kTypeAliases = {{
    {"float", at::kFloat},
    {"float32", at::kFloat},
    {"double", at::kDouble},
    {"float64", at::kDouble},
    {"half", at::kHalf},
    {"float16", at::kHalf},
    {"bfloat16", at::kBFloat16},
    {"int", at::kInt},
    {"long", at::kLong},
    {"short", at::kShort},
    {"char", at::kChar},
    {"byte", at::kByte},
}};

template <std::size_t N>
inline auto
lookup_scalar_type(
    std::string_view key, const std::array<ScalarTypeAlias, N>& aliases)
    -> std::optional<at::ScalarType>
{
  for (const auto& alias : aliases) {
    if (alias.name == key) {
      return alias.type;
    }
  }
  return std::nullopt;
}
}  // namespace detail

// =============================================================================
// datatype_utils
// -----------------------------------------------------------------------------
// Helper functions to convert between torch scalar types, string names, and
// Triton-style datatypes. Also provides utility to obtain the element size of
// a tensor element given its scalar type.
// =============================================================================
inline auto
scalar_type_to_datatype(at::ScalarType type) -> std::string
{
  switch (type) {
    case at::kFloat:
      return "FP32";
    case at::kDouble:
      return "FP64";
    case at::kHalf:
      return "FP16";
    case at::kBFloat16:
      return "BF16";
    case at::kInt:
      return "INT32";
    case at::kLong:
      return "INT64";
    case at::kShort:
      return "INT16";
    case at::kChar:
      return "INT8";
    case at::kByte:
      return "UINT8";
    case at::kBool:
      return "BOOL";
    default:
      throw std::invalid_argument("Unsupported at::ScalarType");
  }
}

inline auto
datatype_to_scalar_type(std::string_view dtype) -> at::ScalarType
{
  std::string dtype_lower(dtype);
  std::ranges::transform(
      dtype_lower, dtype_lower.begin(), [](unsigned char character) noexcept {
        return static_cast<char>(std::tolower(character));
      });

  if (const auto type =
          detail::lookup_scalar_type(dtype_lower, detail::kDatatypeAliases)) {
    return *type;
  }
  throw std::invalid_argument(
      "Unsupported tensor datatype: " + std::string(dtype));
}

inline auto
string_to_scalar_type(std::string_view type_str) -> at::ScalarType
{
  std::string key(type_str);
  std::ranges::transform(
      key, key.begin(), [](unsigned char character) noexcept {
        return static_cast<char>(std::tolower(character));
      });
  if (constexpr std::string_view type_prefix = "type_";
      key.starts_with(type_prefix)) {
    key.erase(0, type_prefix.size());
  }

  if (const auto type = detail::lookup_scalar_type(key, detail::kTypeAliases)) {
    return *type;
  }
  if (const auto type =
          detail::lookup_scalar_type(key, detail::kDatatypeAliases)) {
    return *type;
  }
  throw std::invalid_argument("Unsupported type: " + std::string(type_str));
}

inline auto
scalar_type_to_string(at::ScalarType type) -> std::string
{
  return scalar_type_to_datatype(type);
}

inline auto
element_size(at::ScalarType type) -> size_t
{
  switch (type) {
    case at::kFloat:
      return sizeof(float);
    case at::kDouble:
      return sizeof(double);
    case at::kHalf:
    case at::kBFloat16:
      return 2U;
    case at::kInt:
      return sizeof(int32_t);
    case at::kLong:
      return sizeof(int64_t);
    case at::kShort:
      return sizeof(int16_t);
    case at::kChar:
      return sizeof(int8_t);
    case at::kByte:
      return sizeof(uint8_t);
    case at::kBool:
      return sizeof(bool);
    default:
      throw std::invalid_argument("Unsupported at::ScalarType");
  }
}
}  // namespace starpu_server
