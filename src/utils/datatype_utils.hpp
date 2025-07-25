#pragma once
#include <ATen/core/ScalarType.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "transparent_hash.hpp"

namespace starpu_server {
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
      return "FP32";
  }
}

inline auto
datatype_to_scalar_type(std::string_view dtype) -> at::ScalarType
{
  static const std::unordered_map<
      std::string, at::ScalarType, TransparentHash, std::equal_to<>>
      type_map = {{"FP32", at::kFloat},  {"FP64", at::kDouble},
                  {"FP16", at::kHalf},   {"BF16", at::kBFloat16},
                  {"INT32", at::kInt},   {"INT64", at::kLong},
                  {"INT16", at::kShort}, {"INT8", at::kChar},
                  {"UINT8", at::kByte},  {"BOOL", at::kBool}};

  const auto iter = type_map.find(dtype);
  if (iter == type_map.end()) {
    throw std::invalid_argument(
        "Unsupported tensor datatype: " + std::string(dtype));
  }
  return iter->second;
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
      return sizeof(float);
  }
}
}  // namespace starpu_server
