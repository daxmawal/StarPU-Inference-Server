#pragma once

#include <ATen/core/ScalarType.h>

#include <array>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace starpu_server::test_utils {

inline constexpr std::array<at::ScalarType, 10> kSupportedScalarTypes = {
    at::kFloat, at::kDouble, at::kHalf, at::kBFloat16, at::kInt,
    at::kLong,  at::kShort,  at::kChar, at::kByte,     at::kBool};

inline auto
supported_scalar_types() -> const std::vector<at::ScalarType>&
{
  static const std::vector<at::ScalarType> kSupported{
      kSupportedScalarTypes.begin(), kSupportedScalarTypes.end()};
  return kSupported;
}

inline auto
unsupported_scalar_types() -> const std::vector<at::ScalarType>&
{
  using Enum = at::ScalarType;
  using U = std::underlying_type_t<Enum>;

  static const std::vector<Enum> kUnsupported = [] {
    const std::unordered_set<Enum> supported(
        kSupportedScalarTypes.begin(), kSupportedScalarTypes.end());

    const int first = static_cast<int>(std::to_underlying(Enum::Undefined));
    const int last = static_cast<int>(std::to_underlying(Enum::NumOptions));

    std::vector<Enum> result;
    result.reserve(static_cast<std::size_t>(last - first));
    for (int value = first; value < last; ++value) {
      const auto type = static_cast<Enum>(static_cast<U>(value));
      if (!supported.contains(type)) {
        result.push_back(type);
      }
    }
    return result;
  }();

  return kUnsupported;
}

}  // namespace starpu_server::test_utils
