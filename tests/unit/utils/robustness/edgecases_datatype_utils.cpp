#include <ATen/core/ScalarType.h>
#include <gtest/gtest.h>

#include <array>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "utils/datatype_utils.hpp"

namespace {
using Enum = at::ScalarType;
using U = std::underlying_type_t<Enum>;

constexpr std::array<Enum, 10> kSupportedArr = {
    at::kFloat, at::kDouble, at::kHalf, at::kBFloat16, at::kInt,
    at::kLong,  at::kShort,  at::kChar, at::kByte,     at::kBool};

auto
supported_set() -> const std::unordered_set<Enum>&
{
  static const std::unordered_set<Enum> cache(
      std::begin(kSupportedArr), std::end(kSupportedArr));
  return cache;
}

auto
supported_vec() -> std::vector<Enum>
{
  return {std::begin(kSupportedArr), std::end(kSupportedArr)};
}

auto
all_types() -> std::vector<Enum>
{
  const int first = static_cast<int>(std::to_underlying(Enum::Undefined));
  const int last = static_cast<int>(std::to_underlying(Enum::NumOptions));

  std::vector<Enum> types;
  types.reserve(static_cast<std::size_t>(last - first));
  for (int i = first; i < last; ++i) {
    types.push_back(static_cast<Enum>(static_cast<U>(i)));
  }
  return types;
}

auto
unsupported_vec() -> std::vector<Enum>
{
  const auto& sup = supported_set();
  std::vector<Enum> types;
  for (const auto type : all_types()) {
    if (!sup.contains(type)) {
      types.push_back(type);
    }
  }
  return types;
}
}  // namespace

class ScalarToDatatype_Unsupported
    : public ::testing::TestWithParam<at::ScalarType> {};

TEST_P(ScalarToDatatype_Unsupported, ThrowsInvalidArgument)
{
  EXPECT_THROW(
      starpu_server::scalar_type_to_datatype(GetParam()),
      std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(
    UnsupportedTypes, ScalarToDatatype_Unsupported,
    ::testing::ValuesIn(unsupported_vec()));

class ElementSize_Unsupported
    : public ::testing::TestWithParam<at::ScalarType> {};
TEST_P(ElementSize_Unsupported, ThrowsInvalidArgument)
{
  EXPECT_THROW(starpu_server::element_size(GetParam()), std::invalid_argument);
}
INSTANTIATE_TEST_SUITE_P(
    UnsupportedTypes, ElementSize_Unsupported,
    ::testing::ValuesIn(unsupported_vec()));

class DatatypeString_Invalid
    : public ::testing::TestWithParam<std::string_view> {};
TEST_P(DatatypeString_Invalid, ThrowsInvalidArgument)
{
  EXPECT_THROW(
      starpu_server::datatype_to_scalar_type(GetParam()),
      std::invalid_argument);
}
INSTANTIATE_TEST_SUITE_P(
    DatatypeUtils, DatatypeString_Invalid,
    ::testing::Values(
        "BADTYPE", "", "#FP32", "UNKNOWN", "notatype", " ", "!", "123",
        "FP32extra", std::string_view(), "𝔽ℙ𝟛𝟚", "fp32\n"));

class ScalarToDatatype_Supported
    : public ::testing::TestWithParam<at::ScalarType> {};

TEST_P(ScalarToDatatype_Supported, NoThrowForSupportedTypes)
{
  const auto type = GetParam();
  EXPECT_NO_THROW({
    auto name = starpu_server::scalar_type_to_datatype(type);
    auto size = starpu_server::element_size(type);
    (void)name;
    (void)size;
  });
}

INSTANTIATE_TEST_SUITE_P(
    SupportedTypes, ScalarToDatatype_Supported,
    ::testing::ValuesIn(supported_vec()));

class InvalidDatatypeTest : public ::testing::TestWithParam<std::string_view> {
};

TEST_P(InvalidDatatypeTest, ThrowsInvalidArgument)
{
  EXPECT_THROW(
      starpu_server::datatype_to_scalar_type(GetParam()),
      std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(
    DatatypeUtils, InvalidDatatypeTest,
    ::testing::Values(
        "BADTYPE", "", "#FP32", "UNKNOWN", "notatype", " ", "!", "123",
        "FP32extra", std::string_view(), "𝔽ℙ𝟛𝟚", "fp32\n"));
