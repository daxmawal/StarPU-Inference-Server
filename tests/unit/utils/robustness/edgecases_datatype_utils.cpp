#include <ATen/core/ScalarType.h>
#include <gtest/gtest.h>

#include <stdexcept>
#include <utility>
#include <unordered_set>

#include "utils/datatype_utils.hpp"

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
    ::testing::Values(
        static_cast<at::ScalarType>(-1), at::kComplexFloat,
        at::kComplexDouble, at::kQInt8, at::kQUInt8));

class ElementSize_Unsupported
    : public ::testing::TestWithParam<at::ScalarType> {};
TEST_P(ElementSize_Unsupported, ThrowsInvalidArgument)
{
  EXPECT_THROW(
      starpu_server::element_size(GetParam()), std::invalid_argument);
}
INSTANTIATE_TEST_SUITE_P(
    UnsupportedTypes, ElementSize_Unsupported,
    ::testing::Values(
        static_cast<at::ScalarType>(-1), at::kComplexFloat,
        at::kComplexDouble, at::kQInt8, at::kQUInt8));

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

TEST(DatatypeUtils_Robustesse, ScalarToDatatype_AllEnumValues)
{
  using Enum = at::ScalarType;
  using U = std::underlying_type_t<Enum>;

  const std::unordered_set<Enum> supported = {
      at::kFloat,    at::kDouble, at::kHalf,    at::kBFloat16, at::kInt,
      at::kLong,     at::kShort,  at::kChar,    at::kByte,     at::kBool};

  const int first = static_cast<int>(std::to_underlying(Enum::Undefined));
  const int last = static_cast<int>(std::to_underlying(Enum::NumOptions));

  for (int i = first; i < last; ++i) {
    const auto type = static_cast<Enum>(static_cast<U>(i));
    if (supported.count(type)) {
      EXPECT_NO_THROW({
        auto name = starpu_server::scalar_type_to_datatype(type);
        auto size = starpu_server::element_size(type);
        (void)name;
        (void)size;
      });
    } else {
      EXPECT_THROW(
          starpu_server::scalar_type_to_datatype(type),
          std::invalid_argument);
      EXPECT_THROW(
          starpu_server::element_size(type), std::invalid_argument);
    }
  }
}

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
