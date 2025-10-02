#include <ATen/core/ScalarType.h>
#include <gtest/gtest.h>

#include <stdexcept>
#include <string_view>

#include "../../../common/datatype_test_utils.hpp"
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
    ::testing::ValuesIn(starpu_server::test_utils::unsupported_scalar_types()));

class ElementSize_Unsupported
    : public ::testing::TestWithParam<at::ScalarType> {};
TEST_P(ElementSize_Unsupported, ThrowsInvalidArgument)
{
  EXPECT_THROW(starpu_server::element_size(GetParam()), std::invalid_argument);
}
INSTANTIATE_TEST_SUITE_P(
    UnsupportedTypes, ElementSize_Unsupported,
    ::testing::ValuesIn(starpu_server::test_utils::unsupported_scalar_types()));

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
    ::testing::ValuesIn(starpu_server::test_utils::supported_scalar_types()));

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
