#include <ATen/core/ScalarType.h>
#include <gtest/gtest.h>

#include "utils/datatype_utils.hpp"

class ScalarToDatatypeCase
    : public ::testing::TestWithParam<std::pair<at::ScalarType, std::string>> {
};

TEST_P(ScalarToDatatypeCase, ConvertsScalarTypeToDatatype)
{
  const auto& [type, expected] = GetParam();
  EXPECT_EQ(starpu_server::scalar_type_to_datatype(type), expected);
}

INSTANTIATE_TEST_SUITE_P(
    KnownTypes, ScalarToDatatypeCase,
    ::testing::Values(
        std::pair{at::kFloat, std::string{"FP32"}},
        std::pair{at::kDouble, std::string{"FP64"}},
        std::pair{at::kHalf, std::string{"FP16"}},
        std::pair{at::kBFloat16, std::string{"BF16"}},
        std::pair{at::kInt, std::string{"INT32"}},
        std::pair{at::kLong, std::string{"INT64"}},
        std::pair{at::kShort, std::string{"INT16"}},
        std::pair{at::kChar, std::string{"INT8"}},
        std::pair{at::kByte, std::string{"UINT8"}},
        std::pair{at::kBool, std::string{"BOOL"}}));

INSTANTIATE_TEST_SUITE_P(
    FallbackTypes, ScalarToDatatypeCase,
    ::testing::Values(
        std::pair{static_cast<at::ScalarType>(-1), std::string{"FP32"}},
        std::pair{at::kComplexFloat, std::string{"FP32"}},
        std::pair{at::kComplexDouble, std::string{"FP32"}},
        std::pair{at::kQInt8, std::string{"FP32"}},
        std::pair{at::kQUInt8, std::string{"FP32"}}));

class ElementSizeCase
    : public ::testing::TestWithParam<std::pair<at::ScalarType, size_t>> {};

TEST_P(ElementSizeCase, ReturnsCorrectElementSize)
{
  const auto& [type, expected_size] = GetParam();
  EXPECT_EQ(starpu_server::element_size(type), expected_size);
}

INSTANTIATE_TEST_SUITE_P(
    KnownTypes, ElementSizeCase,
    ::testing::Values(
        std::pair{at::kFloat, sizeof(float)},
        std::pair{at::kDouble, sizeof(double)},
        std::pair{at::kHalf, size_t{2U}}, std::pair{at::kBFloat16, size_t{2U}},
        std::pair{at::kInt, sizeof(int32_t)},
        std::pair{at::kLong, sizeof(int64_t)},
        std::pair{at::kShort, sizeof(int16_t)},
        std::pair{at::kChar, sizeof(int8_t)},
        std::pair{at::kByte, sizeof(uint8_t)},
        std::pair{at::kBool, sizeof(bool)}));

INSTANTIATE_TEST_SUITE_P(
    FallbackTypes, ElementSizeCase,
    ::testing::Values(
        std::pair{static_cast<at::ScalarType>(-1), sizeof(float)},
        std::pair{at::kComplexFloat, sizeof(float)},
        std::pair{at::kComplexDouble, sizeof(float)},
        std::pair{at::kQInt8, sizeof(float)},
        std::pair{at::kQUInt8, sizeof(float)}));

class DatatypeToScalarCase
    : public ::testing::TestWithParam<std::pair<std::string, at::ScalarType>> {
};

TEST_P(DatatypeToScalarCase, ConvertsDatatypeToScalarType)
{
  const auto& [str, expected_type] = GetParam();
  EXPECT_EQ(starpu_server::datatype_to_scalar_type(str), expected_type);
}

INSTANTIATE_TEST_SUITE_P(
    ValidExactMatch, DatatypeToScalarCase,
    ::testing::Values(
        std::pair{std::string{"FP32"}, at::kFloat},
        std::pair{std::string{"FP64"}, at::kDouble},
        std::pair{std::string{"FP16"}, at::kHalf},
        std::pair{std::string{"BF16"}, at::kBFloat16},
        std::pair{std::string{"INT32"}, at::kInt},
        std::pair{std::string{"INT64"}, at::kLong},
        std::pair{std::string{"INT16"}, at::kShort},
        std::pair{std::string{"INT8"}, at::kChar},
        std::pair{std::string{"UINT8"}, at::kByte},
        std::pair{std::string{"BOOL"}, at::kBool}));

INSTANTIATE_TEST_SUITE_P(
    CaseInsensitive, DatatypeToScalarCase,
    ::testing::Values(
        std::pair{std::string{"fp32"}, at::kFloat},
        std::pair{std::string{"Fp64"}, at::kDouble},
        std::pair{std::string{"InT16"}, at::kShort},
        std::pair{std::string{"bF16"}, at::kBFloat16},
        std::pair{std::string{"Bool"}, at::kBool}));

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

TEST(DatatypeUtils, ScalarTypeToString)
{
  EXPECT_EQ(starpu_server::scalar_type_to_string(at::kShort), "INT16");
  EXPECT_EQ(
      starpu_server::scalar_type_to_string(static_cast<at::ScalarType>(-1)),
      "FP32");
}

TEST(DatatypeUtils, ScalarToDatatypeAllEnumValues)
{
  for (int i = static_cast<int>(at::ScalarType::Undefined);
       i < static_cast<int>(at::ScalarType::NumOptions); ++i) {
    const auto type = static_cast<at::ScalarType>(i);
    EXPECT_NO_THROW({
      auto name = starpu_server::scalar_type_to_datatype(type);
      auto size = starpu_server::element_size(type);
      (void)name;
      (void)size;
    });
  }
}
