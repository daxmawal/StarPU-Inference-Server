#include <ATen/core/ScalarType.h>
#include <gtest/gtest.h>

#include "utils/datatype_utils.hpp"

using namespace starpu_server;

TEST(DatatypeUtils, ScalarToDatatypeKnownTypes)
{
  const std::vector<std::pair<at::ScalarType, std::string>> mapping = {
      {at::kFloat, "FP32"},    {at::kDouble, "FP64"}, {at::kHalf, "FP16"},
      {at::kBFloat16, "BF16"}, {at::kInt, "INT32"},   {at::kLong, "INT64"},
      {at::kShort, "INT16"},   {at::kChar, "INT8"},   {at::kByte, "UINT8"},
      {at::kBool, "BOOL"}};

  for (const auto& [type, expected] : mapping) {
    EXPECT_EQ(scalar_type_to_datatype(type), expected);
  }
}

TEST(DatatypeUtils, ScalarToDatatypeFallbackTypes)
{
  const std::vector<at::ScalarType> fallback_types = {
      static_cast<at::ScalarType>(-1), at::kComplexFloat, at::kComplexDouble,
      at::kQInt8, at::kQUInt8};

  for (auto type : fallback_types) {
    EXPECT_EQ(scalar_type_to_datatype(type), "FP32");
  }
}

TEST(DatatypeUtils, ElementSizeKnownTypes)
{
  const std::vector<std::pair<at::ScalarType, size_t>> sizes = {
      {at::kFloat, sizeof(float)},
      {at::kDouble, sizeof(double)},
      {at::kHalf, 2U},
      {at::kBFloat16, 2U},
      {at::kInt, sizeof(int32_t)},
      {at::kLong, sizeof(int64_t)},
      {at::kShort, sizeof(int16_t)},
      {at::kChar, sizeof(int8_t)},
      {at::kByte, sizeof(uint8_t)},
      {at::kBool, sizeof(bool)}};

  for (const auto& [type, expected_size] : sizes) {
    EXPECT_EQ(element_size(type), expected_size);
  }
}

TEST(DatatypeUtils, ElementSizeFallbackTypes)
{
  const std::vector<at::ScalarType> fallback_types = {
      static_cast<at::ScalarType>(-1), at::kComplexFloat, at::kComplexDouble,
      at::kQInt8, at::kQUInt8};

  for (auto type : fallback_types) {
    EXPECT_EQ(element_size(type), sizeof(float));
  }
}

TEST(DatatypeUtils, DatatypeToScalarValidExactMatch)
{
  const std::vector<std::pair<std::string, at::ScalarType>> mapping = {
      {"FP32", at::kFloat},    {"FP64", at::kDouble}, {"FP16", at::kHalf},
      {"BF16", at::kBFloat16}, {"INT32", at::kInt},   {"INT64", at::kLong},
      {"INT16", at::kShort},   {"INT8", at::kChar},   {"UINT8", at::kByte},
      {"BOOL", at::kBool}};

  for (const auto& [str, expected_type] : mapping) {
    EXPECT_EQ(datatype_to_scalar_type(str), expected_type);
  }
}

TEST(DatatypeUtils, DatatypeToScalarCaseInsensitive)
{
  const std::vector<std::pair<std::string, at::ScalarType>> mapping = {
      {"fp32", at::kFloat},
      {"Fp64", at::kDouble},
      {"InT16", at::kShort},
      {"bF16", at::kBFloat16},
      {"Bool", at::kBool}};

  for (const auto& [str, expected_type] : mapping) {
    EXPECT_EQ(datatype_to_scalar_type(str), expected_type);
  }
}

class InvalidDatatypeTest : public ::testing::TestWithParam<std::string_view> {
};

TEST_P(InvalidDatatypeTest, ThrowsInvalidArgument)
{
  EXPECT_THROW(datatype_to_scalar_type(GetParam()), std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(
    DatatypeUtils, InvalidDatatypeTest,
    ::testing::Values(
        "BADTYPE", "", "#FP32", "UNKNOWN", "notatype", " ", "!", "123",
        "FP32extra", std::string_view(), "𝔽ℙ𝟛𝟚", "fp32\n"));

TEST(DatatypeUtils, ScalarTypeToString)
{
  EXPECT_EQ(scalar_type_to_string(at::kShort), "INT16");
  EXPECT_EQ(scalar_type_to_string(static_cast<at::ScalarType>(-1)), "FP32");
}

TEST(DatatypeUtils, ScalarToDatatypeAllEnumValues)
{
  for (int i = static_cast<int>(at::ScalarType::Undefined);
       i < static_cast<int>(at::ScalarType::NumOptions); ++i) {
    const auto type = static_cast<at::ScalarType>(i);
    EXPECT_NO_THROW({
      auto name = scalar_type_to_datatype(type);
      auto size = element_size(type);
      (void)name;
      (void)size;
    });
  }
}
