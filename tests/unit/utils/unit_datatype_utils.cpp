#include <ATen/core/ScalarType.h>
#include <gtest/gtest.h>

#include <unordered_set>
#include <utility>

#include "utils/datatype_utils.hpp"
#include "utils/device_type.hpp"

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

class ScalarToDatatypeUnsupported
    : public ::testing::TestWithParam<at::ScalarType> {};

TEST_P(ScalarToDatatypeUnsupported, ThrowsInvalidArgument)
{
  EXPECT_THROW(
      starpu_server::scalar_type_to_datatype(GetParam()),
      std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(
    UnsupportedTypes, ScalarToDatatypeUnsupported,
    ::testing::Values(
        static_cast<at::ScalarType>(-1), at::kComplexFloat, at::kComplexDouble,
        at::kQInt8, at::kQUInt8));

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

class ElementSizeUnsupported : public ::testing::TestWithParam<at::ScalarType> {
};

TEST_P(ElementSizeUnsupported, ThrowsInvalidArgument)
{
  EXPECT_THROW(starpu_server::element_size(GetParam()), std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(
    UnsupportedTypes, ElementSizeUnsupported,
    ::testing::Values(
        static_cast<at::ScalarType>(-1), at::kComplexFloat, at::kComplexDouble,
        at::kQInt8, at::kQUInt8));

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

TEST(DatatypeUtils, ScalarTypeToString)
{
  EXPECT_EQ(starpu_server::scalar_type_to_string(at::kShort), "INT16");
  EXPECT_THROW(
      starpu_server::scalar_type_to_string(static_cast<at::ScalarType>(-1)),
      std::invalid_argument);
}

TEST(DatatypeUtils, ScalarToDatatypeAllEnumValues)
{
  using Enum = at::ScalarType;
  using U = std::underlying_type_t<Enum>;

  const std::unordered_set<Enum> supported = {
      at::kFloat, at::kDouble, at::kHalf, at::kBFloat16, at::kInt,
      at::kLong,  at::kShort,  at::kChar, at::kByte,     at::kBool};

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
          starpu_server::scalar_type_to_datatype(type), std::invalid_argument);
      EXPECT_THROW(starpu_server::element_size(type), std::invalid_argument);
    }
  }
}

TEST(DeviceTypeTest, ToString)
{
  EXPECT_STREQ(starpu_server::to_string(starpu_server::DeviceType::CPU), "CPU");
  EXPECT_STREQ(
      starpu_server::to_string(starpu_server::DeviceType::CUDA), "CUDA");
  EXPECT_STREQ(
      starpu_server::to_string(starpu_server::DeviceType::Unknown), "Unknown");
  EXPECT_STREQ(
      starpu_server::to_string(static_cast<starpu_server::DeviceType>(42)),
      "InvalidDeviceType");
}
