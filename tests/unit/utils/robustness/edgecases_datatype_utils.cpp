#include <ATen/core/ScalarType.h>
#include <gtest/gtest.h>

#include <stdexcept>
#include <utility>

#include "utils/datatype_utils.hpp"

class ScalarToDatatype_Fallback
    : public ::testing::TestWithParam<std::pair<at::ScalarType, std::string>> {
};

TEST_P(ScalarToDatatype_Fallback, MapsToFP32Fallback)
{
  const auto& [type, expected] = GetParam();
  EXPECT_EQ(starpu_server::scalar_type_to_datatype(type), expected);
}
INSTANTIATE_TEST_SUITE_P(
    FallbackTypes, ScalarToDatatype_Fallback,
    ::testing::Values(
        std::pair{static_cast<at::ScalarType>(-1), std::string{"FP32"}},
        std::pair{at::kComplexFloat, std::string{"FP32"}},
        std::pair{at::kComplexDouble, std::string{"FP32"}},
        std::pair{at::kQInt8, std::string{"FP32"}},
        std::pair{at::kQUInt8, std::string{"FP32"}}));

class ElementSize_Fallback
    : public ::testing::TestWithParam<std::pair<at::ScalarType, size_t>> {};
TEST_P(ElementSize_Fallback, ReturnsFloatSizeAsFallback)
{
  const auto& [type, expected_size] = GetParam();
  EXPECT_EQ(starpu_server::element_size(type), expected_size);
}
INSTANTIATE_TEST_SUITE_P(
    FallbackTypes, ElementSize_Fallback,
    ::testing::Values(
        std::pair{static_cast<at::ScalarType>(-1), sizeof(float)},
        std::pair{at::kComplexFloat, sizeof(float)},
        std::pair{at::kComplexDouble, sizeof(float)},
        std::pair{at::kQInt8, sizeof(float)},
        std::pair{at::kQUInt8, sizeof(float)}));

// --- Entrées invalides (strings non reconnues)
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

// --- Balayage complet des valeurs enum pour vérifier robustesse (no-throw)
TEST(DatatypeUtils_Robustesse, ScalarToDatatype_AllEnumValues_NoThrow)
{
  using Enum = at::ScalarType;
  using U = std::underlying_type_t<Enum>;

  const int first = static_cast<int>(std::to_underlying(Enum::Undefined));
  const int last = static_cast<int>(std::to_underlying(Enum::NumOptions));

  for (int i = first; i < last; ++i) {
    const auto type = static_cast<Enum>(static_cast<U>(i));
    EXPECT_NO_THROW({
      auto name = starpu_server::scalar_type_to_datatype(type);
      auto size = starpu_server::element_size(type);
      (void)name;
      (void)size;
    });
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
