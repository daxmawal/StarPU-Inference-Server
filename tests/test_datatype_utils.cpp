#include <ATen/core/ScalarType.h>
#include <gtest/gtest.h>

#include "utils/datatype_utils.hpp"

using namespace starpu_server;

TEST(DatatypeUtils, ScalarToDatatypeKnownTypes)
{
  EXPECT_EQ(scalar_type_to_datatype(at::kFloat), "FP32");
  EXPECT_EQ(scalar_type_to_datatype(at::kDouble), "FP64");
  EXPECT_EQ(scalar_type_to_datatype(at::kHalf), "FP16");
  EXPECT_EQ(scalar_type_to_datatype(at::kBFloat16), "BF16");
  EXPECT_EQ(scalar_type_to_datatype(at::kInt), "INT32");
  EXPECT_EQ(scalar_type_to_datatype(at::kLong), "INT64");
  EXPECT_EQ(scalar_type_to_datatype(at::kShort), "INT16");
  EXPECT_EQ(scalar_type_to_datatype(at::kChar), "INT8");
  EXPECT_EQ(scalar_type_to_datatype(at::kByte), "UINT8");
  EXPECT_EQ(scalar_type_to_datatype(at::kBool), "BOOL");
}

TEST(DatatypeUtils, ScalarToDatatypeFallbackTypes)
{
  EXPECT_EQ(scalar_type_to_datatype(static_cast<at::ScalarType>(-1)), "FP32");
  EXPECT_EQ(scalar_type_to_datatype(at::kComplexFloat), "FP32");
  EXPECT_EQ(scalar_type_to_datatype(at::kComplexDouble), "FP32");
  EXPECT_EQ(scalar_type_to_datatype(at::kQInt8), "FP32");
  EXPECT_EQ(scalar_type_to_datatype(at::kQUInt8), "FP32");
}

TEST(DatatypeUtils, ElementSizeKnownTypes)
{
  EXPECT_EQ(element_size(at::kFloat), sizeof(float));
  EXPECT_EQ(element_size(at::kDouble), sizeof(double));
  EXPECT_EQ(element_size(at::kHalf), 2U);
  EXPECT_EQ(element_size(at::kBFloat16), 2U);
  EXPECT_EQ(element_size(at::kInt), sizeof(int32_t));
  EXPECT_EQ(element_size(at::kLong), sizeof(int64_t));
  EXPECT_EQ(element_size(at::kShort), sizeof(int16_t));
  EXPECT_EQ(element_size(at::kChar), sizeof(int8_t));
  EXPECT_EQ(element_size(at::kByte), sizeof(uint8_t));
  EXPECT_EQ(element_size(at::kBool), sizeof(bool));
}

TEST(DatatypeUtils, ElementSizeFallbackTypes)
{
  EXPECT_EQ(element_size(static_cast<at::ScalarType>(-1)), sizeof(float));
  EXPECT_EQ(element_size(at::kComplexFloat), sizeof(float));
  EXPECT_EQ(element_size(at::kComplexDouble), sizeof(float));
  EXPECT_EQ(element_size(at::kQInt8), sizeof(float));
  EXPECT_EQ(element_size(at::kQUInt8), sizeof(float));
}

TEST(DatatypeUtils, DatatypeToScalarValidExactMatch)
{
  EXPECT_EQ(datatype_to_scalar_type("FP32"), at::kFloat);
  EXPECT_EQ(datatype_to_scalar_type("FP64"), at::kDouble);
  EXPECT_EQ(datatype_to_scalar_type("FP16"), at::kHalf);
  EXPECT_EQ(datatype_to_scalar_type("BF16"), at::kBFloat16);
  EXPECT_EQ(datatype_to_scalar_type("INT32"), at::kInt);
  EXPECT_EQ(datatype_to_scalar_type("INT64"), at::kLong);
  EXPECT_EQ(datatype_to_scalar_type("INT16"), at::kShort);
  EXPECT_EQ(datatype_to_scalar_type("INT8"), at::kChar);
  EXPECT_EQ(datatype_to_scalar_type("UINT8"), at::kByte);
  EXPECT_EQ(datatype_to_scalar_type("BOOL"), at::kBool);
}

TEST(DatatypeUtils, DatatypeToScalarCaseInsensitive)
{
  EXPECT_EQ(datatype_to_scalar_type("fp32"), at::kFloat);
  EXPECT_EQ(datatype_to_scalar_type("Fp64"), at::kDouble);
  EXPECT_EQ(datatype_to_scalar_type("InT16"), at::kShort);
  EXPECT_EQ(datatype_to_scalar_type("bF16"), at::kBFloat16);
  EXPECT_EQ(datatype_to_scalar_type("Bool"), at::kBool);
}

TEST(DatatypeUtils, DatatypeToScalarInvalid)
{
  EXPECT_THROW(datatype_to_scalar_type("BADTYPE"), std::invalid_argument);
  EXPECT_THROW(datatype_to_scalar_type(""), std::invalid_argument);
  EXPECT_THROW(datatype_to_scalar_type("#FP32"), std::invalid_argument);
  EXPECT_THROW(datatype_to_scalar_type("UNKNOWN"), std::invalid_argument);
  EXPECT_THROW(datatype_to_scalar_type("notatype"), std::invalid_argument);
  EXPECT_THROW(datatype_to_scalar_type(" "), std::invalid_argument);
  EXPECT_THROW(datatype_to_scalar_type("!"), std::invalid_argument);
  EXPECT_THROW(datatype_to_scalar_type("123"), std::invalid_argument);
  EXPECT_THROW(datatype_to_scalar_type("FP32extra"), std::invalid_argument);
  EXPECT_THROW(
      datatype_to_scalar_type(std::string_view()), std::invalid_argument);
  EXPECT_THROW(
      datatype_to_scalar_type("𝔽ℙ𝟛𝟚"), std::invalid_argument);  // Unicode
  EXPECT_THROW(
      datatype_to_scalar_type("fp32\n"), std::invalid_argument);  // contrôle
}

TEST(DatatypeUtils, ScalarTypeToStringWrapper)
{
  EXPECT_EQ(scalar_type_to_string(at::kShort), "INT16");
  EXPECT_EQ(scalar_type_to_string(static_cast<at::ScalarType>(-1)), "FP32");
}

TEST(DatatypeUtils, ScalarToDatatypeAllEnumValues)
{
  // Exercice de couverture totale des types énumérés
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
