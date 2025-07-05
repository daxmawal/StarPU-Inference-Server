#include <gtest/gtest.h>

#include "utils/datatype_utils.hpp"

using namespace starpu_server;

TEST(DatatypeUtils, ScalarToDatatype)
{
  EXPECT_EQ(scalar_type_to_datatype(at::kFloat), "FP32");
  EXPECT_EQ(scalar_type_to_datatype(at::kInt), "INT32");
}

TEST(DatatypeUtils, DatatypeToScalar)
{
  EXPECT_EQ(datatype_to_scalar_type("FP64"), at::kDouble);
  EXPECT_EQ(datatype_to_scalar_type("INT8"), at::kChar);
  EXPECT_THROW(datatype_to_scalar_type("BADTYPE"), std::invalid_argument);
}

TEST(DatatypeUtils, ElementSize)
{
  EXPECT_EQ(element_size(at::kDouble), sizeof(double));
  EXPECT_EQ(element_size(at::kByte), sizeof(uint8_t));
}