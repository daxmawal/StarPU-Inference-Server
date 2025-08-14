#include <c10/util/Exception.h>
#include <gtest/gtest.h>

#include "core/inference_params.hpp"
#include "core/tensor_builder.hpp"
#include "test_helpers.hpp"
#include "utils/exceptions.hpp"

/*
TEST(TensorBuilder_Unit, MakeParamsForInputsSetsBasicFields)
{
  std::vector<std::vector<int64_t>> shapes = {{3}, {2, 2}};
  std::vector<at::ScalarType> dtypes = {at::kFloat, at::kInt};

  auto params = starpu_server::make_params_for_inputs(shapes, dtypes);

  EXPECT_EQ(params.num_inputs, 2);
  ASSERT_EQ(params.layout.num_dims.size(), 2U);
  EXPECT_EQ(params.layout.num_dims[0], 1);
  EXPECT_EQ(params.layout.num_dims[1], 2);

  ASSERT_EQ(params.layout.input_types.size(), 2U);
  EXPECT_EQ(params.layout.input_types[0], at::kFloat);
  EXPECT_EQ(params.layout.input_types[1], at::kInt);
}
*/
