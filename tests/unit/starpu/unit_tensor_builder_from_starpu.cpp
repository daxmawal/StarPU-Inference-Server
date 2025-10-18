#include <c10/util/Exception.h>
#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>

#include "core/inference_params.hpp"
#include "core/tensor_builder.hpp"
#include "test_constants.hpp"
#include "test_helpers.hpp"
#include "utils/exceptions.hpp"

using starpu_server::StarpuBufferPtr;

TEST(TensorBuilderFromStarPU, BuildsTensors)
{
  std::array<float, 3> input0{
      starpu_server::test_constants::kF1, starpu_server::test_constants::kF2,
      starpu_server::test_constants::kF3};
  std::array<int32_t, 4> input1{1, 2, 3, 4};
  std::array<starpu_variable_interface, 2> buffers_raw{};
  buffers_raw[0].ptr = std::bit_cast<uintptr_t>(input0.data());
  buffers_raw[1].ptr = std::bit_cast<uintptr_t>(input1.data());

  auto params = starpu_server::make_params_for_inputs(
      {{3}, {2, 2}}, {at::kFloat, at::kInt});

  std::array<StarpuBufferPtr, 2> buffers{
      buffers_raw.data(), buffers_raw.data() + 1};
  auto tensors = starpu_server::TensorBuilder::from_starpu_buffers(
      &params, buffers, torch::Device(torch::kCPU));

  ASSERT_EQ(tensors.size(), 2U);
  EXPECT_EQ(tensors[0].sizes(), (torch::IntArrayRef{3}));
  EXPECT_EQ(tensors[0].dtype(), torch::kFloat);
  EXPECT_EQ(tensors[0].data_ptr<float>(), input0.data());
  EXPECT_EQ(tensors[1].sizes(), (torch::IntArrayRef{2, 2}));
  EXPECT_EQ(tensors[1].dtype(), torch::kInt);
  EXPECT_EQ(tensors[1].data_ptr<int32_t>(), input1.data());
}

TEST(TensorBuilderFromStarPU, MakeParamsForInputsSetsBasicFields)
{
  auto params = starpu_server::make_params_for_inputs(
      {{3}, {2, 2}}, {at::kFloat, at::kInt});

  EXPECT_EQ(params.num_inputs, 2U);
  EXPECT_EQ(params.layout.num_dims[0], 1);
  EXPECT_EQ(params.layout.num_dims[1], 2);
  EXPECT_EQ(params.layout.input_types[0], at::kFloat);
  EXPECT_EQ(params.layout.input_types[1], at::kInt);
}
