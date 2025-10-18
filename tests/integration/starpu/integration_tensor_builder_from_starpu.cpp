#include <c10/util/Exception.h>
#include <gtest/gtest.h>

#include "core/inference_params.hpp"
#include "core/tensor_builder.hpp"
#include "test_helpers.hpp"
#include "utils/exceptions.hpp"

using starpu_server::StarpuBufferPtr;

TEST(TensorBuilderFromStarPU_Integration, BuildsTensors)
{
  constexpr float kV0 = 1.0F;
  constexpr float kV1 = 2.0F;
  constexpr float kV2 = 3.0F;
  std::array<float, 3> input0{kV0, kV1, kV2};
  std::array<int32_t, 4> input1{1, 2, 3, 4};
  std::array<starpu_variable_interface, 2> buffers_raw{};
  buffers_raw[0] = starpu_server::make_variable_interface(input0.data());
  buffers_raw[1] = starpu_server::make_variable_interface(input1.data());
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
