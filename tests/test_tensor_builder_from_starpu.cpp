#include <gtest/gtest.h>

#include "core/inference_params.hpp"
#include "core/tensor_builder.hpp"
#include "utils/exceptions.hpp"

using namespace starpu_server;

TEST(TensorBuilderFromStarPU, BuildsTensors)
{
  float input0[3] = {1.0f, 2.0f, 3.0f};
  int32_t input1[4] = {1, 2, 3, 4};
  starpu_variable_interface buffers_raw[2];
  buffers_raw[0].ptr = reinterpret_cast<uintptr_t>(input0);
  buffers_raw[1].ptr = reinterpret_cast<uintptr_t>(input1);

  InferenceParams params{};
  params.num_inputs = 2;
  params.layout.num_dims[0] = 1;
  params.layout.dims[0][0] = 3;
  params.layout.input_types[0] = at::kFloat;
  params.layout.num_dims[1] = 2;
  params.layout.dims[1][0] = 2;
  params.layout.dims[1][1] = 2;
  params.layout.input_types[1] = at::kInt;

  std::array<void*, 2> buffers{&buffers_raw[0], &buffers_raw[1]};
  auto tensors = TensorBuilder::from_starpu_buffers(
      &params, buffers, torch::Device(torch::kCPU));

  ASSERT_EQ(tensors.size(), 2u);
  EXPECT_EQ(tensors[0].sizes(), (torch::IntArrayRef{3}));
  EXPECT_EQ(tensors[0].dtype(), torch::kFloat);
  EXPECT_EQ(tensors[0].data_ptr<float>(), input0);

  EXPECT_EQ(tensors[1].sizes(), (torch::IntArrayRef{2, 2}));
  EXPECT_EQ(tensors[1].dtype(), torch::kInt);
  EXPECT_EQ(tensors[1].data_ptr<int32_t>(), input1);
}

TEST(TensorBuilderFromStarPU, TooManyInputsThrows)
{
  InferenceParams params{};
  params.num_inputs = InferLimits::MaxInputs + 1;
  std::vector<void*> dummy(params.num_inputs, nullptr);

  EXPECT_THROW(
      TensorBuilder::from_starpu_buffers(
          &params, dummy, torch::Device(torch::kCPU)),
      InferenceExecutionException);
}