#include <c10/util/Exception.h>
#include <gtest/gtest.h>

#include "../test_helpers.hpp"
#include "core/inference_params.hpp"
#include "core/tensor_builder.hpp"
#include "utils/exceptions.hpp"

TEST(TensorBuilderFromStarPU, BuildsTensors)
{
  float input0[3] = {1.0f, 2.0f, 3.0f};
  int32_t input1[4] = {1, 2, 3, 4};
  starpu_variable_interface buffers_raw[2];
  buffers_raw[0].ptr = reinterpret_cast<uintptr_t>(input0);
  buffers_raw[1].ptr = reinterpret_cast<uintptr_t>(input1);

  auto params = starpu_server::make_params_for_inputs(
      {{3}, {2, 2}}, {at::kFloat, at::kInt});

  std::array<void*, 2> buffers{&buffers_raw[0], &buffers_raw[1]};
  auto tensors = starpu_server::TensorBuilder::from_starpu_buffers(
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
  const size_t too_many = starpu_server::InferLimits::MaxInputs + 1;
  std::vector<std::vector<int64_t>> shapes(too_many, {1});
  std::vector<at::ScalarType> dtypes(too_many, at::kFloat);
  auto params = starpu_server::make_params_for_inputs(shapes, dtypes);
  std::vector<void*> dummy(params.num_inputs, nullptr);

  EXPECT_THROW(
      starpu_server::TensorBuilder::from_starpu_buffers(
          &params, dummy, torch::Device(torch::kCPU)),
      starpu_server::InferenceExecutionException);
}

TEST(TensorBuilderFromStarPU, NegativeNumDimsThrows)
{
  float input0[1] = {0.0f};
  starpu_variable_interface buf;
  buf.ptr = reinterpret_cast<uintptr_t>(input0);

  auto params = starpu_server::make_params_for_inputs({{1}}, {at::kFloat});
  params.layout.num_dims[0] = -1;

  std::array<void*, 1> buffers{&buf};

  EXPECT_THROW(
      starpu_server::TensorBuilder::from_starpu_buffers(
          &params, buffers, torch::Device(torch::kCPU)),
      c10::Error);
}
