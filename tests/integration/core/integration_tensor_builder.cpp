#include <gtest/gtest.h>

#include <array>
#include <vector>

#define private public
#include "core/tensor_builder.hpp"
#undef private

TEST(TensorBuilder_Integration, StarPUStyleRoundTrip)
{
  std::array<float, 4> input{1.f, 2.f, 3.f, 4.f};
  std::array<float, 4> output{};

  starpu_variable_interface in_v;
  in_v.ptr = reinterpret_cast<uintptr_t>(input.data());
  std::vector<void*> buffers = {&in_v};

  starpu_server::InferenceParams params;
  params.num_inputs = 1;
  params.layout.input_types = {at::kFloat};
  params.layout.num_dims = {2};
  params.layout.dims = {{2, 2}};
  params.limits.max_inputs = starpu_server::InferLimits::MaxInputs;
  params.limits.max_dims = starpu_server::InferLimits::MaxDims;
  params.limits.max_models_gpu = starpu_server::InferLimits::MaxModelsGPU;

  auto tensors = starpu_server::TensorBuilder::from_starpu_buffers(
      &params, buffers, torch::kCPU);
  ASSERT_EQ(tensors.size(), 1U);
  auto t = tensors[0];

  starpu_server::TensorBuilder::copy_output_to_buffer(
      t, output.data(), t.numel(), t.scalar_type());

  for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(output[i], input[i]);
}
