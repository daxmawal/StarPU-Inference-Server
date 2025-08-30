#include <c10/util/Exception.h>
#include <gtest/gtest.h>

#include <bit>

#include "core/inference_params.hpp"
#include "core/tensor_builder.hpp"
#include "test_helpers.hpp"
#include "utils/exceptions.hpp"

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
  std::array<float, 1> input0{0.0F};
  starpu_variable_interface buf{};
  buf.ptr = std::bit_cast<uintptr_t>(input0.data());
  auto params = starpu_server::make_params_for_inputs({{1}}, {at::kFloat});
  params.layout.num_dims[0] = -1;
  std::array<void*, 1> buffers{&buf};
  EXPECT_THROW(
      starpu_server::TensorBuilder::from_starpu_buffers(
          &params, buffers, torch::Device(torch::kCPU)),
      c10::Error);
}
