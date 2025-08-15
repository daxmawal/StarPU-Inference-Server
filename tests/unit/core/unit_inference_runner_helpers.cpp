#include <gtest/gtest.h>
#include <torch/script.h>

#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <vector>

#include "core/inference_runner.hpp"
#include "test_helpers.hpp"
#include "test_inference_runner.hpp"
#include "utils/exceptions.hpp"

TEST(InferenceRunnerHelpers, RunReferenceInferenceTensor)
{
  auto model = starpu_server::make_mul_two_model();
  std::vector<torch::Tensor> inputs{torch::ones({2})};
  auto outputs = starpu_server::run_reference_inference(model, inputs);
  ASSERT_EQ(outputs.size(), 1U);
  EXPECT_TRUE(torch::allclose(outputs[0], inputs[0] * 2));
}

TEST(InferenceRunnerHelpers, RunReferenceInferenceTuple)
{
  auto model = starpu_server::make_tuple_model();
  std::vector<torch::Tensor> inputs{torch::ones({2})};
  auto outputs = starpu_server::run_reference_inference(model, inputs);
  ASSERT_EQ(outputs.size(), 2U);
  EXPECT_TRUE(torch::allclose(outputs[0], inputs[0]));
  EXPECT_TRUE(torch::allclose(outputs[1], inputs[0] + 1));
}

TEST(InferenceRunnerHelpers, RunReferenceInferenceTensorList)
{
  auto model = starpu_server::make_tensor_list_model();
  std::vector<torch::Tensor> inputs{torch::ones({2})};
  auto outputs = starpu_server::run_reference_inference(model, inputs);
  ASSERT_EQ(outputs.size(), 2U);
  EXPECT_TRUE(torch::allclose(outputs[0], inputs[0]));
  EXPECT_TRUE(torch::allclose(outputs[1], inputs[0] + 1));
}
