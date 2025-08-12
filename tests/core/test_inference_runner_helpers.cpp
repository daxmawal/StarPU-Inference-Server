#include <gtest/gtest.h>
#include <torch/script.h>

#include <format>
#include <iostream>
#include <vector>

#include "core/inference_runner.hpp"
#include "inference_runner_test_utils.hpp"
#include "test_helpers.hpp"
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

TEST(InferenceRunnerHelpers, RunReferenceInferenceUnsupportedOutput)
{
  auto model = starpu_server::make_constant_model();
  std::vector<torch::Tensor> inputs{torch::ones({1})};
  EXPECT_THROW(
      starpu_server::run_reference_inference(model, inputs),
      starpu_server::UnsupportedModelOutputTypeException);
}

class LoadModelAndReferenceOutputError
    : public ::testing::TestWithParam<at::ScalarType> {};

TEST_P(LoadModelAndReferenceOutputError, MissingFile)
{
  starpu_server::RuntimeConfig opts;
  opts.model_path = "nonexistent_model.pt";
  opts.input_shapes = {{1}};
  opts.input_types = {GetParam()};
  starpu_server::CaptureStream capture{std::cerr};
  auto [cpu_model, gpu_models, refs] =
      starpu_server::load_model_and_reference_output(opts);
  auto err = capture.str();
  EXPECT_TRUE(gpu_models.empty());
  EXPECT_TRUE(refs.empty());
  EXPECT_NE(
      err.find("Failed to load model or run reference inference"),
      std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(
    InferenceRunnerHelpers, LoadModelAndReferenceOutputError,
    ::testing::Values(torch::kFloat32, at::kFloat));
