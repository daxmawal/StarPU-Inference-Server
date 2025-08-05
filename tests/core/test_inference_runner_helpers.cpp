#include <gtest/gtest.h>
#include <torch/script.h>

#include <format>
#include <iostream>
#include <vector>

#include "../test_helpers.hpp"
#include "core/inference_runner.hpp"
#include "inference_runner_test_utils.hpp"
#include "utils/exceptions.hpp"

TEST(InferenceRunnerHelpers, RunReferenceInferenceTensor)
{
  auto m = starpu_server::make_mul_two_model();
  std::vector<torch::Tensor> inputs{torch::ones({2})};
  auto outputs = starpu_server::run_reference_inference(m, inputs);
  ASSERT_EQ(outputs.size(), 1u);
  EXPECT_TRUE(torch::allclose(outputs[0], inputs[0] * 2));
}

TEST(InferenceRunnerHelpers, RunReferenceInferenceTuple)
{
  auto m = starpu_server::make_tuple_model();
  std::vector<torch::Tensor> inputs{torch::ones({2})};
  auto outputs = starpu_server::run_reference_inference(m, inputs);
  ASSERT_EQ(outputs.size(), 2u);
  EXPECT_TRUE(torch::allclose(outputs[0], inputs[0]));
  EXPECT_TRUE(torch::allclose(outputs[1], inputs[0] + 1));
}

TEST(InferenceRunnerHelpers, RunReferenceInferenceTensorList)
{
  auto m = starpu_server::make_tensor_list_model();
  std::vector<torch::Tensor> inputs{torch::ones({2})};
  auto outputs = starpu_server::run_reference_inference(m, inputs);
  ASSERT_EQ(outputs.size(), 2u);
  EXPECT_TRUE(torch::allclose(outputs[0], inputs[0]));
  EXPECT_TRUE(torch::allclose(outputs[1], inputs[0] + 1));
}

TEST(InferenceRunnerHelpers, RunReferenceInferenceUnsupportedOutput)
{
  auto m = starpu_server::make_constant_model();
  std::vector<torch::Tensor> inputs{torch::ones({1})};
  EXPECT_THROW(
      starpu_server::run_reference_inference(m, inputs),
      starpu_server::UnsupportedModelOutputTypeException);
}

TEST(InferenceRunnerHelpers, LoadModelAndReferenceOutputError)
{
  starpu_server::RuntimeConfig opts;
  opts.model_path = "nonexistent_model.pt";
  opts.input_shapes = {{1}};
  opts.input_types = {torch::kFloat32};
  starpu_server::CaptureStream capture{std::cerr};
  auto [cpu_model, gpu_models, refs] = load_model_and_reference_output(opts);
  auto err = capture.str();
  EXPECT_TRUE(gpu_models.empty());
  EXPECT_TRUE(refs.empty());
  EXPECT_NE(
      err.find("Failed to load model or run reference inference"),
      std::string::npos);
}

TEST(InferenceRunnerHelpers, LoadModelAndReferenceOutputMissingFile)
{
  starpu_server::RuntimeConfig opts;
  opts.model_path = "nonexistent_model.pt";
  opts.input_shapes = {{1}};
  opts.input_types = {at::kFloat};
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