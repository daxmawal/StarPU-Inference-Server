#include <gtest/gtest.h>
#include <torch/script.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <vector>

#include "core/inference_runner.hpp"
#include "test_helpers.hpp"
#include "test_inference_runner.hpp"
#include "utils/exceptions.hpp"

namespace {

struct InferenceRunnerParam {
  std::string name;
  std::function<torch::jit::Module()> make_module;
  std::size_t expected_outputs;
  std::vector<std::function<void(
      const std::vector<torch::Tensor>&, const std::vector<torch::Tensor>&)>>
      assertions;
};

}  // namespace

class InferenceRunnerHelpersTest
    : public ::testing::TestWithParam<InferenceRunnerParam> {};

TEST_P(InferenceRunnerHelpersTest, RunReferenceInference)
{
  const auto& param = GetParam();
  auto model = param.make_module();
  std::vector<torch::Tensor> inputs{torch::ones({2})};
  auto outputs = starpu_server::run_reference_inference(model, inputs);

  ASSERT_EQ(outputs.size(), param.expected_outputs);

  for (const auto& assertion : param.assertions) {
    assertion(inputs, outputs);
  }
}

INSTANTIATE_TEST_SUITE_P(
    RunReferenceInference, InferenceRunnerHelpersTest,
    ::testing::Values(
        InferenceRunnerParam{
            "Tensor",
            [] { return starpu_server::make_mul_two_model(); },
            1U,
            {[](const auto& inputs, const auto& outputs) {
              EXPECT_TRUE(torch::allclose(outputs[0], inputs[0] * 2));
            }}},
        InferenceRunnerParam{
            "Tuple",
            [] { return starpu_server::make_tuple_model(); },
            2U,
            {[](const auto& inputs, const auto& outputs) {
               EXPECT_TRUE(torch::allclose(outputs[0], inputs[0]));
             },
             [](const auto& inputs, const auto& outputs) {
               EXPECT_TRUE(torch::allclose(outputs[1], inputs[0] + 1));
             }}},
        InferenceRunnerParam{
            "TensorList",
            [] { return starpu_server::make_tensor_list_model(); },
            2U,
            {[](const auto& inputs, const auto& outputs) {
               EXPECT_TRUE(torch::allclose(outputs[0], inputs[0]));
             },
             [](const auto& inputs, const auto& outputs) {
               EXPECT_TRUE(torch::allclose(outputs[1], inputs[0] + 1));
             }}}),
    [](const ::testing::TestParamInfo<InferenceRunnerParam>& info) {
      return info.param.name;
    });
