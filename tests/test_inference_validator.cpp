#include <gtest/gtest.h>
#include <torch/script.h>

#include "core/inference_runner.hpp"
#include "utils/inference_validator.hpp"
#include "utils/logger.hpp"

using namespace starpu_server;

static auto
make_add_one_model() -> torch::jit::script::Module
{
  torch::jit::script::Module m{"m"};
  m.define(R"JIT(
      def forward(self, x):
          return x + 1
  )JIT");
  return m;
}

TEST(InferenceValidator, SuccessfulValidation)
{
  auto model = make_add_one_model();

  InferenceResult result;
  result.inputs = {torch::tensor({1, 2, 3})};
  result.results = {torch::tensor({2, 3, 4})};
  result.job_id = 42;
  result.executed_on = DeviceType::CPU;

  EXPECT_TRUE(validate_inference_result(result, model, VerbosityLevel::Silent));
}

TEST(InferenceValidator, FailsOnMismatch)
{
  auto model = make_add_one_model();

  InferenceResult result;
  result.inputs = {torch::tensor({1, 2, 3})};
  result.results = {torch::tensor({1, 2, 3})};
  result.job_id = 43;
  result.executed_on = DeviceType::CPU;

  EXPECT_FALSE(
      validate_inference_result(result, model, VerbosityLevel::Silent));
}