#include <gtest/gtest.h>
#include <torch/script.h>

#include <string>

#include "core/inference_runner.hpp"
#include "utils/exceptions.hpp"
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

static auto
make_tuple_non_tensor_model() -> torch::jit::script::Module
{
  torch::jit::script::Module m{"m"};
  m.define(R"JIT(
      def forward(self, x):
          return (x, 1)
  )JIT");
  return m;
}

static auto
make_string_model() -> torch::jit::script::Module
{
  torch::jit::script::Module m{"m"};
  m.define(R"JIT(
      def forward(self, x):
          return "hello"
  )JIT");
  return m;
}

static auto
make_error_model() -> torch::jit::script::Module
{
  torch::jit::script::Module m{"m"};
  m.define(R"JIT(
      def forward(self, x):
          return torch.mm(x, x)
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

  testing::internal::CaptureStderr();
  EXPECT_FALSE(
      validate_inference_result(result, model, VerbosityLevel::Silent));
  std::string logs = testing::internal::GetCapturedStderr();
  EXPECT_NE(logs.find("Mismatch on output"), std::string::npos);
}

TEST(InferenceValidator, ThrowsOnUnknownDevice)
{
  auto model = make_add_one_model();

  InferenceResult result;
  result.inputs = {torch::tensor({1, 2, 3})};
  result.results = {torch::tensor({2, 3, 4})};
  result.job_id = 44;
  result.executed_on = DeviceType::Unknown;

  EXPECT_THROW(
      validate_inference_result(result, model, VerbosityLevel::Silent),
      InferenceExecutionException);
}

TEST(InferenceValidator, ThrowsOnInvalidDeviceValue)
{
  auto model = make_add_one_model();

  InferenceResult result;
  result.inputs = {torch::tensor({1, 2, 3})};
  result.results = {torch::tensor({2, 3, 4})};
  result.job_id = 144;
  result.executed_on = static_cast<DeviceType>(255);

  EXPECT_THROW(
      validate_inference_result(result, model, VerbosityLevel::Silent),
      InferenceExecutionException);
}

TEST(InferenceValidator, OutputCountMismatch)
{
  auto model = make_add_one_model();

  InferenceResult result;
  result.inputs = {torch::tensor({1, 2, 3})};
  result.results = {torch::tensor({2, 3, 4}), torch::tensor({2, 3, 4})};
  result.job_id = 45;
  result.executed_on = DeviceType::CPU;

  EXPECT_FALSE(
      validate_inference_result(result, model, VerbosityLevel::Silent));
}

TEST(InferenceValidator, ThrowsOnNonTensorTupleElement)
{
  auto model = make_tuple_non_tensor_model();

  InferenceResult result;
  result.inputs = {torch::tensor({1})};
  result.results = {torch::tensor({1})};
  result.job_id = 46;
  result.executed_on = DeviceType::CPU;

  EXPECT_THROW(
      validate_inference_result(result, model, VerbosityLevel::Silent),
      InferenceExecutionException);
}

TEST(InferenceValidator, ThrowsOnUnsupportedOutputType)
{
  auto model = make_string_model();

  InferenceResult result;
  result.inputs = {torch::tensor({1})};
  result.job_id = 47;
  result.executed_on = DeviceType::CPU;

  EXPECT_THROW(
      validate_inference_result(result, model, VerbosityLevel::Silent),
      InferenceExecutionException);
}

/* TODO: There is a core dump on this test
TEST(InferenceValidator, ReturnsFalseOnC10Error)
{
  auto model = make_error_model();

  InferenceResult result;
  result.inputs = {torch::tensor({1, 2, 3})};
  result.results = {torch::tensor({2, 3, 4})};
  result.job_id = 145;
  result.executed_on = DeviceType::CPU;

  testing::internal::CaptureStderr();
  EXPECT_FALSE(
      validate_inference_result(result, model, VerbosityLevel::Silent));
  std::string logs = testing::internal::GetCapturedStderr();
  EXPECT_NE(logs.find("C10 error"), std::string::npos);
}
*/