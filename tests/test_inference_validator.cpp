#include <gtest/gtest.h>
#include <torch/script.h>
#include <torch/torch.h>

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

static auto
make_empty_tuple_model() -> torch::jit::script::Module
{
  torch::jit::script::Module m{"m"};
  m.define(R"JIT(
      def forward(self, x):
          return ()
  )JIT");
  return m;
}

static auto
make_shape_error_model() -> torch::jit::script::Module
{
  torch::jit::script::Module m{"m"};
  m.define(R"JIT(
      def forward(self, x):
          return x.view(-1, 0)
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

TEST(InferenceValidator, EmptyTupleOutput)
{
  auto model = make_empty_tuple_model();

  InferenceResult result;
  result.inputs = {torch::tensor({1})};
  result.results = {};
  result.job_id = 48;
  result.executed_on = DeviceType::CPU;

  EXPECT_TRUE(validate_inference_result(result, model, VerbosityLevel::Silent));
}

TEST(InferenceValidator, SuccessfulValidationCuda)
{
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available";
  }

  auto model = make_add_one_model();
  model.to(torch::kCUDA);

  InferenceResult result;
  result.inputs = {torch::tensor({1, 2, 3}).to(torch::kCUDA)};
  result.results = {torch::tensor({2, 3, 4}).to(torch::kCUDA)};
  result.job_id = 100;
  result.executed_on = DeviceType::CUDA;
  result.device_id = 0;
  result.worker_id = 0;

  EXPECT_TRUE(validate_inference_result(result, model, VerbosityLevel::Silent));
}

TEST(InferenceValidator, FailsOnMismatchCuda)
{
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available";
  }

  auto model = make_add_one_model();
  model.to(torch::kCUDA);

  InferenceResult result;
  result.inputs = {torch::tensor({1, 2, 3}).to(torch::kCUDA)};
  result.results = {torch::tensor({1, 2, 3}).to(torch::kCUDA)};
  result.job_id = 101;
  result.executed_on = DeviceType::CUDA;
  result.device_id = 0;
  result.worker_id = 0;

  testing::internal::CaptureStderr();
  EXPECT_FALSE(
      validate_inference_result(result, model, VerbosityLevel::Silent));
  std::string logs = testing::internal::GetCapturedStderr();
  EXPECT_NE(logs.find("Mismatch on output"), std::string::npos);
}

TEST(InferenceValidator, CudaModelOnCpuInputsThrows)
{
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available";
  }

  auto model = make_add_one_model();
  model.to(torch::kCUDA);

  InferenceResult result;
  result.inputs = {torch::tensor({1, 2, 3})};
  result.results = {torch::tensor({2, 3, 4}).to(torch::kCUDA)};
  result.job_id = 102;
  result.executed_on = DeviceType::CUDA;
  result.device_id = 0;
  result.worker_id = 0;

  EXPECT_TRUE(validate_inference_result(result, model, VerbosityLevel::Silent));
}

TEST(InferenceValidator, OutputCountMismatch)
{
  auto model = make_add_one_model();

  InferenceResult result;
  result.inputs = {torch::tensor({1, 2, 3})};
  result.results = {torch::tensor({2, 3, 4}), torch::tensor({2, 3, 4})};
  result.job_id = 45;
  result.executed_on = DeviceType::CPU;

  testing::internal::CaptureStderr();
  EXPECT_FALSE(
      validate_inference_result(result, model, VerbosityLevel::Silent));
  std::string logs = testing::internal::GetCapturedStderr();
  EXPECT_NE(logs.find("Output count mismatch"), std::string::npos);
}
