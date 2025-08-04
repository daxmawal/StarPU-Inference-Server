#include <gtest/gtest.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <string>

#include "core/inference_runner.hpp"
#include "inference_validator_test_utils.hpp"
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
  auto result = make_result(
      {torch::tensor({1, 2, 3})}, {torch::tensor({2, 3, 4})}, 42,
      DeviceType::CPU);
  EXPECT_TRUE(validate_inference_result(result, model, VerbosityLevel::Silent));
}

TEST(InferenceValidator, FailsOnMismatch)
{
  auto model = make_add_one_model();
  auto result = make_result(
      {torch::tensor({1, 2, 3})}, {torch::tensor({1, 2, 3})}, 43,
      DeviceType::CPU);
  testing::internal::CaptureStderr();
  EXPECT_FALSE(
      validate_inference_result(result, model, VerbosityLevel::Silent));
  std::string logs = testing::internal::GetCapturedStderr();
  EXPECT_NE(logs.find("Mismatch on output"), std::string::npos);
}

TEST(InferenceValidator, ThrowsOnUnknownDevice)
{
  auto model = make_add_one_model();
  auto result = make_result(
      {torch::tensor({1, 2, 3})}, {torch::tensor({2, 3, 4})}, 44,
      DeviceType::Unknown);
  EXPECT_THROW(
      validate_inference_result(result, model, VerbosityLevel::Silent),
      InferenceExecutionException);
}

TEST(InferenceValidator, ThrowsOnInvalidDeviceValue)
{
  auto model = make_add_one_model();
  auto result = make_result(
      {torch::tensor({1, 2, 3})}, {torch::tensor({2, 3, 4})}, 144,
      static_cast<DeviceType>(255));
  EXPECT_THROW(
      validate_inference_result(result, model, VerbosityLevel::Silent),
      InferenceExecutionException);
}

TEST(InferenceValidator, ThrowsOnNonTensorTupleElement)
{
  auto model = make_tuple_non_tensor_model();
  auto result = make_result(
      {torch::tensor({1})}, {torch::tensor({1})}, 46, DeviceType::CPU);
  EXPECT_THROW(
      validate_inference_result(result, model, VerbosityLevel::Silent),
      InferenceExecutionException);
}

TEST(InferenceValidator, ThrowsOnUnsupportedOutputType)
{
  auto model = make_string_model();
  auto result = make_result({torch::tensor({1})}, {}, 47, DeviceType::CPU);
  EXPECT_THROW(
      validate_inference_result(result, model, VerbosityLevel::Silent),
      InferenceExecutionException);
}

TEST(InferenceValidator, EmptyTupleOutput)
{
  auto model = make_empty_tuple_model();
  auto result = make_result({torch::tensor({1})}, {}, 48, DeviceType::CPU);
  EXPECT_TRUE(validate_inference_result(result, model, VerbosityLevel::Silent));
}

TEST(InferenceValidator, SuccessfulValidationCuda)
{
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available";
  }
  auto model = make_add_one_model();
  model.to(torch::kCUDA);
  auto result = make_result(
      {torch::tensor({1, 2, 3}).to(torch::kCUDA)},
      {torch::tensor({2, 3, 4}).to(torch::kCUDA)}, 100, DeviceType::CUDA);
  EXPECT_TRUE(validate_inference_result(result, model, VerbosityLevel::Silent));
}

TEST(InferenceValidator, FailsOnMismatchCuda)
{
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available";
  }
  auto model = make_add_one_model();
  model.to(torch::kCUDA);
  auto result = make_result(
      {torch::tensor({1, 2, 3}).to(torch::kCUDA)},
      {torch::tensor({1, 2, 3}).to(torch::kCUDA)}, 101, DeviceType::CUDA);
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
  auto result = make_result(
      {torch::tensor({1, 2, 3})}, {torch::tensor({2, 3, 4}).to(torch::kCUDA)},
      102, DeviceType::CUDA);
  EXPECT_TRUE(validate_inference_result(result, model, VerbosityLevel::Silent));
}

TEST(InferenceValidator, OutputCountMismatch)
{
  auto model = make_add_one_model();
  auto result = make_result(
      {torch::tensor({1, 2, 3})},
      {torch::tensor({2, 3, 4}), torch::tensor({2, 3, 4})}, 45,
      DeviceType::CPU);
  testing::internal::CaptureStderr();
  EXPECT_FALSE(
      validate_inference_result(result, model, VerbosityLevel::Silent));
  std::string logs = testing::internal::GetCapturedStderr();
  EXPECT_NE(logs.find("Output count mismatch"), std::string::npos);
}
