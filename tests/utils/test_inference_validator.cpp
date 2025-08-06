#include <gtest/gtest.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <string>
#include <utility>
#include <vector>

#include "core/inference_runner.hpp"
#include "test_helpers.hpp"
#include "utils/exceptions.hpp"
#include "utils/inference_validator.hpp"
#include "utils/logger.hpp"

namespace starpu_server {

inline auto
make_result(
    std::vector<torch::Tensor> inputs, std::vector<torch::Tensor> outputs,
    int job_id, DeviceType device, int device_id = 0,
    int worker_id = 0) -> InferenceResult
{
  InferenceResult result;
  result.inputs = std::move(inputs);
  result.results = std::move(outputs);
  result.job_id = job_id;
  result.executed_on = device;
  result.device_id = device_id;
  result.worker_id = worker_id;
  return result;
}

}  // namespace starpu_server

class InferenceValidatorTest : public ::testing::Test {};

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

enum class ValidationExpectation { Success, InferenceExecutionException };

struct ValidationCase {
  std::function<torch::jit::script::Module()> make_model;
  std::vector<torch::Tensor> inputs;
  std::vector<torch::Tensor> outputs;
  starpu_server::DeviceType device;
  ValidationExpectation expectation;
};

class InferenceValidatorParamTest
    : public InferenceValidatorTest,
      public ::testing::WithParamInterface<ValidationCase> {};

TEST_P(InferenceValidatorParamTest, Validates)
{
  const auto& param = GetParam();
  auto model = param.make_model();
  auto result =
      starpu_server::make_result(param.inputs, param.outputs, 50, param.device);

  if (param.expectation == ValidationExpectation::Success) {
    EXPECT_TRUE(validate_inference_result(
        result, model, starpu_server::VerbosityLevel::Silent));
  } else {
    EXPECT_THROW(
        validate_inference_result(
            result, model, starpu_server::VerbosityLevel::Silent),
        starpu_server::InferenceExecutionException);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ValidationCases, InferenceValidatorParamTest,
    ::testing::Values(
        ValidationCase{
            starpu_server::make_add_one_model,
            {torch::tensor({1, 2, 3})},
            {torch::tensor({2, 3, 4})},
            starpu_server::DeviceType::CPU,
            ValidationExpectation::Success},
        ValidationCase{
            starpu_server::make_add_one_model,
            {torch::tensor({1, 2, 3})},
            {torch::tensor({2, 3, 4})},
            starpu_server::DeviceType::Unknown,
            ValidationExpectation::InferenceExecutionException},
        ValidationCase{
            starpu_server::make_add_one_model,
            {torch::tensor({1, 2, 3})},
            {torch::tensor({2, 3, 4})},
            static_cast<starpu_server::DeviceType>(255),
            ValidationExpectation::InferenceExecutionException},
        ValidationCase{
            make_tuple_non_tensor_model,
            {torch::tensor({1})},
            {torch::tensor({1})},
            starpu_server::DeviceType::CPU,
            ValidationExpectation::InferenceExecutionException},
        ValidationCase{
            make_string_model,
            {torch::tensor({1})},
            {},
            starpu_server::DeviceType::CPU,
            ValidationExpectation::InferenceExecutionException},
        ValidationCase{
            make_empty_tuple_model,
            {torch::tensor({1})},
            {},
            starpu_server::DeviceType::CPU,
            ValidationExpectation::Success}));

TEST_F(InferenceValidatorTest, FailsOnMismatch)
{
  auto model = starpu_server::make_add_one_model();
  auto result = starpu_server::make_result(
      {torch::tensor({1, 2, 3})}, {torch::tensor({1, 2, 3})}, 43,
      starpu_server::DeviceType::CPU);
  testing::internal::CaptureStderr();
  EXPECT_FALSE(validate_inference_result(
      result, model, starpu_server::VerbosityLevel::Silent));
  std::string logs = testing::internal::GetCapturedStderr();
  EXPECT_NE(logs.find("Mismatch on output"), std::string::npos);
}

TEST_F(InferenceValidatorTest, SuccessfulValidationCuda)
{
  SKIP_IF_NO_CUDA();
  auto model = starpu_server::make_add_one_model();
  model.to(torch::kCUDA);
  auto result = starpu_server::make_result(
      {torch::tensor({1, 2, 3}).to(torch::kCUDA)},
      {torch::tensor({2, 3, 4}).to(torch::kCUDA)}, 100,
      starpu_server::DeviceType::CUDA);
  EXPECT_TRUE(validate_inference_result(
      result, model, starpu_server::VerbosityLevel::Silent));
}

TEST_F(InferenceValidatorTest, FailsOnMismatchCuda)
{
  SKIP_IF_NO_CUDA();
  auto model = starpu_server::make_add_one_model();
  model.to(torch::kCUDA);
  auto result = starpu_server::make_result(
      {torch::tensor({1, 2, 3}).to(torch::kCUDA)},
      {torch::tensor({1, 2, 3}).to(torch::kCUDA)}, 101,
      starpu_server::DeviceType::CUDA);
  testing::internal::CaptureStderr();
  EXPECT_FALSE(validate_inference_result(
      result, model, starpu_server::VerbosityLevel::Silent));
  std::string logs = testing::internal::GetCapturedStderr();
  EXPECT_NE(logs.find("Mismatch on output"), std::string::npos);
}

TEST_F(InferenceValidatorTest, CudaModelOnCpuInputsThrows)
{
  SKIP_IF_NO_CUDA();
  auto model = starpu_server::make_add_one_model();
  model.to(torch::kCUDA);
  auto result = starpu_server::make_result(
      {torch::tensor({1, 2, 3})}, {torch::tensor({2, 3, 4}).to(torch::kCUDA)},
      102, starpu_server::DeviceType::CUDA);
  EXPECT_TRUE(validate_inference_result(
      result, model, starpu_server::VerbosityLevel::Silent));
}

TEST_F(InferenceValidatorTest, OutputCountMismatch)
{
  auto model = starpu_server::make_add_one_model();
  auto result = starpu_server::make_result(
      {torch::tensor({1, 2, 3})},
      {torch::tensor({2, 3, 4}), torch::tensor({2, 3, 4})}, 45,
      starpu_server::DeviceType::CPU);
  testing::internal::CaptureStderr();
  EXPECT_FALSE(validate_inference_result(
      result, model, starpu_server::VerbosityLevel::Silent));
  std::string logs = testing::internal::GetCapturedStderr();
  EXPECT_NE(logs.find("Output count mismatch"), std::string::npos);
}
