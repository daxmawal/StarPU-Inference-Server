#include <gtest/gtest.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <string>

#include "core/inference_runner.hpp"
#include "utils/exceptions.hpp"
#include "utils/inference_validator.hpp"
#include "utils/logger.hpp"

#define skip_if_no_cuda()                      \
  do {                                         \
    if (!torch::cuda::is_available()) {        \
      GTEST_SKIP() << "CUDA is not available"; \
    }                                          \
  } while (0)

class InferenceValidatorTest : public ::testing::Test {
 protected:
  static auto make_add_one_model() -> torch::jit::script::Module
  {
    torch::jit::script::Module m{"m"};
    m.define(R"JIT(
        def forward(self, x):
            return x + 1
    )JIT");
    return m;
  }

  static auto make_result(
      std::vector<torch::Tensor> inputs, std::vector<torch::Tensor> outputs,
      int job_id, starpu_server::DeviceType device, int device_id = 0,
      int worker_id = 0) -> starpu_server::InferenceResult
  {
    starpu_server::InferenceResult result;
    result.inputs = std::move(inputs);
    result.results = std::move(outputs);
    result.job_id = job_id;
    result.executed_on = device;
    result.device_id = device_id;
    result.worker_id = worker_id;
    return result;
  }
};

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

TEST_F(InferenceValidatorTest, SuccessfulValidation)
{
  auto model = make_add_one_model();
  auto result = make_result(
      {torch::tensor({1, 2, 3})}, {torch::tensor({2, 3, 4})}, 42,
      starpu_server::DeviceType::CPU);
  EXPECT_TRUE(validate_inference_result(
      result, model, starpu_server::VerbosityLevel::Silent));
}

TEST_F(InferenceValidatorTest, FailsOnMismatch)
{
  auto model = make_add_one_model();
  auto result = make_result(
      {torch::tensor({1, 2, 3})}, {torch::tensor({1, 2, 3})}, 43,
      starpu_server::DeviceType::CPU);
  testing::internal::CaptureStderr();
  EXPECT_FALSE(validate_inference_result(
      result, model, starpu_server::VerbosityLevel::Silent));
  std::string logs = testing::internal::GetCapturedStderr();
  EXPECT_NE(logs.find("Mismatch on output"), std::string::npos);
}

TEST_F(InferenceValidatorTest, ThrowsOnUnknownDevice)
{
  auto model = make_add_one_model();
  auto result = make_result(
      {torch::tensor({1, 2, 3})}, {torch::tensor({2, 3, 4})}, 44,
      starpu_server::DeviceType::Unknown);
  EXPECT_THROW(
      validate_inference_result(
          result, model, starpu_server::VerbosityLevel::Silent),
      starpu_server::InferenceExecutionException);
}

TEST_F(InferenceValidatorTest, ThrowsOnInvalidDeviceValue)
{
  auto model = make_add_one_model();
  auto result = make_result(
      {torch::tensor({1, 2, 3})}, {torch::tensor({2, 3, 4})}, 144,
      static_cast<starpu_server::DeviceType>(255));
  EXPECT_THROW(
      validate_inference_result(
          result, model, starpu_server::VerbosityLevel::Silent),
      starpu_server::InferenceExecutionException);
}

TEST_F(InferenceValidatorTest, ThrowsOnNonTensorTupleElement)
{
  auto model = make_tuple_non_tensor_model();
  auto result = make_result(
      {torch::tensor({1})}, {torch::tensor({1})}, 46,
      starpu_server::DeviceType::CPU);
  EXPECT_THROW(
      validate_inference_result(
          result, model, starpu_server::VerbosityLevel::Silent),
      starpu_server::InferenceExecutionException);
}

TEST_F(InferenceValidatorTest, ThrowsOnUnsupportedOutputType)
{
  auto model = make_string_model();
  auto result =
      make_result({torch::tensor({1})}, {}, 47, starpu_server::DeviceType::CPU);
  EXPECT_THROW(
      validate_inference_result(
          result, model, starpu_server::VerbosityLevel::Silent),
      starpu_server::InferenceExecutionException);
}

TEST_F(InferenceValidatorTest, EmptyTupleOutput)
{
  auto model = make_empty_tuple_model();
  auto result =
      make_result({torch::tensor({1})}, {}, 48, starpu_server::DeviceType::CPU);
  EXPECT_TRUE(validate_inference_result(
      result, model, starpu_server::VerbosityLevel::Silent));
}

TEST_F(InferenceValidatorTest, SuccessfulValidationCuda)
{
  skip_if_no_cuda();
  auto model = make_add_one_model();
  model.to(torch::kCUDA);
  auto result = make_result(
      {torch::tensor({1, 2, 3}).to(torch::kCUDA)},
      {torch::tensor({2, 3, 4}).to(torch::kCUDA)}, 100,
      starpu_server::DeviceType::CUDA);
  EXPECT_TRUE(validate_inference_result(
      result, model, starpu_server::VerbosityLevel::Silent));
}

TEST_F(InferenceValidatorTest, FailsOnMismatchCuda)
{
  skip_if_no_cuda();
  auto model = make_add_one_model();
  model.to(torch::kCUDA);
  auto result = make_result(
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
  skip_if_no_cuda();
  auto model = make_add_one_model();
  model.to(torch::kCUDA);
  auto result = make_result(
      {torch::tensor({1, 2, 3})}, {torch::tensor({2, 3, 4}).to(torch::kCUDA)},
      102, starpu_server::DeviceType::CUDA);
  EXPECT_TRUE(validate_inference_result(
      result, model, starpu_server::VerbosityLevel::Silent));
}

TEST_F(InferenceValidatorTest, OutputCountMismatch)
{
  auto model = make_add_one_model();
  auto result = make_result(
      {torch::tensor({1, 2, 3})},
      {torch::tensor({2, 3, 4}), torch::tensor({2, 3, 4})}, 45,
      starpu_server::DeviceType::CPU);
  testing::internal::CaptureStderr();
  EXPECT_FALSE(validate_inference_result(
      result, model, starpu_server::VerbosityLevel::Silent));
  std::string logs = testing::internal::GetCapturedStderr();
  EXPECT_NE(logs.find("Output count mismatch"), std::string::npos);
}
