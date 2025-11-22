#include "test_helpers.hpp"
#include "test_inference_validator.hpp"
#include "utils/inference_validator.hpp"

class InferenceValidator_Integration : public ::testing::Test {};

TEST_F(InferenceValidator_Integration, SuccessfulValidationCuda)
{
  skip_if_no_cuda();
  auto model = starpu_server::make_add_one_model();
  model.to(torch::kCUDA);
  auto opts = torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA);
  constexpr int kJobId = 100;
  constexpr int64_t kInputStart = 1;
  constexpr int64_t kInputEnd = 4;
  constexpr int64_t kOutputStart = 2;
  constexpr int64_t kOutputEnd = 5;

  auto input_tensor = torch::arange(kInputStart, kInputEnd, opts);
  auto expected_output = torch::arange(kOutputStart, kOutputEnd, opts);
  auto result = starpu_server::make_result(
      {input_tensor}, {expected_output}, kJobId,
      starpu_server::DeviceType::CUDA);
  EXPECT_TRUE(validate_inference_result(
      result, model, starpu_server::VerbosityLevel::Silent));
}

TEST_F(InferenceValidator_Integration, FailsOnMismatchCuda_ReturnsFalseAndLogs)
{
  skip_if_no_cuda();
  auto model = starpu_server::make_add_one_model();
  model.to(torch::kCUDA);
  auto opts = torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA);
  constexpr int kJobId = 101;
  constexpr int64_t kInputStart = 1;
  constexpr int64_t kInputEnd = 4;

  auto input_tensor = torch::arange(kInputStart, kInputEnd, opts);
  auto wrong = torch::arange(1, 4, opts);
  auto res = starpu_server::make_result(
      {input_tensor}, {wrong}, kJobId, starpu_server::DeviceType::CUDA);

  testing::internal::CaptureStderr();
  EXPECT_FALSE(validate_inference_result(
      res, model, starpu_server::VerbosityLevel::Silent));
  std::string logs = testing::internal::GetCapturedStderr();
  EXPECT_NE(logs.find("Mismatch on output"), std::string::npos);
}

TEST_F(InferenceValidator_Integration, CudaModelOnCpuInputs_OK)
{
  skip_if_no_cuda();
  auto model = starpu_server::make_add_one_model();
  model.to(torch::kCUDA);

  constexpr int kJobId = 102;
  constexpr int64_t kExpectedStart = 2;
  constexpr int64_t kExpectedEnd = 5;

  auto res = starpu_server::make_result(
      {torch::tensor({1, 2, 3})},
      {torch::arange(
          kExpectedStart, kExpectedEnd,
          torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA))},
      kJobId, starpu_server::DeviceType::CUDA);

  EXPECT_TRUE(validate_inference_result(
      res, model, starpu_server::VerbosityLevel::Silent));
}
