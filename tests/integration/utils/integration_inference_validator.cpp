#include "test_inference_validator.hpp"

class InferenceValidator_Integration : public ::testing::Test {};

TEST_F(InferenceValidator_Integration, SuccessfulValidationCuda)
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

TEST_F(InferenceValidator_Integration, FailsOnMismatchCuda_ReturnsFalseAndLogs)
{
  SKIP_IF_NO_CUDA();
  auto model = starpu_server::make_add_one_model();
  model.to(torch::kCUDA);
  auto res = starpu_server::make_result(
      {torch::tensor({1, 2, 3}).to(torch::kCUDA)},
      {torch::tensor({1, 2, 3}).to(torch::kCUDA)}, 101,
      starpu_server::DeviceType::CUDA);

  testing::internal::CaptureStderr();
  EXPECT_FALSE(validate_inference_result(
      res, model, starpu_server::VerbosityLevel::Silent));
  std::string logs = testing::internal::GetCapturedStderr();
  EXPECT_NE(logs.find("Mismatch on output"), std::string::npos);
}

TEST_F(InferenceValidator_Integration, CudaModelOnCpuInputs_OK)
{
  SKIP_IF_NO_CUDA();
  auto model = starpu_server::make_add_one_model();
  model.to(torch::kCUDA);

  auto res = starpu_server::make_result(
      {torch::tensor({1, 2, 3})}, {torch::tensor({2, 3, 4}).to(torch::kCUDA)},
      102, starpu_server::DeviceType::CUDA);

  EXPECT_TRUE(validate_inference_result(
      res, model, starpu_server::VerbosityLevel::Silent));
}
