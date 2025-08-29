#include "test_inference_validator.hpp"

class InferenceValidator_Integration : public ::testing::Test {};

TEST_F(InferenceValidator_Integration, SuccessfulValidationCuda)
{
  skip_if_no_cuda();
  auto model = starpu_server::make_add_one_model();
  model.to(torch::kCUDA);
  auto opts = torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA);
  auto in = torch::arange(1, 4, opts);
  auto out = torch::arange(2, 5, opts);
  auto result = starpu_server::make_result(
      {in}, {out}, 100, starpu_server::DeviceType::CUDA);
  EXPECT_TRUE(validate_inference_result(
      result, model, starpu_server::VerbosityLevel::Silent));
}

TEST_F(InferenceValidator_Integration, FailsOnMismatchCuda_ReturnsFalseAndLogs)
{
  skip_if_no_cuda();
  auto model = starpu_server::make_add_one_model();
  model.to(torch::kCUDA);
  auto opts = torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA);
  auto in = torch::arange(1, 4, opts);
  auto wrong = torch::arange(1, 4, opts);
  auto res = starpu_server::make_result(
      {in}, {wrong}, 101, starpu_server::DeviceType::CUDA);

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

  auto res = starpu_server::make_result(
      {torch::tensor({1, 2, 3})},
      {torch::arange(
          2, 5,
          torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA))},
      102, starpu_server::DeviceType::CUDA);

  EXPECT_TRUE(validate_inference_result(
      res, model, starpu_server::VerbosityLevel::Silent));
}
