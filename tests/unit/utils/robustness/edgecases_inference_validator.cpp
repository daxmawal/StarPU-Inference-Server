
#include "test_inference_validator.hpp"


class InferenceValidator_Robustesse : public ::testing::Test {};

TEST_F(InferenceValidator_Robustesse, UnknownDevice_Throws)
{
  auto model = starpu_server::make_add_one_model();
  auto res = starpu_server::make_result(
      {torch::tensor({1, 2, 3})}, {torch::tensor({2, 3, 4})}, 60,
      starpu_server::DeviceType::Unknown);

  EXPECT_THROW(
      validate_inference_result(
          res, model, starpu_server::VerbosityLevel::Silent),
      starpu_server::InferenceExecutionException);
}

TEST_F(InferenceValidator_Robustesse, InvalidDeviceEnum_Throws)
{
  auto model = starpu_server::make_add_one_model();
  auto res = starpu_server::make_result(
      {torch::tensor({1, 2, 3})}, {torch::tensor({2, 3, 4})}, 61,
      static_cast<starpu_server::DeviceType>(255));

  EXPECT_THROW(
      validate_inference_result(
          res, model, starpu_server::VerbosityLevel::Silent),
      starpu_server::InferenceExecutionException);
}

TEST_F(InferenceValidator_Robustesse, TupleWithNonTensor_Throws)
{
  auto model = make_tuple_non_tensor_model();
  auto res = starpu_server::make_result(
      {torch::tensor({1})}, {torch::tensor({1})}, 62,
      starpu_server::DeviceType::CPU);

  EXPECT_THROW(
      validate_inference_result(
          res, model, starpu_server::VerbosityLevel::Silent),
      starpu_server::InferenceExecutionException);
}

TEST_F(InferenceValidator_Robustesse, StringOutput_Throws)
{
  auto model = make_string_model();
  auto res = starpu_server::make_result(
      {torch::tensor({1})}, /*outputs*/ {}, 63, starpu_server::DeviceType::CPU);

  EXPECT_THROW(
      validate_inference_result(
          res, model, starpu_server::VerbosityLevel::Silent),
      starpu_server::InferenceExecutionException);
}

TEST_F(InferenceValidator_Robustesse, ShapeErrorModel_Throws)
{
  auto model = make_shape_error_model();
  auto res = starpu_server::make_result(
      {torch::rand({2, 2})}, /*outputs*/ {}, 64,
      starpu_server::DeviceType::CPU);

  EXPECT_THROW(
      validate_inference_result(
          res, model, starpu_server::VerbosityLevel::Silent),
      starpu_server::InferenceExecutionException);
}

TEST_F(InferenceValidator_Robustesse, OutputMismatch_ReturnsFalseAndLogs)
{
  auto model = starpu_server::make_add_one_model();
  auto res = starpu_server::make_result(
      {torch::tensor({1, 2, 3})}, {torch::tensor({1, 2, 3})}, 65,
      starpu_server::DeviceType::CPU);

  testing::internal::CaptureStderr();
  EXPECT_FALSE(validate_inference_result(
      res, model, starpu_server::VerbosityLevel::Silent));
  std::string logs = testing::internal::GetCapturedStderr();
  EXPECT_NE(logs.find("Mismatch on output"), std::string::npos);
}

TEST_F(InferenceValidator_Robustesse, OutputCountMismatch_ReturnsFalseAndLogs)
{
  auto model = starpu_server::make_add_one_model();
  auto res = starpu_server::make_result(
      {torch::tensor({1, 2, 3})},
      {torch::tensor({2, 3, 4}), torch::tensor({2, 3, 4})}, 66,
      starpu_server::DeviceType::CPU);

  testing::internal::CaptureStderr();
  EXPECT_FALSE(validate_inference_result(
      res, model, starpu_server::VerbosityLevel::Silent));
  std::string logs = testing::internal::GetCapturedStderr();
  EXPECT_NE(logs.find("Output count mismatch"), std::string::npos);
}
