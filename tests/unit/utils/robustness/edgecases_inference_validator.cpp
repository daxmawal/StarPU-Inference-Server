
#include <limits>

#include "test_inference_validator.hpp"

namespace {
constexpr int kLatencyUnknown = 60;
constexpr int kLatencyInvalid = 61;
constexpr int kLatencyTuple = 62;
constexpr int kLatencyString = 63;
constexpr int kLatencyShape = 64;
constexpr int kLatencyMismatch = 65;
constexpr int kLatencyCountMismatch = 66;

inline auto
make_invalid_device() -> starpu_server::DeviceType
{
  auto raw = static_cast<uint8_t>(std::numeric_limits<uint8_t>::max());
  return static_cast<starpu_server::DeviceType>(raw);
}
}  // namespace


class InferenceValidator_Robustesse : public ::testing::Test {};

TEST_F(InferenceValidator_Robustesse, UnknownDevice_Throws)
{
  auto model = starpu_server::make_add_one_model();
  auto res = starpu_server::make_result(
      {torch::tensor({1, 2, 3})}, {torch::tensor({2, 3, 4})}, kLatencyUnknown,
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
      {torch::tensor({1, 2, 3})}, {torch::tensor({2, 3, 4})}, kLatencyInvalid,
      make_invalid_device());

  EXPECT_THROW(
      validate_inference_result(
          res, model, starpu_server::VerbosityLevel::Silent),
      starpu_server::InferenceExecutionException);
}

TEST_F(InferenceValidator_Robustesse, TupleWithNonTensor_Throws)
{
  auto model = make_tuple_non_tensor_model();
  auto res = starpu_server::make_result(
      {torch::tensor({1})}, {torch::tensor({1})}, kLatencyTuple,
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
      {torch::tensor({1})}, /*outputs*/ {}, kLatencyString,
      starpu_server::DeviceType::CPU);

  EXPECT_THROW(
      validate_inference_result(
          res, model, starpu_server::VerbosityLevel::Silent),
      starpu_server::InferenceExecutionException);
}

TEST_F(InferenceValidator_Robustesse, ShapeErrorModel_Throws)
{
  auto model = make_shape_error_model();
  auto res = starpu_server::make_result(
      {torch::rand({2, 2})}, /*outputs*/ {}, kLatencyShape,
      starpu_server::DeviceType::CPU);

  EXPECT_THROW(
      validate_inference_result(
          res, model, starpu_server::VerbosityLevel::Silent),
      starpu_server::InferenceExecutionException);
}

TEST_F(InferenceValidator_Robustesse, ShapeErrorModel_LogsC10Error)
{
  auto model = make_shape_error_model();
  auto res = starpu_server::make_result(
      {torch::rand({2, 2})}, /*outputs*/ {}, kLatencyShape,
      starpu_server::DeviceType::CPU);

  testing::internal::CaptureStderr();
  EXPECT_THROW(
      validate_inference_result(
          res, model, starpu_server::VerbosityLevel::Silent),
      starpu_server::InferenceExecutionException);
  auto logs = testing::internal::GetCapturedStderr();
  EXPECT_NE(logs.find("[Validator] C10 error"), std::string::npos);
}

TEST_F(InferenceValidator_Robustesse, OutputMismatch_ReturnsFalseAndLogs)
{
  auto model = starpu_server::make_add_one_model();
  auto res = starpu_server::make_result(
      {torch::tensor({1, 2, 3})}, {torch::tensor({1, 2, 3})}, kLatencyMismatch,
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
      {torch::tensor({2, 3, 4}), torch::tensor({2, 3, 4})},
      kLatencyCountMismatch, starpu_server::DeviceType::CPU);

  testing::internal::CaptureStderr();
  EXPECT_FALSE(validate_inference_result(
      res, model, starpu_server::VerbosityLevel::Silent));
  std::string logs = testing::internal::GetCapturedStderr();
  EXPECT_NE(logs.find("Output count mismatch"), std::string::npos);
}
