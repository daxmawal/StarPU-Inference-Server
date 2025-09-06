#include <bit>

#include "test_inference_validator.hpp"

namespace {
constexpr int kJobIdMismatch = 43;
constexpr int kJobIdTolerance = 150;
constexpr int kCudaMismatchJobId = 101;
constexpr int kCudaCpuInputsJobId = 102;
constexpr int kOutputCountMismatchJobId = 45;
constexpr float kDelta = 0.01F;
constexpr float kBase2F = 2.0F;
constexpr float kBase3F = 3.0F;
constexpr float kBase4F = 4.0F;
}  // namespace

enum class ValidationExpectation : std::uint8_t {
  Success,
  InferenceExecutionException
};

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
  constexpr int kCaseJobId = 50;
  auto result = starpu_server::make_result(
      param.inputs, param.outputs, kCaseJobId, param.device);

  auto validate_case =
      [&](ValidationExpectation exp) -> ::testing::AssertionResult {
    try {
      const bool is_valid = validate_inference_result(
          result, model, starpu_server::VerbosityLevel::Silent);
      if (exp == ValidationExpectation::Success) {
        return is_valid ? ::testing::AssertionSuccess()
                        : (::testing::AssertionFailure()
                           << "Validation returned false");
      }
      return ::testing::AssertionFailure()
             << "Expected exception but validation succeeded";
    }
    catch (const starpu_server::InferenceExecutionException&) {
      if (exp == ValidationExpectation::InferenceExecutionException) {
        return ::testing::AssertionSuccess();
      }
      return ::testing::AssertionFailure()
             << "Unexpected InferenceExecutionException thrown";
    }
  };

  EXPECT_TRUE(validate_case(param.expectation));
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
            std::bit_cast<starpu_server::DeviceType>(std::uint8_t{255}),
            ValidationExpectation::InferenceExecutionException},
        ValidationCase{
            make_tuple_non_tensor_model,
            {torch::tensor({1})},
            {torch::tensor({1})},
            starpu_server::DeviceType::CPU,
            ValidationExpectation::InferenceExecutionException},
        ValidationCase{
            make_tensor_list_model,
            {torch::tensor({1, 2, 3})},
            {torch::tensor({1, 2, 3}), torch::tensor({2, 3, 4})},
            starpu_server::DeviceType::CPU,
            ValidationExpectation::Success},
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
      {torch::tensor({1, 2, 3})}, {torch::tensor({1, 2, 3})}, kJobIdMismatch,
      starpu_server::DeviceType::CPU);
  testing::internal::CaptureStderr();
  EXPECT_FALSE(validate_inference_result(
      result, model, starpu_server::VerbosityLevel::Silent));
  std::string logs = testing::internal::GetCapturedStderr();
  EXPECT_NE(logs.find("Mismatch on output"), std::string::npos);
}

TEST_F(InferenceValidatorTest, CustomTolerancePasses)
{
  auto model = starpu_server::make_add_one_model();
  auto result = starpu_server::make_result(
      {torch::tensor({1, 2, 3})},
      {torch::tensor({kBase2F + kDelta, kBase3F + kDelta, kBase4F + kDelta})},
      kJobIdTolerance, starpu_server::DeviceType::CPU);
  EXPECT_TRUE(validate_inference_result(
      result, model, starpu_server::VerbosityLevel::Silent, 1e-1, 1e-1));
}

TEST_F(InferenceValidatorTest, FailsOnMismatchCuda)
{
  skip_if_no_cuda();
  auto model = starpu_server::make_add_one_model();
  model.to(torch::kCUDA);
  auto result = starpu_server::make_result(
      {torch::tensor({1, 2, 3}).to(torch::kCUDA)},
      {torch::tensor({1, 2, 3}).to(torch::kCUDA)}, kCudaMismatchJobId,
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
  auto model = starpu_server::make_add_one_model();
  model.to(torch::kCUDA);
  auto result = starpu_server::make_result(
      {torch::tensor({1, 2, 3})}, {torch::tensor({2, 3, 4}).to(torch::kCUDA)},
      kCudaCpuInputsJobId, starpu_server::DeviceType::CUDA);
  EXPECT_TRUE(validate_inference_result(
      result, model, starpu_server::VerbosityLevel::Silent));
}

TEST_F(InferenceValidatorTest, OutputCountMismatch)
{
  auto model = starpu_server::make_add_one_model();
  auto result = starpu_server::make_result(
      {torch::tensor({1, 2, 3})},
      {torch::tensor({2, 3, 4}), torch::tensor({2, 3, 4})},
      kOutputCountMismatchJobId, starpu_server::DeviceType::CPU);
  testing::internal::CaptureStderr();
  EXPECT_FALSE(validate_inference_result(
      result, model, starpu_server::VerbosityLevel::Silent));
  std::string logs = testing::internal::GetCapturedStderr();
  EXPECT_NE(logs.find("Output count mismatch"), std::string::npos);
}
