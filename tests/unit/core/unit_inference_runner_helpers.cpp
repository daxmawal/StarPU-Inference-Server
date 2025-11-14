#include <gtest/gtest.h>
#include <torch/script.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "core/inference_runner.hpp"
#include "test_helpers.hpp"
#include "test_inference_runner.hpp"
#include "torch_cuda_device_count_override.hpp"
#include "utils/exceptions.hpp"

namespace {

struct InferenceRunnerParam {
  std::string name;
  std::function<torch::jit::Module()> make_module;
  std::size_t expected_outputs;
  std::vector<std::function<void(
      const std::vector<torch::Tensor>&, const std::vector<torch::Tensor>&)>>
      assertions;
};

class CudaDeviceCountOverrideGuard {
 public:
  explicit CudaDeviceCountOverrideGuard(int override_count)
  {
    starpu_server::detail::set_cuda_device_count_override(override_count);
  }

  CudaDeviceCountOverrideGuard(const CudaDeviceCountOverrideGuard&) = delete;
  auto operator=(const CudaDeviceCountOverrideGuard&)
      -> CudaDeviceCountOverrideGuard& = delete;

  CudaDeviceCountOverrideGuard(CudaDeviceCountOverrideGuard&&) = delete;
  auto operator=(CudaDeviceCountOverrideGuard&&)
      -> CudaDeviceCountOverrideGuard& = delete;

  ~CudaDeviceCountOverrideGuard()
  {
    starpu_server::detail::set_cuda_device_count_override(std::nullopt);
  }
};

}  // namespace

class InferenceRunnerHelpersTest
    : public ::testing::TestWithParam<InferenceRunnerParam> {};

TEST_P(InferenceRunnerHelpersTest, RunReferenceInference)
{
  const auto& param = GetParam();
  auto model = param.make_module();
  std::vector<torch::Tensor> inputs{torch::ones({2})};
  auto outputs = starpu_server::run_reference_inference(model, inputs);

  ASSERT_EQ(outputs.size(), param.expected_outputs);

  for (const auto& assertion : param.assertions) {
    assertion(inputs, outputs);
  }
}

INSTANTIATE_TEST_SUITE_P(
    RunReferenceInference, InferenceRunnerHelpersTest,
    ::testing::Values(
        InferenceRunnerParam{
            "Tensor",
            [] { return starpu_server::make_mul_two_model(); },
            1U,
            {[](const auto& inputs, const auto& outputs) {
              EXPECT_TRUE(torch::allclose(outputs[0], inputs[0] * 2));
            }}},
        InferenceRunnerParam{
            "Tuple",
            [] { return starpu_server::make_tuple_model(); },
            2U,
            {[](const auto& inputs, const auto& outputs) {
               EXPECT_TRUE(torch::allclose(outputs[0], inputs[0]));
             },
             [](const auto& inputs, const auto& outputs) {
               EXPECT_TRUE(torch::allclose(outputs[1], inputs[0] + 1));
             }}},
        InferenceRunnerParam{
            "TensorList",
            [] { return starpu_server::make_tensor_list_model(); },
            2U,
            {[](const auto& inputs, const auto& outputs) {
               EXPECT_TRUE(torch::allclose(outputs[0], inputs[0]));
             },
             [](const auto& inputs, const auto& outputs) {
               EXPECT_TRUE(torch::allclose(outputs[1], inputs[0] + 1));
             }}}),
    [](const ::testing::TestParamInfo<InferenceRunnerParam>& info) {
      return info.param.name;
    });

TEST(InferenceRunnerDeviceValidationTest, HandlesMockedLargeCudaDeviceCount)
{
  constexpr int kLargeDeviceCount = 512;
  const CudaDeviceCountOverrideGuard guard(kLargeDeviceCount);

  ASSERT_EQ(starpu_server::detail::get_cuda_device_count(), kLargeDeviceCount);

  const std::vector<int> valid_ids{0, 255, 511};
  EXPECT_NO_THROW(starpu_server::detail::validate_device_ids(
      valid_ids, starpu_server::detail::get_cuda_device_count()));

  const std::vector<int> invalid_ids{512};
  EXPECT_THROW(
      starpu_server::detail::validate_device_ids(
          invalid_ids, starpu_server::detail::get_cuda_device_count()),
      starpu_server::InvalidGpuDeviceException);
}

TEST(
    InferenceRunnerDeviceValidationTest,
    GetCudaDeviceCountThrowsOnNegativeRawCount)
{
  starpu_server::detail::set_cuda_device_count_override(std::nullopt);
  const starpu_server::testing::TorchCudaDeviceCountOverrideGuard guard(
      static_cast<c10::DeviceIndex>(-1));

  EXPECT_THROW(
      starpu_server::detail::get_cuda_device_count(),
      starpu_server::InvalidGpuDeviceException);
}

TEST(
    InferenceRunnerDeviceValidationTest,
    GetCudaDeviceCountThrowsWhenRawCountExceedsIntMax)
{
  starpu_server::detail::set_cuda_device_count_override(std::nullopt);

  if (std::numeric_limits<c10::DeviceIndex>::max() <=
      std::numeric_limits<int>::max()) {
    GTEST_SKIP()
        << "c10::DeviceIndex max ("
        << static_cast<long long>(std::numeric_limits<c10::DeviceIndex>::max())
        << ") does not exceed std::numeric_limits<int>::max(); overflow "
           "scenario cannot occur on this platform.";
  }

  const auto overflowing_raw_count = static_cast<c10::DeviceIndex>(
      static_cast<long long>(std::numeric_limits<int>::max()) + 1);
  const starpu_server::testing::TorchCudaDeviceCountOverrideGuard guard(
      overflowing_raw_count);

  EXPECT_THROW(
      starpu_server::detail::get_cuda_device_count(),
      starpu_server::InvalidGpuDeviceException);
}

TEST(InferenceRunnerHelpers, LoadModelAndReferenceOutputCorruptFile)
{
  namespace fs = std::filesystem;
  auto tmp_path = fs::temp_directory_path() / "corrupt_model.pt";
  {
    std::ofstream tmp_file{tmp_path};
    tmp_file << "invalid";
  }

  auto opts = starpu_server::make_single_model_runtime_config(
      tmp_path, std::vector<int64_t>{1}, at::kFloat);

  starpu_server::CaptureStream capture{std::cerr};
  auto result = starpu_server::load_model_and_reference_output(opts);
  auto err = capture.str();

  EXPECT_EQ(result, std::nullopt);
  EXPECT_NE(
      err.find("Failed to load model or run reference inference"),
      std::string::npos);

  fs::remove(tmp_path);
}

TEST(InferenceRunnerHelpers, RunReferenceInferenceUnsupportedOutput)
{
  auto model = starpu_server::make_constant_model();
  std::vector<torch::Tensor> inputs{torch::ones({1})};
  EXPECT_THROW(
      starpu_server::run_reference_inference(model, inputs),
      starpu_server::UnsupportedModelOutputTypeException);
}

class LoadModelAndReferenceOutputError
    : public ::testing::TestWithParam<at::ScalarType> {};

TEST_P(LoadModelAndReferenceOutputError, MissingFile)
{
  auto opts = starpu_server::make_single_model_runtime_config(
      "nonexistent_model.pt", std::vector<int64_t>{1}, GetParam());
  starpu_server::CaptureStream capture{std::cerr};
  auto result = starpu_server::load_model_and_reference_output(opts);
  auto err = capture.str();
  EXPECT_EQ(result, std::nullopt);
  EXPECT_NE(
      err.find("Failed to load model or run reference inference"),
      std::string::npos);
}

INSTANTIATE_TEST_SUITE_P(
    InferenceRunnerHelpers, LoadModelAndReferenceOutputError,
    ::testing::Values(torch::kFloat32, at::kFloat));
