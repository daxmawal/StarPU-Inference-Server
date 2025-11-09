#include <gtest/gtest.h>
#include <torch/script.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <optional>
#include <vector>

#include "core/inference_runner.hpp"
#include "test_helpers.hpp"
#include "test_inference_runner.hpp"
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
