#include <gtest/gtest.h>
#include <torch/script.h>

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "test_helpers.hpp"

TEST(RunInferenceLoopIntegration, CpuAddOneModel)
{
  const auto output = starpu_server::run_add_one_inference_loop(true, false);
  EXPECT_NE(output.find("StarPUTaskRunner stopped"), std::string::npos);
}

TEST(RunInferenceLoopIntegration, CudaAddOneModel)
{
  skip_if_no_cuda();
  const auto output = starpu_server::run_add_one_inference_loop(false, true, 0);
  EXPECT_NE(output.find("StarPUTaskRunner stopped"), std::string::npos);
}

TEST(RunInferenceLoopIntegration, CudaAddOneModelNonContiguousDeviceIds)
{
  skip_if_no_cuda();
  const auto device_count = torch::cuda::device_count();
  if (device_count < 3) {
    GTEST_SKIP() << "Need at least 3 CUDA devices for non-contiguous IDs";
  }
  const auto output = starpu_server::run_add_one_inference_loop(
      false, true, std::nullopt, true, std::vector<int>{0, 2});
  EXPECT_NE(output.find("StarPUTaskRunner stopped"), std::string::npos);
}

TEST(RunInferenceLoopIntegration, DisableValidationSkipsChecks)
{
  const auto output = starpu_server::run_add_one_inference_loop(
      true, false, std::nullopt, false);
  EXPECT_NE(output.find("StarPUTaskRunner stopped"), std::string::npos);
  EXPECT_NE(
      output.find("Result validation disabled; skipping server-side checks."),
      std::string::npos);
}
