#include <gtest/gtest.h>
#include <torch/script.h>

#include <filesystem>
#include <string>

#include "test_helpers.hpp"

TEST(RunInferenceLoopIntegration, CpuAddOneModel)
{
  const auto output = starpu_server::run_add_one_inference_loop(true, false);
  EXPECT_NE(output.find("Job 0 passed"), std::string::npos);
}

TEST(RunInferenceLoopIntegration, CudaAddOneModel)
{
  if (!torch::cuda::is_available()) {
    GTEST_SKIP();
  }
  const auto output = starpu_server::run_add_one_inference_loop(false, true, 0);
  EXPECT_NE(output.find("Job 0 passed"), std::string::npos);
}
