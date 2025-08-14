#include <gtest/gtest.h>
#include <starpu.h>
#include <torch/script.h>

#include <memory>
#include <string>
#include <vector>

#include "core/inference_task.hpp"
#include "utils/exceptions.hpp"

TEST(InferenceTask_Limits_Unit, ConstructAndCheckLimitsOK)
{
  auto job = std::make_shared<starpu_server::InferenceJob>();
  torch::jit::script::Module model_cpu{"m"};
  std::vector<torch::jit::script::Module> models_gpu;
  starpu_server::RuntimeConfig opts;

  starpu_server::InferenceTask task(
      nullptr, job, &model_cpu, &models_gpu, &opts);

  EXPECT_NO_THROW(task.check_limits(1));
}
