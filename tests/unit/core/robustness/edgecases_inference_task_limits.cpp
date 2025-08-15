#include <gtest/gtest.h>
#include <starpu.h>
#include <torch/script.h>

#include <memory>
#include <string>
#include <vector>

#include "core/inference_task.hpp"
#include "utils/exceptions.hpp"

class InferenceTaskLimitsTest_Robustesse : public ::testing::Test {
 protected:
  void SetUp() override
  {
    job = std::make_shared<starpu_server::InferenceJob>();
    model_cpu = torch::jit::script::Module{"m"};
    models_gpu.clear();
    opts = starpu_server::RuntimeConfig{};
    task = std::make_unique<starpu_server::InferenceTask>(
        nullptr, job, &model_cpu, &models_gpu, &opts);
  }

  std::shared_ptr<starpu_server::InferenceJob> job;
  torch::jit::script::Module model_cpu;
  std::vector<torch::jit::script::Module> models_gpu;
  starpu_server::RuntimeConfig opts;
  std::unique_ptr<starpu_server::InferenceTask> task;
};

TEST_F(InferenceTaskLimitsTest_Robustesse, CheckLimitsTooManyInputs)
{
  const size_t num_inputs = starpu_server::InferLimits::MaxInputs + 1;
  EXPECT_THROW(
      task->check_limits(num_inputs),
      starpu_server::InferenceExecutionException);
}

TEST_F(InferenceTaskLimitsTest_Robustesse, CheckLimitsTooManyGpuModels)
{
  models_gpu.reserve(starpu_server::InferLimits::MaxModelsGPU + 1);
  for (size_t i = 0; i < starpu_server::InferLimits::MaxModelsGPU + 1; ++i) {
    models_gpu.emplace_back(std::string{"m"} + std::to_string(i));
  }
  EXPECT_THROW(task->check_limits(1), starpu_server::TooManyGpuModelsException);
}

TEST_F(InferenceTaskLimitsTest_Robustesse, FillInputLayoutTooManyDims)
{
  std::vector<int64_t> dims(starpu_server::InferLimits::MaxDims + 1, 1);
  auto tensor = torch::ones(dims);
  job->set_input_tensors({tensor});
  job->set_input_types({tensor.scalar_type()});

  auto params = std::make_shared<starpu_server::InferenceParams>();
  EXPECT_THROW(
      task->fill_input_layout(params, 1),
      starpu_server::InferenceExecutionException);
}
