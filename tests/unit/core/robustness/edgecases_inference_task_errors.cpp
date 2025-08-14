#include <gtest/gtest.h>
#include <starpu.h>
#include <torch/script.h>

#include "core/inference_task.hpp"
#include "utils/exceptions.hpp"

TEST(InferenceTask_Robustesse, SubmitNullJobThrows)
{
  auto job = std::shared_ptr<starpu_server::InferenceJob>(nullptr);
  torch::jit::script::Module model_cpu;
  std::vector<torch::jit::script::Module> models_gpu;
  starpu_server::RuntimeConfig opts;

  starpu_server::InferenceTask task(
      nullptr, job, &model_cpu, &models_gpu, &opts);
  EXPECT_THROW(task.submit(), starpu_server::InvalidInferenceJobException);
}

TEST(InferenceTask_Robustesse, CheckLimitsTooManyInputsThrows)
{
  auto job = std::make_shared<starpu_server::InferenceJob>();
  torch::jit::script::Module model_cpu{"m"};
  std::vector<torch::jit::script::Module> models_gpu;
  starpu_server::RuntimeConfig opts;

  starpu_server::InferenceTask task(
      nullptr, job, &model_cpu, &models_gpu, &opts);

  const size_t num_inputs = starpu_server::InferLimits::MaxInputs + 1;
  EXPECT_THROW(
      task.check_limits(num_inputs),
      starpu_server::InferenceExecutionException);
}

TEST(InferenceTask_Robustesse, CheckLimitsTooManyGpuModelsThrows)
{
  auto job = std::make_shared<starpu_server::InferenceJob>();
  torch::jit::script::Module model_cpu{"m"};
  std::vector<torch::jit::script::Module> models_gpu;
  models_gpu.reserve(starpu_server::InferLimits::MaxModelsGPU + 1);
  for (size_t i = 0; i < starpu_server::InferLimits::MaxModelsGPU + 1; ++i) {
    models_gpu.emplace_back(std::string{"m"} + std::to_string(i));
  }
  starpu_server::RuntimeConfig opts;

  starpu_server::InferenceTask task(
      nullptr, job, &model_cpu, &models_gpu, &opts);
  EXPECT_THROW(task.check_limits(1), starpu_server::TooManyGpuModelsException);
}
