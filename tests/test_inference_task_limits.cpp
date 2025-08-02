#include <gtest/gtest.h>
#include <starpu.h>
#include <torch/script.h>

#include "core/inference_task.hpp"
#include "utils/exceptions.hpp"

using namespace starpu_server;

TEST(InferenceTaskLimits, SafeRegisterUndefinedTensorThrows)
{
  torch::Tensor undefined;
  EXPECT_THROW(
      InferenceTask::safe_register_tensor_vector(undefined, "t"),
      StarPURegistrationException);
}

TEST(InferenceTaskLimits, CheckLimitsTooManyInputs)
{
  auto job = std::make_shared<InferenceJob>();
  torch::jit::script::Module model_cpu{"m"};
  std::vector<torch::jit::script::Module> models_gpu;
  RuntimeConfig opts;
  InferenceTask task(nullptr, job, &model_cpu, &models_gpu, &opts);

  const size_t num_inputs = InferLimits::MaxInputs + 1;
  EXPECT_THROW(task.check_limits(num_inputs), InferenceExecutionException);
}

TEST(InferenceTaskLimits, CheckLimitsTooManyGpuModels)
{
  auto job = std::make_shared<InferenceJob>();
  torch::jit::script::Module model_cpu{"m"};
  std::vector<torch::jit::script::Module> models_gpu;
  for (size_t i = 0; i < InferLimits::MaxModelsGPU + 1; ++i) {
    models_gpu.emplace_back(
        torch::jit::script::Module{std::string{"m"} + std::to_string(i)});
  }
  RuntimeConfig opts;
  InferenceTask task(nullptr, job, &model_cpu, &models_gpu, &opts);

  EXPECT_THROW(task.check_limits(1), TooManyGpuModelsException);
}

TEST(InferenceTaskLimits, FillInputLayoutTooManyDims)
{
  auto job = std::make_shared<InferenceJob>();
  std::vector<int64_t> dims(InferLimits::MaxDims + 1, 1);
  auto tensor = torch::ones(dims);
  job->set_input_tensors({tensor});
  job->set_input_types({tensor.scalar_type()});

  torch::jit::script::Module model_cpu{"m"};
  std::vector<torch::jit::script::Module> models_gpu;
  RuntimeConfig opts;
  InferenceTask task(nullptr, job, &model_cpu, &models_gpu, &opts);

  auto params = std::make_shared<InferenceParams>();
  EXPECT_THROW(task.fill_input_layout(params, 1), InferenceExecutionException);
}

TEST(InferenceTaskLimits, AssignFixedWorkerNegativeThrows)
{
  auto job = std::make_shared<InferenceJob>();
  job->set_fixed_worker_id(-1);

  torch::jit::script::Module model_cpu{"m"};
  std::vector<torch::jit::script::Module> models_gpu;
  RuntimeConfig opts;
  InferenceTask task(nullptr, job, &model_cpu, &models_gpu, &opts);

  starpu_task task_struct{};
  EXPECT_THROW(
      task.assign_fixed_worker_if_needed(&task_struct), std::invalid_argument);
}
