#include <gtest/gtest.h>
#include <starpu.h>
#include <torch/script.h>

#include "core/inference_task.hpp"
#include "utils/exceptions.hpp"

TEST(InferenceTaskErrors, SubmitNullJob)
{
  auto job = std::shared_ptr<starpu_server::InferenceJob>(nullptr);
  torch::jit::script::Module model_cpu;
  std::vector<torch::jit::script::Module> models_gpu;
  starpu_server::RuntimeConfig opts;
  starpu_server::InferenceTask task(
      nullptr, job, &model_cpu, &models_gpu, &opts);
  EXPECT_THROW(task.submit(), starpu_server::InvalidInferenceJobException);
}

TEST(InferenceTaskErrors, RecordAndRunCompletionCallbackNoCallback)
{
  auto job = std::make_shared<starpu_server::InferenceJob>();
  std::vector<torch::Tensor> outputs = {torch::tensor({1})};
  job->set_outputs_tensors(outputs);
  const auto start = std::chrono::high_resolution_clock::now();
  const auto end = start + std::chrono::milliseconds(5);
  job->set_start_time(start);
  starpu_server::RuntimeConfig opts;
  starpu_server::InferenceCallbackContext ctx(job, nullptr, &opts, 0, {}, {});
  ASSERT_NO_THROW(
      starpu_server::InferenceTask::record_and_run_completion_callback(
          &ctx, end));
  EXPECT_EQ(job->timing_info().callback_end_time, end);
  EXPECT_FALSE(job->has_on_complete());
}

TEST(InferenceTaskErrors, CheckLimitsTooManyInputs)
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

TEST(InferenceTaskErrors, CheckLimitsTooManyGpuModels)
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
