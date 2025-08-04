#include <gtest/gtest.h>
#include <starpu.h>
#include <torch/script.h>

#include "core/inference_task.hpp"
#include "utils/exceptions.hpp"

using namespace starpu_server;

TEST(InferenceTaskErrors, SubmitNullJob)
{
  auto job = std::shared_ptr<InferenceJob>(nullptr);
  torch::jit::script::Module model_cpu;
  std::vector<torch::jit::script::Module> models_gpu;
  RuntimeConfig opts;
  InferenceTask task(nullptr, job, &model_cpu, &models_gpu, &opts);
  EXPECT_THROW(task.submit(), InvalidInferenceJobException);
}

TEST(InferenceTaskErrors, SafeRegisterTensorVectorUndefined)
{
  torch::Tensor undef;
  EXPECT_THROW(
      InferenceTask::safe_register_tensor_vector(undef, "x"),
      StarPURegistrationException);
}

TEST(InferenceTaskErrors, AssignFixedWorkerInvalid)
{
  auto job = std::make_shared<InferenceJob>();
  job->set_fixed_worker_id(-1);
  torch::jit::script::Module model_cpu;
  std::vector<torch::jit::script::Module> models_gpu;
  RuntimeConfig opts;
  InferenceTask task(nullptr, job, &model_cpu, &models_gpu, &opts);
  starpu_task task_struct{};
  EXPECT_THROW(
      task.assign_fixed_worker_if_needed(&task_struct), std::invalid_argument);
}

TEST(InferenceTaskErrors, RecordAndRunCompletionCallbackNoCallback)
{
  auto job = std::make_shared<InferenceJob>();
  std::vector<torch::Tensor> outputs = {torch::tensor({1})};
  job->set_outputs_tensors(outputs);
  const auto start = std::chrono::high_resolution_clock::now();
  const auto end = start + std::chrono::milliseconds(5);
  job->set_start_time(start);

  RuntimeConfig opts;
  InferenceCallbackContext ctx(job, nullptr, &opts, 0, {}, {});

  ASSERT_NO_THROW(InferenceTask::record_and_run_completion_callback(&ctx, end));
  EXPECT_EQ(job->timing_info().callback_end_time, end);
  EXPECT_FALSE(job->has_on_complete());
}

TEST(InferenceTaskErrors, CheckLimitsTooManyInputs)
{
  auto job = std::make_shared<InferenceJob>();
  torch::jit::script::Module model_cpu{"m"};
  std::vector<torch::jit::script::Module> models_gpu;
  RuntimeConfig opts;
  InferenceTask task(nullptr, job, &model_cpu, &models_gpu, &opts);
  const size_t num_inputs = InferLimits::MaxInputs + 1;
  EXPECT_THROW(task.check_limits(num_inputs), InferenceExecutionException);
}

TEST(InferenceTaskErrors, CheckLimitsTooManyGpuModels)
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
