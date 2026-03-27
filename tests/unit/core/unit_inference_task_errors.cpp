#include <gtest/gtest.h>
#include <starpu.h>
#include <torch/script.h>

#include <chrono>
#include <memory>
#include <vector>

#include "core/inference_task.hpp"
#include "test_inference_task.hpp"
#include "utils/exceptions.hpp"

TEST(InferenceTask_Unit, RecordAndRunCompletionCallbackNoCallback)
{
  auto job = std::make_shared<starpu_server::InferenceJob>();
  std::vector<torch::Tensor> outputs = {torch::tensor({1})};
  job->set_output_tensors(outputs);

  const auto start = starpu_server::MonotonicClock::now();
  const auto end = start + std::chrono::milliseconds(5);
  job->set_start_time(start);

  auto ctx = make_callback_context(job);

  ASSERT_NO_THROW(
      starpu_server::InferenceTask::record_and_run_completion_callback(
          ctx.get(), end));
  EXPECT_EQ(job->timing_info().callback_end_time, end);
  EXPECT_FALSE(job->completion().has_on_complete());
}

TEST(InferenceTask_Unit, RecordAndRunCompletionCallbackNullJob)
{
  std::shared_ptr<starpu_server::InferenceJob> job;
  auto ctx = make_callback_context(job);

  const auto end = starpu_server::MonotonicClock::now();

  ASSERT_NO_THROW(
      starpu_server::InferenceTask::record_and_run_completion_callback(
          ctx.get(), end));
  EXPECT_EQ(ctx->job, nullptr);
}

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
