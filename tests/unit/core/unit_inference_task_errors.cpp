#include <gtest/gtest.h>
#include <starpu.h>
#include <torch/script.h>

#include "core/inference_task.hpp"
#include "test_inference_task.hpp"
#include "utils/exceptions.hpp"

TEST(InferenceTask_Unit, RecordAndRunCompletionCallbackNoCallback)
{
  auto job = std::make_shared<starpu_server::InferenceJob>();
  std::vector<torch::Tensor> outputs = {torch::tensor({1})};
  job->set_output_tensors(outputs);

  const auto start = std::chrono::high_resolution_clock::now();
  const auto end = start + std::chrono::milliseconds(5);
  job->set_start_time(start);

  starpu_server::RuntimeConfig opts;
  auto ctx = make_callback_context(job, &opts);

  ASSERT_NO_THROW(
      starpu_server::InferenceTask::record_and_run_completion_callback(
          ctx.get(), end));
  EXPECT_EQ(job->timing_info().callback_end_time, end);
  EXPECT_FALSE(job->has_on_complete());
}

TEST(InferenceTask_Unit, RecordAndRunCompletionCallbackNullJob)
{
  std::shared_ptr<starpu_server::InferenceJob> job;
  starpu_server::RuntimeConfig opts;
  auto ctx = make_callback_context(job, &opts);

  const auto end = std::chrono::high_resolution_clock::now();

  ASSERT_NO_THROW(
      starpu_server::InferenceTask::record_and_run_completion_callback(
          ctx.get(), end));
  EXPECT_EQ(ctx->job, nullptr);
}
