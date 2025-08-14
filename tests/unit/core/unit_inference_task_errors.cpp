#include <gtest/gtest.h>
#include <starpu.h>
#include <torch/script.h>

#include "core/inference_task.hpp"
#include "utils/exceptions.hpp"

TEST(InferenceTask_Unit, RecordAndRunCompletionCallbackNoCallback)
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
