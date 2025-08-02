#include <gtest/gtest.h>
#include <starpu.h>
#include <torch/script.h>

#include "core/inference_task.hpp"

using namespace starpu_server;

TEST(InferenceTaskUtils, RecordAndRunCompletionCallback)
{
  auto job = std::make_shared<InferenceJob>();
  std::vector<torch::Tensor> outputs = {torch::tensor({1})};
  job->set_outputs_tensors(outputs);

  bool called = false;
  std::vector<torch::Tensor> results_arg;
  double latency_arg = -1.0;

  job->set_on_complete(
      [&called, &results_arg, &latency_arg](
          const std::vector<torch::Tensor>& results, double latency) {
        called = true;
        results_arg = results;
        latency_arg = latency;
      });

  const auto start = std::chrono::high_resolution_clock::now();
  const auto end = start + std::chrono::milliseconds(5);
  job->set_start_time(start);

  RuntimeConfig opts;
  InferenceCallbackContext ctx(job, nullptr, &opts, 0, {}, {});

  InferenceTask::record_and_run_completion_callback(&ctx, end);

  EXPECT_TRUE(called);
  ASSERT_EQ(results_arg.size(), outputs.size());
  EXPECT_TRUE(torch::equal(results_arg[0], outputs[0]));

  const double expected_latency =
      std::chrono::duration<double, std::milli>(end - start).count();
  EXPECT_DOUBLE_EQ(latency_arg, expected_latency);
}
