#include <gtest/gtest.h>

#include <chrono>
#include <vector>

#include "core/inference_runner.hpp"

using namespace starpu_server;

TEST(InferenceRunner, MakeShutdownJob)
{
  auto job = InferenceJob::make_shutdown_job();
  ASSERT_NE(job, nullptr);
  EXPECT_TRUE(job->is_shutdown());
}

TEST(InferenceJob, SettersAndGettersAndCallback)
{
  // Create dummy tensors and types
  std::vector<torch::Tensor> inputs{torch::ones({2, 2})};
  std::vector<at::ScalarType> types{at::kFloat};
  std::vector<torch::Tensor> outputs{torch::zeros({2, 2})};

  auto job = std::make_shared<InferenceJob>();
  job->set_job_id(7);
  job->set_input_tensors(inputs);
  job->set_input_types(types);
  job->set_outputs_tensors(outputs);
  job->set_fixed_worker_id(3);
  auto start = std::chrono::high_resolution_clock::now();
  job->set_start_time(start);

  bool callback_called = false;
  std::vector<torch::Tensor> cb_tensors;
  double cb_latency = 0.0;
  job->set_on_complete([&](std::vector<torch::Tensor> t, double latency) {
    callback_called = true;
    cb_tensors = std::move(t);
    cb_latency = latency;
  });

  // Verify getters
  EXPECT_EQ(job->get_job_id(), 7);
  ASSERT_EQ(job->get_input_tensors().size(), 1);
  EXPECT_TRUE(job->get_input_tensors()[0].equal(inputs[0]));
  ASSERT_EQ(job->get_input_types().size(), 1);
  EXPECT_EQ(job->get_input_types()[0], at::kFloat);
  ASSERT_EQ(job->get_output_tensors().size(), 1);
  EXPECT_TRUE(job->get_output_tensors()[0].equal(outputs[0]));
  EXPECT_EQ(job->get_start_time(), start);
  ASSERT_TRUE(job->get_fixed_worker_id().has_value());
  EXPECT_EQ(job->get_fixed_worker_id().value(), 3);
  ASSERT_TRUE(job->has_on_complete());

  // Trigger callback and verify
  job->get_on_complete()(job->get_output_tensors(), 123.0);
  EXPECT_TRUE(callback_called);
  ASSERT_EQ(cb_tensors.size(), 1);
  EXPECT_TRUE(cb_tensors[0].equal(outputs[0]));
  EXPECT_DOUBLE_EQ(cb_latency, 123.0);
}
