#include <gtest/gtest.h>

#include <format>

#include "../test_helpers.hpp"
#include "utils/client_utils.hpp"
#include "utils/logger.hpp"
#include "utils/time_utils.hpp"

TEST(ClientUtils, PickRandomInputDeterministic)
{
  std::vector<std::vector<torch::Tensor>> pool;
  for (int i = 0; i < 3; ++i) {
    pool.push_back(
        {torch::tensor({i}, torch::TensorOptions().dtype(torch::kInt))});
  }
  std::mt19937 rng_a(123);
  std::mt19937 rng_b(123);
  for (int i = 0; i < 5; ++i) {
    const auto& chosen =
        starpu_server::client_utils::pick_random_input(pool, rng_a);
    int expected_idx =
        std::uniform_int_distribution<int>(0, pool.size() - 1)(rng_b);
    ASSERT_EQ(&chosen, &pool[expected_idx]);
    ASSERT_EQ(chosen.size(), 1u);
    EXPECT_EQ(chosen[0].item<int>(), expected_idx);
  }
}

TEST(ClientUtils, CreateJobProducesExpectedFields)
{
  std::vector<torch::Tensor> inputs = {torch::empty({}), torch::empty({})};
  std::vector<torch::Tensor> outputs_ref = {torch::empty({}), torch::empty({})};
  auto job = starpu_server::client_utils::create_job(inputs, outputs_ref, 7);
  ASSERT_EQ(job->get_job_id(), 7);
  ASSERT_EQ(job->get_input_tensors().size(), inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    EXPECT_TRUE(job->get_input_tensors()[i].defined());
    EXPECT_EQ(job->get_input_tensors()[i].sizes(), inputs[i].sizes());
    EXPECT_EQ(job->get_input_types()[i], inputs[i].scalar_type());
  }
  ASSERT_EQ(job->get_output_tensors().size(), outputs_ref.size());
  for (size_t i = 0; i < outputs_ref.size(); ++i) {
    EXPECT_EQ(job->get_output_tensors()[i].sizes(), outputs_ref[i].sizes());
    EXPECT_EQ(job->get_output_tensors()[i].dtype(), outputs_ref[i].dtype());
  }
  EXPECT_EQ(job->get_start_time(), job->timing_info().enqueued_time);
  EXPECT_GT(job->get_start_time().time_since_epoch().count(), 0);
}

TEST(ClientUtils, PreGenerateInputsProducesValidTensors)
{
  starpu_server::RuntimeConfig opts;
  opts.input_shapes = {{2, 3}, {1}};
  opts.input_types = {at::kFloat, at::kInt};
  const size_t N = 3;
  auto batches = starpu_server::client_utils::pre_generate_inputs(opts, N);
  ASSERT_EQ(batches.size(), N);
  for (const auto& tensors : batches) {
    ASSERT_EQ(tensors.size(), opts.input_shapes.size());
    EXPECT_EQ(tensors[0].sizes(), (torch::IntArrayRef{2, 3}));
    EXPECT_EQ(tensors[0].dtype(), at::kFloat);
    EXPECT_EQ(tensors[1].sizes(), (torch::IntArrayRef{1}));
    EXPECT_EQ(tensors[1].dtype(), at::kInt);
  }
}

TEST(ClientUtils, LogJobEnqueuedPrintsTraceMessage)
{
  starpu_server::RuntimeConfig opts;
  opts.verbosity = starpu_server::VerbosityLevel::Trace;
  const int job_id = 2;
  const int iterations = 5;
  auto now = std::chrono::high_resolution_clock::now();
  starpu_server::CaptureStream capture{std::cout};
  starpu_server::client_utils::log_job_enqueued(opts, job_id, iterations, now);
  auto timestamp = starpu_server::time_utils::format_timestamp(now);
  std::string expected = expected_log_line(
      starpu_server::VerbosityLevel::Trace,
      std::format(
          "[Inference] Job ID {} Iteration {}/{} Enqueued at {}", job_id,
          job_id + 1, iterations, timestamp));
  EXPECT_EQ(capture.str(), expected);
}
