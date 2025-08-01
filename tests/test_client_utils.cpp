#include <gtest/gtest.h>

#include "utils/client_utils.hpp"

using namespace starpu_server;
using namespace starpu_server::client_utils;

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
    const auto& chosen = pick_random_input(pool, rng_a);
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

  auto job = create_job(inputs, outputs_ref, 7);

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