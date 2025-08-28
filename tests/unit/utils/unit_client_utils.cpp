#include <gtest/gtest.h>

#include <chrono>
#include <ctime>
#include <format>
#include <regex>
#include <stdexcept>
#include <string>

#include "test_helpers.hpp"
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
    ASSERT_EQ(chosen.size(), 1U);
    EXPECT_EQ(chosen[0].item<int>(), expected_idx);
  }
}

TEST(ClientUtils, PickRandomInputEmptyPoolThrows)
{
  std::vector<std::vector<torch::Tensor>> pool;
  std::mt19937 rng(42);
  EXPECT_THROW(
      starpu_server::client_utils::pick_random_input(pool, rng),
      std::invalid_argument);
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
  opts.inputs = {
      {"input0", {2, 3}, at::kFloat},
      {"input1", {1}, at::kInt},
  };
  const size_t batch_size = 3;
  auto batches =
      starpu_server::client_utils::pre_generate_inputs(opts, batch_size);
  ASSERT_EQ(batches.size(), batch_size);
  for (const auto& tensors : batches) {
    ASSERT_EQ(tensors.size(), opts.inputs.size());
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

TEST(TimeUtils, FormatTimestamp_FormatRegex)
{
  auto now = std::chrono::high_resolution_clock::now();
  std::string time = starpu_server::time_utils::format_timestamp(now);
  EXPECT_TRUE(std::regex_match(
      time, std::regex("^[0-9]{2}:[0-9]{2}:[0-9]{2}\\.[0-9]{3}$")));
}

TEST(TimeUtils, FormatTimestamp_KnownTime)
{
  std::tm time = {};
  time.tm_year = 123;  // 2023 - 1900
  time.tm_mon = 0;     // Janvier
  time.tm_mday = 1;
  time.tm_hour = 12;
  time.tm_min = 34;
  time.tm_sec = 56;

  using namespace std::chrono;
  const auto y = year{time.tm_year + 1900};
  const auto m = month{static_cast<unsigned>(time.tm_mon + 1)};
  const auto d = day{static_cast<unsigned>(time.tm_mday)};

  local_time<seconds> local_time = local_days(y / m / d) + hours{time.tm_hour} +
                                   minutes{time.tm_min} + seconds{time.tm_sec};

  sys_time<seconds> base_time = current_zone()->to_sys(local_time);

  auto time_point =
      time_point_cast<std::chrono::high_resolution_clock::duration>(base_time) +
      std::chrono::milliseconds(789);

  std::string time_stamp =
      starpu_server::time_utils::format_timestamp(time_point);
  EXPECT_TRUE(time_stamp.ends_with(".789"));
}

TEST(TimeUtils, FormatTimestamp_MillisecondBoundaries)
{
  auto base_time = std::chrono::time_point_cast<std::chrono::seconds>(
      std::chrono::system_clock::now());
  auto tp000 =
      time_point_cast<std::chrono::high_resolution_clock::duration>(base_time) +
      std::chrono::milliseconds(0);
  auto tp999 = tp000 + std::chrono::milliseconds(999);
  std::string ts000 = starpu_server::time_utils::format_timestamp(tp000);
  std::string ts999 = starpu_server::time_utils::format_timestamp(tp999);
  EXPECT_TRUE(ts000.ends_with(".000"));
  EXPECT_TRUE(ts999.ends_with(".999"));
}
