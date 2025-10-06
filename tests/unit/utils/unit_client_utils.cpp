#include <gtest/gtest.h>

#include <chrono>
#include <ctime>
#include <format>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#include "test_helpers.hpp"
#include "utils/client_utils.hpp"
#include "utils/logger.hpp"
#include "utils/time_utils.hpp"

namespace {
constexpr int kSeed123 = 123;
constexpr int kSeed42 = 42;
constexpr int kInitPool3 = 3;
constexpr int kInitPool4 = 4;
constexpr int kPickTrials = 5;
constexpr int kWithinBoundsTrials = 100;
constexpr int kJobId = 7;

constexpr int kKnownYear = 2023;
constexpr int kKnownMonth = 1;
constexpr int kKnownDay = 1;
constexpr int kKnownHour = 12;
constexpr int kKnownMinute = 34;
constexpr int kKnownSecond = 56;
constexpr int kMillis789 = 789;
constexpr int kMillis999 = 999;
constexpr int kTmYearBase = 1900;

inline auto
JobHasExpectedInputs(
    const std::shared_ptr<starpu_server::InferenceJob>& job,
    const std::vector<torch::Tensor>& inputs) -> ::testing::AssertionResult
{
  if (job->get_input_tensors().size() != inputs.size()) {
    return ::testing::AssertionFailure()
           << "Input tensor count mismatch: got "
           << job->get_input_tensors().size() << " expected " << inputs.size();
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& input_tensor = job->get_input_tensors()[i];
    if (!input_tensor.defined()) {
      return ::testing::AssertionFailure()
             << "Input tensor " << i << " is undefined";
    }
    if (input_tensor.sizes() != inputs[i].sizes()) {
      return ::testing::AssertionFailure()
             << "Input tensor sizes mismatch at index " << i;
    }
    if (job->get_input_types()[i] != inputs[i].scalar_type()) {
      return ::testing::AssertionFailure()
             << "Input dtype mismatch at index " << i;
    }
  }
  return ::testing::AssertionSuccess();
}

inline auto
JobHasExpectedOutputs(
    const std::shared_ptr<starpu_server::InferenceJob>& job,
    const std::vector<torch::Tensor>& outputs_ref) -> ::testing::AssertionResult
{
  if (job->get_output_tensors().size() != outputs_ref.size()) {
    return ::testing::AssertionFailure() << "Output tensor count mismatch: got "
                                         << job->get_output_tensors().size()
                                         << " expected " << outputs_ref.size();
  }
  for (size_t i = 0; i < outputs_ref.size(); ++i) {
    if (job->get_output_tensors()[i].sizes() != outputs_ref[i].sizes()) {
      return ::testing::AssertionFailure()
             << "Output tensor sizes mismatch at index " << i;
    }
    if (job->get_output_tensors()[i].dtype() != outputs_ref[i].dtype()) {
      return ::testing::AssertionFailure()
             << "Output dtype mismatch at index " << i;
    }
  }
  return ::testing::AssertionSuccess();
}

inline auto
BatchesMatchConfig(
    const std::vector<std::vector<torch::Tensor>>& batches,
    const starpu_server::RuntimeConfig& opts) -> ::testing::AssertionResult
{
  for (const auto& tensors : batches) {
    if (tensors.size() != opts.models[0].inputs.size()) {
      return ::testing::AssertionFailure()
             << "Batch size mismatch: got " << tensors.size() << " expected "
             << opts.models[0].inputs.size();
    }
    if (tensors[0].sizes() != torch::IntArrayRef{2, 3} ||
        tensors[0].dtype() != at::kFloat) {
      return ::testing::AssertionFailure() << "First tensor mismatch";
    }
    if (tensors[1].sizes() != torch::IntArrayRef{1} ||
        tensors[1].dtype() != at::kInt) {
      return ::testing::AssertionFailure() << "Second tensor mismatch";
    }
  }
  return ::testing::AssertionSuccess();
}
}  // namespace

TEST(ClientUtils, PickRandomInputDeterministic)
{
  std::vector<std::vector<torch::Tensor>> pool;
  pool.reserve(kInitPool3);
  for (int i = 0; i < kInitPool3; ++i) {
    pool.push_back(
        {torch::tensor({i}, torch::TensorOptions().dtype(torch::kInt))});
  }
  std::mt19937 rng_a(kSeed123);
  std::mt19937 rng_b(kSeed123);
  for (int i = 0; i < kPickTrials; ++i) {
    const auto& chosen =
        starpu_server::client_utils::pick_random_input(pool, rng_a);
    int expected_idx = std::uniform_int_distribution<int>(
        0, static_cast<int>(pool.size() - 1))(rng_b);
    ASSERT_EQ(&chosen, &pool[expected_idx]);
    ASSERT_EQ(chosen.size(), 1U);
    EXPECT_EQ(chosen[0].item<int>(), expected_idx);
  }
}

TEST(ClientUtils, PickRandomInputWithinBounds)
{
  std::vector<std::vector<torch::Tensor>> pool;
  pool.reserve(kInitPool4);
  for (int i = 0; i < kInitPool4; ++i) {
    pool.push_back(
        {torch::tensor({i}, torch::TensorOptions().dtype(torch::kInt))});
  }
  std::mt19937 rng(kSeed123);
  for (int i = 0; i < kWithinBoundsTrials; ++i) {
    const auto& chosen =
        starpu_server::client_utils::pick_random_input(pool, rng);
    bool found = false;
    for (const auto& item : pool) {
      if (&item == &chosen) {
        found = true;
        break;
      }
    }
    EXPECT_TRUE(found);
  }
}

TEST(ClientUtils, PickRandomInputEmptyPoolThrows)
{
  std::vector<std::vector<torch::Tensor>> pool;
  std::mt19937 rng(kSeed42);
  EXPECT_THROW(
      starpu_server::client_utils::pick_random_input(pool, rng),
      std::invalid_argument);
}

TEST(ClientUtils, CreateJobProducesExpectedFields)
{
  std::vector<torch::Tensor> inputs = {torch::empty({}), torch::empty({})};
  std::vector<torch::Tensor> outputs_ref = {torch::empty({}), torch::empty({})};
  auto job =
      starpu_server::client_utils::create_job(inputs, outputs_ref, kJobId);
  ASSERT_EQ(job->get_job_id(), kJobId);
  EXPECT_TRUE(JobHasExpectedInputs(job, inputs));
  EXPECT_TRUE(JobHasExpectedOutputs(job, outputs_ref));
  EXPECT_LE(job->get_start_time(), job->timing_info().enqueued_time);
  EXPECT_GT(job->get_start_time().time_since_epoch().count(), 0);
}

TEST(ClientUtils, PreGenerateInputsProducesValidTensors)
{
  starpu_server::RuntimeConfig opts;
  opts.models.resize(1);
  opts.models[0].inputs = {
      {"input0", {2, 3}, at::kFloat},
      {"input1", {1}, at::kInt},
  };
  const size_t batch_size = 3;
  auto batches =
      starpu_server::client_utils::pre_generate_inputs(opts, batch_size);
  ASSERT_EQ(batches.size(), batch_size);
  EXPECT_TRUE(BatchesMatchConfig(batches, opts));
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
          "[Inference] Request ID {} Iteration {}/{} Enqueued at {}", job_id,
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
  time.tm_year = kKnownYear - kTmYearBase;
  time.tm_mon = kKnownMonth - 1;
  time.tm_mday = kKnownDay;
  time.tm_hour = kKnownHour;
  time.tm_min = kKnownMinute;
  time.tm_sec = kKnownSecond;

  using namespace std::chrono;
  const auto year_val = year{time.tm_year + kTmYearBase};
  const auto month_val = month{static_cast<unsigned>(time.tm_mon + 1)};
  const auto day_val = day{static_cast<unsigned>(time.tm_mday)};

  local_time<seconds> local_time = local_days(year_val / month_val / day_val) +
                                   hours{time.tm_hour} + minutes{time.tm_min} +
                                   seconds{time.tm_sec};

  sys_time<seconds> base_time = current_zone()->to_sys(local_time);

  auto time_point =
      time_point_cast<std::chrono::high_resolution_clock::duration>(base_time) +
      std::chrono::milliseconds(kMillis789);

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
  auto tp999 = tp000 + std::chrono::milliseconds(kMillis999);
  std::string ts000 = starpu_server::time_utils::format_timestamp(tp000);
  std::string ts999 = starpu_server::time_utils::format_timestamp(tp999);
  EXPECT_TRUE(ts000.ends_with(".000"));
  EXPECT_TRUE(ts999.ends_with(".999"));
}
