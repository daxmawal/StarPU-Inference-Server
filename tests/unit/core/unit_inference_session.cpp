#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

#define private public
#include "core/inference_session.hpp"
#undef private

#include "core/inference_runner.hpp"
#include "core/starpu_setup.hpp"
#include "test_helpers.hpp"
#include "utils/runtime_config.hpp"

namespace starpu_server {

TEST(InferenceSession, LaunchThreadsThrowsWhenWorkerMissing)
{
  RuntimeConfig opts;
  StarPUSetup starpu(opts);
  InferenceSession session(opts, starpu);

  EXPECT_THROW(session.launch_threads(), std::logic_error);
}

TEST(InferenceSession, ReportLatencyStatsLogsWhenVerbosityAllows)
{
  RuntimeConfig opts;
  opts.models.resize(1);
  opts.models[0].inputs = {{"input0", {1}, at::kFloat}};
  opts.models[0].outputs = {{"output0", {1}, at::kFloat}};
  opts.verbosity = VerbosityLevel::Stats;

  StarPUSetup starpu(opts);
  InferenceSession session(opts, starpu);

  InferenceResult result{};
  result.latency_ms = 1.0;
  session.results_.push_back(result);

  CaptureStream capture{std::cout};
  session.report_latency_stats();
  EXPECT_NE(capture.str().find("Latency stats"), std::string::npos);
}

TEST(InferenceSession, ReportLatencyStatsSilentWhenVerbosityTooLow)
{
  RuntimeConfig opts;
  opts.models.resize(1);
  opts.models[0].inputs = {{"input0", {1}, at::kFloat}};
  opts.models[0].outputs = {{"output0", {1}, at::kFloat}};
  opts.verbosity = VerbosityLevel::Info;

  StarPUSetup starpu(opts);
  InferenceSession session(opts, starpu);

  InferenceResult result{};
  result.latency_ms = 2.0;
  session.results_.push_back(result);

  CaptureStream capture{std::cout};
  session.report_latency_stats();
  EXPECT_EQ(capture.str(), "");
}

TEST(InferenceSession, LaunchClientThreadStartsClientThread)
{
  RuntimeConfig opts;
  opts.batching.request_nb = 1;
  opts.models.resize(1);
  opts.models[0].inputs = {{"input0", {1}, at::kFloat}};
  opts.models[0].outputs = {{"output0", {1}, at::kFloat}};

  StarPUSetup starpu(opts);

  bool client_routine_called = false;
  InferenceSession::ClientRoutine client_routine =
      [&client_routine_called](
          InferenceQueue&, const RuntimeConfig&,
          const std::vector<torch::Tensor>&,
          int) { client_routine_called = true; };

  InferenceSession session(opts, starpu, client_routine);

  ASSERT_FALSE(session.client_thread_.joinable());
  session.launch_client_thread();
  EXPECT_TRUE(session.client_thread_.joinable());

  if (session.client_thread_.joinable()) {
    session.client_thread_.join();
  }
  EXPECT_TRUE(client_routine_called);
}

TEST(InferenceSession, AwaitCompletionWaitsForCompletion)
{
  RuntimeConfig opts;
  opts.batching.request_nb = 2;
  opts.models.resize(1);
  opts.models[0].inputs = {{"input0", {1}, at::kFloat}};
  opts.models[0].outputs = {{"output0", {1}, at::kFloat}};

  StarPUSetup starpu(opts);
  InferenceSession session(opts, starpu);

  auto incrementer_thread = std::jthread([&session]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    session.completed_jobs_.store(1, std::memory_order_release);
    session.all_done_cv_.notify_one();

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    session.completed_jobs_.store(2, std::memory_order_release);
    session.all_done_cv_.notify_one();
  });

  session.await_completion();

  EXPECT_GE(session.completed_jobs_.load(), opts.batching.request_nb);
}

TEST(InferenceSession, ProcessResultsCallsDetailProcessResults)
{
  RuntimeConfig opts;
  opts.models.resize(1);
  opts.models[0].inputs = {{"input0", {1}, at::kFloat}};
  opts.models[0].outputs = {{"output0", {1}, at::kFloat}};

  StarPUSetup starpu(opts);
  InferenceSession session(opts, starpu);

  InferenceResult result{};
  result.latency_ms = 1.5;
  session.results_.push_back(result);

  EXPECT_NO_THROW(session.process_results());
}

}  // namespace starpu_server
