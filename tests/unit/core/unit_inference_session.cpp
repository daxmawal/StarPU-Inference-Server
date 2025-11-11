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

}  // namespace starpu_server
