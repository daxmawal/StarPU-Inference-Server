#include <gtest/gtest.h>

#include <cstdlib>
#include <functional>
#include <memory>
#include <utility>

#include "../test_helpers.hpp"
#include "core/inference_task.hpp"
#include "utils/exceptions.hpp"
#include "utils/logger.hpp"

using namespace starpu_server;

TEST(Logger, VerbosityStyle)
{
  using enum VerbosityLevel;
  auto check = [](VerbosityLevel level, const char* color, const char* label) {
    auto [got_color, got_label] = verbosity_style(level);
    EXPECT_STREQ(got_color, color);
    EXPECT_STREQ(got_label, label);
  };
  check(Info, "\o{33}[1;32m", "[INFO] ");
  check(Stats, "\o{33}[1;35m", "[STATS] ");
  check(Debug, "\o{33}[1;34m", "[DEBUG] ");
  check(Trace, "\o{33}[1;90m", "[TRACE] ");
  check(Silent, "", "");
  check(static_cast<VerbosityLevel>(255), "", "");
}

struct VerboseParam {
  VerbosityLevel level;
  const char* message;
};

class LogVerboseLevels : public ::testing::TestWithParam<VerboseParam> {};

TEST_P(LogVerboseLevels, LogsWhenEnabled)
{
  const auto& p = GetParam();
  CaptureStream capture{std::cout};
  log_verbose(p.level, p.level, p.message);
  EXPECT_EQ(capture.str(), expected_log_line(p.level, p.message));
}

INSTANTIATE_TEST_SUITE_P(
    Logger, LogVerboseLevels,
    ::testing::Values(
        VerboseParam{VerbosityLevel::Info, "hello"},
        VerboseParam{VerbosityLevel::Stats, "statline"},
        VerboseParam{VerbosityLevel::Debug, "debugline"},
        VerboseParam{VerbosityLevel::Trace, "traceline"}));

TEST(Logger, NoOutputWhenLevelTooLow)
{
  CaptureStream capture{std::cout};
  log_verbose(VerbosityLevel::Stats, VerbosityLevel::Info, "hidden");
  EXPECT_EQ(capture.str(), "");
}

using WrapperFn = void (*)(VerbosityLevel, const std::string&);
struct WrapperParam {
  WrapperFn fn;
  VerbosityLevel call_level;
  VerbosityLevel inherent_level;
  const char* message;
  bool expect_output;
};

class LogWrappers : public ::testing::TestWithParam<WrapperParam> {};

TEST_P(LogWrappers, ProducesExpectedOutput)
{
  const auto& p = GetParam();
  CaptureStream capture{std::cout};
  p.fn(p.call_level, p.message);
  if (p.expect_output) {
    EXPECT_EQ(capture.str(), expected_log_line(p.inherent_level, p.message));
  } else {
    EXPECT_EQ(capture.str(), "");
  }
}

INSTANTIATE_TEST_SUITE_P(
    Logger, LogWrappers,
    ::testing::Values(
        WrapperParam{
            log_info, VerbosityLevel::Info, VerbosityLevel::Info,
            "wrapped info", true},
        WrapperParam{
            log_info, VerbosityLevel::Silent, VerbosityLevel::Info,
            "should not appear", false},
        WrapperParam{
            log_stats, VerbosityLevel::Stats, VerbosityLevel::Stats,
            "wrapped stats", true},
        WrapperParam{
            log_stats, VerbosityLevel::Info, VerbosityLevel::Stats,
            "should not appear", false},
        WrapperParam{
            log_debug, VerbosityLevel::Debug, VerbosityLevel::Debug,
            "wrapped debug", true},
        WrapperParam{
            log_debug, VerbosityLevel::Info, VerbosityLevel::Debug,
            "should not appear", false},
        WrapperParam{
            log_trace, VerbosityLevel::Trace, VerbosityLevel::Trace,
            "wrapped trace", true},
        WrapperParam{
            log_trace, VerbosityLevel::Info, VerbosityLevel::Trace,
            "should not appear", false}));

TEST(Logger, LogWarning)
{
  CaptureStream capture{std::cerr};
  log_warning("msg");
  EXPECT_EQ(capture.str(), expected_log_line(WarningLevel, "msg"));
}

TEST(Logger, LogError)
{
  CaptureStream capture{std::cerr};
  log_error("err");
  EXPECT_EQ(capture.str(), expected_log_line(ErrorLevel, "err"));
}

TEST(Logger, LogFatalDeath)
{
  EXPECT_DEATH({ log_fatal("boom"); }, "\\[FATAL\\] boom");
}

TEST(Logger, LogFatalExit)
{
  EXPECT_EXIT(
      log_fatal("fatal test"), ::testing::KilledBySignal(SIGABRT),
      "fatal test");
}

struct ExceptionCase {
  std::function<std::unique_ptr<std::exception>()> make_exception;
  std::string expected;
};

class LogExceptionTest : public ::testing::TestWithParam<ExceptionCase> {};

TEST_P(LogExceptionTest, LogsExpectedMessage)
{
  auto param = GetParam();
  auto e = param.make_exception();
  CaptureStream capture{std::cerr};
  InferenceTask::log_exception("ctx", *e);
  EXPECT_EQ(capture.str(), param.expected);
}

INSTANTIATE_TEST_SUITE_P(
    ExceptionLogging, LogExceptionTest,
    ::testing::Values(
        ExceptionCase{
            []() {
              return std::make_unique<InferenceExecutionException>("exec");
            },
            expected_log_line(
                ErrorLevel, "InferenceExecutionException in ctx: exec")},
        ExceptionCase{
            []() {
              return std::make_unique<StarPUTaskSubmissionException>("sub");
            },
            expected_log_line(
                ErrorLevel, "StarPU submission error in ctx: sub")},
        ExceptionCase{
            []() { return std::make_unique<std::runtime_error>("boom"); },
            expected_log_line(ErrorLevel, "std::exception in ctx: boom")}));
