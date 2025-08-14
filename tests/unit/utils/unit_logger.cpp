#include <gtest/gtest.h>

#include <cstdlib>
#include <functional>
#include <memory>
#include <utility>

#include "core/inference_task.hpp"
#include "test_helpers.hpp"
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
  const auto& param = GetParam();
  CaptureStream capture{std::cout};
  log_verbose(param.level, param.level, param.message);
  EXPECT_EQ(capture.str(), expected_log_line(param.level, param.message));
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
  const auto& param = GetParam();
  CaptureStream capture{std::cout};
  param.fn(param.call_level, param.message);
  if (param.expect_output) {
    EXPECT_EQ(
        capture.str(), expected_log_line(param.inherent_level, param.message));
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
