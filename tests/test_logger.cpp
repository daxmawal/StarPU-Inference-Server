#include <gtest/gtest.h>

#include <cstdlib>
#include <sstream>
#include <utility>

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

TEST(Logger, LogsWhenLevelEnabled)
{
  std::ostringstream oss;
  auto* old_buf = std::cout.rdbuf(oss.rdbuf());

  log_verbose(VerbosityLevel::Info, VerbosityLevel::Info, "hello");

  std::cout.rdbuf(old_buf);
  auto [color, label] = verbosity_style(VerbosityLevel::Info);
  EXPECT_EQ(oss.str(), std::string(color) + label + "hello\o{33}[0m\n");
}

TEST(Logger, NoOutputWhenLevelTooLow)
{
  std::ostringstream oss;
  auto* old_buf = std::cout.rdbuf(oss.rdbuf());

  log_verbose(VerbosityLevel::Stats, VerbosityLevel::Info, "hidden");

  std::cout.rdbuf(old_buf);
  EXPECT_EQ(oss.str(), "");
}

TEST(Logger, LogStatsLevel)
{
  std::ostringstream oss;
  auto* old_buf = std::cout.rdbuf(oss.rdbuf());

  log_verbose(VerbosityLevel::Stats, VerbosityLevel::Stats, "statline");

  std::cout.rdbuf(old_buf);
  auto [color, label] = verbosity_style(VerbosityLevel::Stats);
  EXPECT_EQ(oss.str(), std::string(color) + label + "statline\o{33}[0m\n");
}

TEST(Logger, LogDebugLevel)
{
  std::ostringstream oss;
  auto* old_buf = std::cout.rdbuf(oss.rdbuf());

  log_verbose(VerbosityLevel::Debug, VerbosityLevel::Debug, "debugline");

  std::cout.rdbuf(old_buf);
  auto [color, label] = verbosity_style(VerbosityLevel::Debug);
  EXPECT_EQ(oss.str(), std::string(color) + label + "debugline\o{33}[0m\n");
}

TEST(Logger, LogTraceLevel)
{
  std::ostringstream oss;
  auto* old_buf = std::cout.rdbuf(oss.rdbuf());

  log_verbose(VerbosityLevel::Trace, VerbosityLevel::Trace, "traceline");

  std::cout.rdbuf(old_buf);
  auto [color, label] = verbosity_style(VerbosityLevel::Trace);
  EXPECT_EQ(oss.str(), std::string(color) + label + "traceline\o{33}[0m\n");
}

TEST(Logger, LogInfoWrapper)
{
  std::ostringstream oss;
  auto* old_buf = std::cout.rdbuf(oss.rdbuf());

  log_info(VerbosityLevel::Info, "wrapped info");

  std::cout.rdbuf(old_buf);
  auto [color, label] = verbosity_style(VerbosityLevel::Info);
  EXPECT_EQ(oss.str(), std::string(color) + label + "wrapped info\o{33}[0m\n");
}

TEST(Logger, LogInfoWrapperBelowLevel)
{
  std::ostringstream oss;
  auto* old_buf = std::cout.rdbuf(oss.rdbuf());

  log_info(VerbosityLevel::Silent, "should not appear");

  std::cout.rdbuf(old_buf);
  EXPECT_EQ(oss.str(), "");
}

TEST(Logger, LogStatsWrapper)
{
  std::ostringstream oss;
  auto* old_buf = std::cout.rdbuf(oss.rdbuf());

  log_stats(VerbosityLevel::Stats, "wrapped stats");

  std::cout.rdbuf(old_buf);
  auto [color, label] = verbosity_style(VerbosityLevel::Stats);
  EXPECT_EQ(oss.str(), std::string(color) + label + "wrapped stats\o{33}[0m\n");
}

TEST(Logger, LogStatsWrapperBelowLevel)
{
  std::ostringstream oss;
  auto* old_buf = std::cout.rdbuf(oss.rdbuf());

  log_stats(VerbosityLevel::Info, "should not appear");

  std::cout.rdbuf(old_buf);
  EXPECT_EQ(oss.str(), "");
}

TEST(Logger, LogDebugWrapper)
{
  std::ostringstream oss;
  auto* old_buf = std::cout.rdbuf(oss.rdbuf());

  log_debug(VerbosityLevel::Debug, "wrapped debug");

  std::cout.rdbuf(old_buf);
  auto [color, label] = verbosity_style(VerbosityLevel::Debug);
  EXPECT_EQ(oss.str(), std::string(color) + label + "wrapped debug\o{33}[0m\n");
}

TEST(Logger, LogDebugWrapperBelowLevel)
{
  std::ostringstream oss;
  auto* old_buf = std::cout.rdbuf(oss.rdbuf());

  log_debug(VerbosityLevel::Info, "should not appear");

  std::cout.rdbuf(old_buf);
  EXPECT_EQ(oss.str(), "");
}

TEST(Logger, LogTraceWrapper)
{
  std::ostringstream oss;
  auto* old_buf = std::cout.rdbuf(oss.rdbuf());

  log_trace(VerbosityLevel::Trace, "wrapped trace");

  std::cout.rdbuf(old_buf);
  auto [color, label] = verbosity_style(VerbosityLevel::Trace);
  EXPECT_EQ(oss.str(), std::string(color) + label + "wrapped trace\o{33}[0m\n");
}

TEST(Logger, LogTraceWrapperBelowLevel)
{
  std::ostringstream oss;
  auto* old_buf = std::cout.rdbuf(oss.rdbuf());

  log_trace(VerbosityLevel::Info, "should not appear");

  std::cout.rdbuf(old_buf);
  EXPECT_EQ(oss.str(), "");
}

TEST(Logger, LogWarning)
{
  std::ostringstream oss;
  auto* old_buf = std::cerr.rdbuf(oss.rdbuf());

  log_warning("msg");

  std::cerr.rdbuf(old_buf);
  EXPECT_EQ(oss.str(), "\o{33}[1;33m[WARNING] msg\o{33}[0m\n");
}

TEST(Logger, LogError)
{
  std::ostringstream oss;
  auto* old_buf = std::cerr.rdbuf(oss.rdbuf());

  log_error("err");

  std::cerr.rdbuf(old_buf);
  EXPECT_EQ(oss.str(), "\o{33}[1;31m[ERROR] err\o{33}[0m\n");
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
