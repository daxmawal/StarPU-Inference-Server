#include <gtest/gtest.h>

#include <cstdlib>
#include <sstream>
#include <utility>

#include "utils/logger.hpp"

using namespace starpu_server;

TEST(Logger, LogsWhenLevelEnabled)
{
  std::ostringstream oss;
  auto* old_buf = std::cout.rdbuf(oss.rdbuf());

  log_verbose(VerbosityLevel::Info, VerbosityLevel::Info, "hello");

  std::cout.rdbuf(old_buf);
  auto [color, label] = verbosity_style(VerbosityLevel::Info);
  EXPECT_EQ(oss.str(), std::string(color) + label + "hello\033[0m\n");
}

TEST(Logger, NoOutputWhenLevelTooLow)
{
  std::ostringstream oss;
  auto* old_buf = std::cout.rdbuf(oss.rdbuf());

  log_verbose(VerbosityLevel::Stats, VerbosityLevel::Info, "hidden");

  std::cout.rdbuf(old_buf);
  EXPECT_EQ(oss.str(), "");
}

TEST(Logger, LogWarning)
{
  std::ostringstream oss;
  auto* old_buf = std::cerr.rdbuf(oss.rdbuf());

  log_warning("msg");

  std::cerr.rdbuf(old_buf);
  EXPECT_EQ(oss.str(), "\033[1;33m[WARNING] msg\033[0m\n");
}

TEST(Logger, LogError)
{
  std::ostringstream oss;
  auto* old_buf = std::cerr.rdbuf(oss.rdbuf());

  log_error("err");

  std::cerr.rdbuf(old_buf);
  EXPECT_EQ(oss.str(), "\033[1;31m[ERROR] err\033[0m\n");
}

TEST(Logger, LogFatalDeath)
{
  EXPECT_DEATH({ log_fatal("boom"); }, "boom");
}