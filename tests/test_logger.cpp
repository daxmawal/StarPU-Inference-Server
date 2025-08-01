#include <gtest/gtest.h>

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