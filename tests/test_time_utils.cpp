#include <gtest/gtest.h>

#include <chrono>
#include <regex>

#include "utils/time_utils.hpp"

using namespace starpu_server::time_utils;

TEST(TimeUtils, FormatTimestamp)
{
  auto now = std::chrono::high_resolution_clock::now();
  std::string ts = format_timestamp(now);
  std::regex pattern("^[0-9]{2}:[0-9]{2}:[0-9]{2}\\.[0-9]{3}$");
  EXPECT_TRUE(std::regex_match(ts, pattern));
}