#include <gtest/gtest.h>

#include <chrono>
#include <ctime>
#include <regex>
#include <string>

#include "utils/time_utils.hpp"

TEST(TimeUtils, FormatTimestamp_FormatRegex)
{
  auto now = std::chrono::high_resolution_clock::now();
  std::string ts = starpu_server::time_utils::format_timestamp(now);
  std::regex pattern("^[0-9]{2}:[0-9]{2}:[0-9]{2}\\.[0-9]{3}$");
  EXPECT_TRUE(std::regex_match(ts, pattern));
}

TEST(TimeUtils, FormatTimestamp_KnownTime)
{
  std::tm tm = {};
  tm.tm_year = 123;  // 2023 - 1900
  tm.tm_mon = 0;     // Janvier
  tm.tm_mday = 1;
  tm.tm_hour = 12;
  tm.tm_min = 34;
  tm.tm_sec = 56;
  std::time_t t = std::mktime(&tm);

  auto base_time = std::chrono::system_clock::from_time_t(t);
  auto time_point =
      time_point_cast<std::chrono::high_resolution_clock::duration>(base_time) +
      std::chrono::milliseconds(789);

  std::string ts = starpu_server::time_utils::format_timestamp(time_point);

  EXPECT_TRUE(ts.ends_with(".789"));
}

TEST(TimeUtils, FormatTimestamp_MillisecondBoundaries)
{
  std::time_t now = std::time(nullptr);
  auto base_time = std::chrono::system_clock::from_time_t(now);
  auto tp000 =
      time_point_cast<std::chrono::high_resolution_clock::duration>(base_time) +
      std::chrono::milliseconds(0);
  auto tp999 = tp000 + std::chrono::milliseconds(999);

  std::string ts000 = starpu_server::time_utils::format_timestamp(tp000);
  std::string ts999 = starpu_server::time_utils::format_timestamp(tp999);

  EXPECT_TRUE(ts000.ends_with(".000"));
  EXPECT_TRUE(ts999.ends_with(".999"));
}
