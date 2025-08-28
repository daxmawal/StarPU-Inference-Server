#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <string>

#include "utils/time_utils.hpp"

using starpu_server::time_utils::format_timestamp;

namespace {

void
SetUtcTimezone()
{
  setenv("TZ", "UTC", 1);
  tzset();
}

}  // namespace

TEST(TimeUtils, FormatTimestampEpoch)
{
  SetUtcTimezone();
  std::chrono::high_resolution_clock::time_point tp{};
  auto formatted = format_timestamp(tp);
  EXPECT_EQ(formatted, "00:00:00.000");
}

TEST(TimeUtils, FormatTimestampKnownTime)
{
  SetUtcTimezone();
  using namespace std::chrono;
  sys_days days = 2023y / January / 2;
  sys_time<milliseconds> sys_tp =
      days + hours{3} + minutes{4} + seconds{5} + milliseconds{6};
  auto hr_tp = time_point_cast<high_resolution_clock::duration>(sys_tp);
  auto formatted = format_timestamp(hr_tp);
  EXPECT_EQ(formatted, "03:04:05.006");
}
