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
  std::chrono::high_resolution_clock::time_point time_point{};
  auto formatted = format_timestamp(time_point);
  EXPECT_EQ(formatted, "00:00:00.000");
}

TEST(TimeUtils, FormatTimestampKnownTime)
{
  SetUtcTimezone();
  using namespace std::chrono;
  constexpr year kYear = 2023y;
  constexpr month kMonth = January;
  constexpr day kDay{2};
  constexpr hours kHours{3};
  constexpr minutes kMinutes{4};
  constexpr seconds kSeconds{5};
  constexpr milliseconds kMillis{6};
  sys_days days = kYear / kMonth / kDay;
  sys_time<milliseconds> sys_tp = days + kHours + kMinutes + kSeconds + kMillis;
  auto hr_tp = time_point_cast<high_resolution_clock::duration>(sys_tp);
  auto formatted = format_timestamp(hr_tp);
  EXPECT_EQ(formatted, "03:04:05.006");
}
