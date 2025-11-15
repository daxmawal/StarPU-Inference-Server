#include <gtest/gtest.h>

#include <filesystem>
#include <optional>
#include <sstream>

#include "monitoring/metrics.hpp"

using starpu_server::monitoring::detail::CpuTotals;
using starpu_server::monitoring::detail::make_cpu_usage_provider;
using starpu_server::monitoring::detail::read_total_cpu_times;

namespace {

TEST(CpuUsageProvider, ReturnsFalseWhenPathDoesNotExist)
{
  CpuTotals totals{};
  const std::filesystem::path missing_path{
      "/this/path/should/not/exist/for/cpu_times"};
  EXPECT_FALSE(read_total_cpu_times(missing_path, totals));

  auto provider = make_cpu_usage_provider([missing_path](CpuTotals& out) {
    return read_total_cpu_times(missing_path, out);
  });
  EXPECT_EQ(provider(), std::nullopt);
}

TEST(CpuUsageProvider, ReturnsFalseWhenFirstTokenIsNotCpu)
{
  CpuTotals totals{};
  std::istringstream input{"notcpu 1 2 3 4 5 6 7 8"};
  EXPECT_FALSE(read_total_cpu_times(input, totals));

  auto provider = make_cpu_usage_provider([](CpuTotals& out) {
    std::istringstream failing{"notcpu 1 2 3 4 5 6 7 8"};
    return read_total_cpu_times(failing, out);
  });
  EXPECT_EQ(provider(), std::nullopt);
}

TEST(CpuUsageProvider, ReturnsFalseWhenStreamIsEmpty)
{
  CpuTotals totals{};
  std::istringstream input{""};
  EXPECT_FALSE(read_total_cpu_times(input, totals));

  auto provider = make_cpu_usage_provider([](CpuTotals& out) {
    std::istringstream failing_stream{""};
    return read_total_cpu_times(failing_stream, out);
  });
  EXPECT_EQ(provider(), std::nullopt);
}

TEST(CpuUsageProvider, ReturnsFalseWhenCpuTotalsAreIncomplete)
{
  CpuTotals totals{};
  std::istringstream input{"cpu 1 2 3 4 5 6"};
  EXPECT_FALSE(read_total_cpu_times(input, totals));

  auto provider = make_cpu_usage_provider([](CpuTotals& out) {
    std::istringstream failing_stream{"cpu 1 2 3 4 5 6"};
    return read_total_cpu_times(failing_stream, out);
  });
  EXPECT_EQ(provider(), std::nullopt);
}

TEST(CpuUsageProvider, FailureDoesNotOverwritePreviousSample)
{
  CpuTotals first{};
  first.user = 100;
  first.system = 50;
  first.idle = 200;

  CpuTotals second{};
  second.user = 150;
  second.system = 70;
  second.idle = 210;

  int call_count = 0;
  auto provider = make_cpu_usage_provider([&](CpuTotals& out) {
    ++call_count;
    if (call_count == 1) {
      out = first;
      return true;
    }
    if (call_count == 2) {
      return false;
    }
    if (call_count == 3) {
      out = second;
      return true;
    }
    return false;
  });

  EXPECT_EQ(provider(), std::nullopt);

  const auto usage = provider();
  ASSERT_TRUE(usage.has_value());
  EXPECT_NEAR(*usage, 87.5, 1e-6);
  EXPECT_EQ(call_count, 3);
}

TEST(CpuUsageProvider, ClampsNegativeUsageToZero)
{
  CpuTotals samples[2]{};
  samples[0].user = 500;
  samples[0].system = 250;
  samples[0].idle = 400;

  samples[1].user = 300;  // Non-idle time decreases.
  samples[1].system = 100;
  samples[1].idle = 900;  // Idle grows enough to make usage negative.

  int sample_index = 0;
  auto provider = make_cpu_usage_provider([&](CpuTotals& out) {
    if (sample_index >= 2) {
      return false;
    }
    out = samples[sample_index++];
    return true;
  });

  const auto usage = provider();
  ASSERT_TRUE(usage.has_value());
  EXPECT_DOUBLE_EQ(*usage, 0.0);
}

TEST(CpuUsageProvider, ReportsFullUtilizationWhenIdleDoesNotChange)
{
  CpuTotals samples[2]{};
  samples[0].user = 200;
  samples[0].system = 150;
  samples[0].idle = 600;

  samples[1].user = 800;
  samples[1].system = 550;
  samples[1].idle = 600;  // No additional idle cycles recorded.

  int sample_index = 0;
  auto provider = make_cpu_usage_provider([&](CpuTotals& out) {
    if (sample_index >= 2) {
      return false;
    }
    out = samples[sample_index++];
    return true;
  });

  const auto usage = provider();
  ASSERT_TRUE(usage.has_value());
  EXPECT_DOUBLE_EQ(*usage, 100.0);
}

}  // namespace
