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

}  // namespace
