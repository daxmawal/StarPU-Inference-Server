#include <gtest/gtest.h>

#include <vector>

#include "core/latency_statistics.hpp"

namespace starpu_server { namespace {

TEST(LatencyStatisticsTest, ReturnsNulloptWhenNoSamples)
{
  std::vector<double> latencies;
  const auto stats = compute_latency_statistics(latencies);
  EXPECT_FALSE(stats.has_value());
}

TEST(LatencyStatisticsTest, ComputesStatsForOddSample)
{
  const std::vector<double> latencies{35.0, 5.0, 15.0, 45.0, 25.0};
  const auto stats = compute_latency_statistics(latencies);
  ASSERT_TRUE(stats.has_value());
  EXPECT_DOUBLE_EQ(stats->p100, 45.0);
  EXPECT_NEAR(stats->p95, 43.0, 1e-9);
  EXPECT_NEAR(stats->p85, 39.0, 1e-9);
  EXPECT_DOUBLE_EQ(stats->p50, 25.0);
  EXPECT_DOUBLE_EQ(stats->mean, 25.0);
}

TEST(LatencyStatisticsTest, ComputesStatsForEvenSample)
{
  const std::vector<double> latencies{10.0, 30.0, 40.0, 20.0};
  const auto stats = compute_latency_statistics(latencies);
  ASSERT_TRUE(stats.has_value());
  EXPECT_DOUBLE_EQ(stats->p100, 40.0);
  EXPECT_NEAR(stats->p95, 38.5, 1e-9);
  EXPECT_NEAR(stats->p85, 35.5, 1e-9);
  EXPECT_DOUBLE_EQ(stats->p50, 25.0);
  EXPECT_DOUBLE_EQ(stats->mean, 25.0);
}

}}  // namespace starpu_server
