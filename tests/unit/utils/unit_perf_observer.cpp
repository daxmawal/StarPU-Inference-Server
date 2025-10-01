#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>

#include "utils/perf_observer.hpp"

using starpu_server::perf_observer::record_job;
using starpu_server::perf_observer::reset;
using starpu_server::perf_observer::snapshot;

namespace {

class PerfObserverTest : public ::testing::Test {
 protected:
  void SetUp() override { reset(); }
};

}  // namespace

TEST_F(PerfObserverTest, SnapshotWithoutDataReturnsNullopt)
{
  const auto result = snapshot();
  EXPECT_FALSE(result.has_value());
}

TEST_F(PerfObserverTest, SnapshotAfterRecordingJobReturnsMetrics)
{
  using namespace std::chrono_literals;
  const auto base = std::chrono::high_resolution_clock::time_point{};
  const auto enqueue_time = base + 1ms;
  const auto completion_time = base + 101ms;
  constexpr std::size_t kBatchSize = 10;

  record_job(
      enqueue_time, completion_time, kBatchSize, /*is_warmup_job=*/false);

  const auto result = snapshot();
  ASSERT_TRUE(result.has_value());
  const auto& snapshot_value = *result;

  const double expected_duration =
      std::chrono::duration<double>(completion_time - enqueue_time).count();
  const double expected_throughput =
      static_cast<double>(kBatchSize) / expected_duration;

  EXPECT_EQ(snapshot_value.total_inferences, kBatchSize);
  EXPECT_DOUBLE_EQ(snapshot_value.duration_seconds, expected_duration);
  EXPECT_DOUBLE_EQ(snapshot_value.throughput, expected_throughput);
}

TEST_F(PerfObserverTest, SnapshotWithNonIncreasingDurationReturnsNullopt)
{
  const auto time_point = std::chrono::high_resolution_clock::time_point{};
  constexpr std::size_t kBatchSize = 5;

  record_job(time_point, time_point, kBatchSize, /*is_warmup_job=*/false);

  const auto result = snapshot();
  EXPECT_FALSE(result.has_value());
}

TEST_F(PerfObserverTest, SnapshotIgnoresZeroBatchJob)
{
  using namespace std::chrono_literals;
  const auto base = std::chrono::high_resolution_clock::time_point{};
  const auto enqueue_time = base + 1ms;
  const auto completion_time = base + 2ms;

  record_job(
      enqueue_time, completion_time, /*batch_size=*/0, /*is_warmup_job=*/false);

  const auto result = snapshot();
  EXPECT_FALSE(result.has_value());
}
