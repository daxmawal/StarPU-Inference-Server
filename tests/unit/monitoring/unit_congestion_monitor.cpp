#include <gtest/gtest.h>
#include <prometheus/metric_family.h>

#include <chrono>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

#include "core/inference_runner.hpp"
#define private public
#include "monitoring/congestion_monitor.hpp"
#undef private
#include "monitoring/metrics.hpp"
#include "starpu_task_worker/inference_queue.hpp"

using namespace std::chrono_literals;

namespace {

auto
wait_until(
    const std::function<bool()>& predicate, std::chrono::milliseconds timeout,
    std::chrono::milliseconds poll_interval = 20ms) -> bool
{
  const auto sleep_interval =
      poll_interval > std::chrono::milliseconds::zero() ? poll_interval : 1ms;
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(sleep_interval);
  }
  return predicate();
}

auto
find_gauge_value(
    const std::vector<prometheus::MetricFamily>& families,
    std::string_view family_name) -> std::optional<double>
{
  for (const auto& family : families) {
    if (family.name != family_name || family.metric.empty()) {
      continue;
    }
    return family.metric.front().gauge.value;
  }
  return std::nullopt;
}

class CongestionMonitorTest : public ::testing::Test {
 protected:
  void TearDown() override { starpu_server::congestion::shutdown(); }
};

class CongestionShutdownGuard {
 public:
  ~CongestionShutdownGuard() { starpu_server::congestion::shutdown(); }
};

TEST_F(CongestionMonitorTest, PanicOnRejectionSetsCongestionFlag)
{
  starpu_server::InferenceQueue queue(8);
  starpu_server::congestion::Config cfg;
  cfg.tick_interval = 20ms;
  cfg.entry_horizon = 40ms;
  cfg.exit_horizon = 60ms;
  ASSERT_TRUE(starpu_server::congestion::start(&queue, cfg));
  CongestionShutdownGuard shutdown_guard;

  starpu_server::congestion::record_arrival(1);
  starpu_server::congestion::record_rejection(1);

  std::optional<starpu_server::congestion::Snapshot> panic_snapshot;
  EXPECT_TRUE(wait_until(
      [&panic_snapshot] {
        panic_snapshot = starpu_server::congestion::snapshot();
        return panic_snapshot.has_value() && panic_snapshot->congestion &&
               panic_snapshot->rejection_rps > 0.0;
      },
      1s, 5ms));

  ASSERT_TRUE(panic_snapshot.has_value());
  EXPECT_GT(panic_snapshot->rejection_rps, 0.0);
  EXPECT_TRUE(panic_snapshot->congestion);
}

TEST_F(CongestionMonitorTest, ClearsAfterExitConditionsHold)
{
  starpu_server::InferenceQueue queue(10);
  starpu_server::congestion::Config cfg;
  cfg.tick_interval = 30ms;
  cfg.entry_horizon = 60ms;
  cfg.exit_horizon = 120ms;
  cfg.latency_slo_ms = 100.0;
  ASSERT_TRUE(starpu_server::congestion::start(&queue, cfg));
  CongestionShutdownGuard shutdown_guard;

  for (int i = 0; i < 8; ++i) {
    ASSERT_TRUE(queue.push(std::make_shared<starpu_server::InferenceJob>()));
  }
  starpu_server::congestion::record_arrival(8);

  EXPECT_TRUE(wait_until(
      [] { return starpu_server::congestion::is_congested(); }, 800ms));

  std::shared_ptr<starpu_server::InferenceJob> drained;
  while (queue.try_pop(drained)) {
  }
  starpu_server::congestion::record_completion(
      16, starpu_server::congestion::CompletionLatencies{
              .queue_latency_ms = 1.0,
              .e2e_latency_ms = 10.0,
          });
  starpu_server::congestion::record_completion(
      16, starpu_server::congestion::CompletionLatencies{
              .queue_latency_ms = 1.0,
              .e2e_latency_ms = 10.0,
          });

  EXPECT_TRUE(wait_until(
      [] { return !starpu_server::congestion::is_congested(); }, 1200ms));

  const auto snap = starpu_server::congestion::snapshot();
  ASSERT_TRUE(snap.has_value());
  EXPECT_FALSE(snap->congestion);
  EXPECT_LT(snap->fill_ewma, cfg.fill_high);
  EXPECT_LT(snap->rho_ewma, cfg.rho_high);
}

TEST_F(CongestionMonitorTest, StartReturnsFalseWithoutQueue)
{
  starpu_server::congestion::Config cfg;
  cfg.enabled = true;

  EXPECT_FALSE(starpu_server::congestion::start(nullptr, cfg));
  EXPECT_FALSE(starpu_server::congestion::is_congested());
  EXPECT_FALSE(starpu_server::congestion::snapshot().has_value());
}

TEST_F(CongestionMonitorTest, SnapshotReturnsNulloptWhenStopped)
{
  starpu_server::congestion::shutdown();
  EXPECT_FALSE(starpu_server::congestion::snapshot().has_value());
}

TEST_F(CongestionMonitorTest, StartPublishesMetricsWhenGlobalMetricsAvailable)
{
  starpu_server::shutdown_metrics();
  ASSERT_TRUE(starpu_server::init_metrics(0));
  struct MetricsGuard {
    ~MetricsGuard() { starpu_server::shutdown_metrics(); }
  } guard;

  starpu_server::InferenceQueue queue(4);
  starpu_server::congestion::Config cfg;
  cfg.tick_interval = 20ms;
  cfg.entry_horizon = 40ms;
  cfg.exit_horizon = 60ms;
  cfg.latency_slo_ms = 10.0;
  ASSERT_TRUE(starpu_server::congestion::start(&queue, cfg));
  CongestionShutdownGuard shutdown_guard;

  for (int i = 0; i < 3; ++i) {
    ASSERT_TRUE(queue.push(std::make_shared<starpu_server::InferenceJob>()));
  }
  starpu_server::congestion::record_arrival(3);
  starpu_server::congestion::record_completion(
      2, starpu_server::congestion::CompletionLatencies{
             .queue_latency_ms = 4.0,
             .e2e_latency_ms = 8.0,
         });

  EXPECT_TRUE(wait_until(
      [] {
        const auto snap = starpu_server::congestion::snapshot();
        return snap.has_value() && snap->queue_size == 3U &&
               snap->queue_capacity == 4U;
      },
      800ms, 10ms));

  EXPECT_TRUE(wait_until(
      [] {
        auto metrics = starpu_server::get_metrics();
        if (metrics == nullptr) {
          return false;
        }
        const auto families = metrics->registry()->Collect();
        const auto arrival_rate =
            find_gauge_value(families, "inference_lambda_rps");
        const auto completion_rate =
            find_gauge_value(families, "inference_mu_rps");
        const auto fill_ewma =
            find_gauge_value(families, "inference_queue_fill_ratio_ewma");
        const auto queue_p95 =
            find_gauge_value(families, "inference_queue_latency_p95_ms");
        const auto e2e_p95 =
            find_gauge_value(families, "inference_e2e_latency_p95_ms");
        return arrival_rate.has_value() && completion_rate.has_value() &&
               fill_ewma.has_value() && queue_p95.has_value() &&
               e2e_p95.has_value() && *arrival_rate > 0.0 &&
               *completion_rate > 0.0 && *fill_ewma > 0.0 && *queue_p95 > 0.0 &&
               *e2e_p95 > 0.0;
      },
      1200ms, 10ms));
}

TEST(PercentileTest, ReturnsMinWhenPctNonPositive)
{
  std::vector<double> samples{3.0, 1.0, 2.0};
  const auto value = starpu_server::congestion::percentile_for_test(samples, 0);
  ASSERT_TRUE(value.has_value());
  EXPECT_DOUBLE_EQ(*value, 1.0);
}

TEST(PercentileTest, ReturnsMaxWhenPctAtLeastHundred)
{
  std::vector<double> samples{3.0, 1.0, 2.0};
  const auto value =
      starpu_server::congestion::percentile_for_test(samples, 100.0);
  ASSERT_TRUE(value.has_value());
  EXPECT_DOUBLE_EQ(*value, 3.0);
}

TEST(PercentileTest, ReturnsSampleWhenPositionIsExactIndex)
{
  std::vector<double> samples{5.0, 1.0, 3.0, 2.0, 4.0};
  const auto value =
      starpu_server::congestion::percentile_for_test(samples, 50.0);
  ASSERT_TRUE(value.has_value());
  EXPECT_DOUBLE_EQ(*value, 3.0);
}

TEST(UpdateEwmaTest, TreatsNonFiniteSampleAsZero)
{
  std::optional<double> state;
  const double sample = std::numeric_limits<double>::quiet_NaN();
  const double alpha = 0.5;

  const auto value =
      starpu_server::congestion::update_ewma_for_test(state, sample, alpha);

  ASSERT_TRUE(state.has_value());
  EXPECT_DOUBLE_EQ(*state, 0.0);
  EXPECT_DOUBLE_EQ(value, 0.0);
}

TEST(RecordArrivalTest, SkipsZeroCount)
{
  const auto arrivals = starpu_server::congestion::record_arrival_for_test(0);
  EXPECT_EQ(arrivals, 0U);
}

TEST(RecordRejectionTest, SkipsZeroCount)
{
  const auto rejections =
      starpu_server::congestion::record_rejection_for_test(0);
  EXPECT_EQ(rejections, 0U);
}

TEST(LatencyFlagsTest, UsesQueueBudgetForDangerWhenE2EMissing)
{
  const auto flags =
      starpu_server::congestion::evaluate_latency_flags_for_test(10.0, 15.0);
  EXPECT_TRUE(flags.danger);
  EXPECT_FALSE(flags.ok);
}

TEST(LatencyFlagsTest, UsesQueueBudgetForOkWhenE2EMissing)
{
  const auto flags =
      starpu_server::congestion::evaluate_latency_flags_for_test(10.0, 5.0);
  EXPECT_FALSE(flags.danger);
  EXPECT_TRUE(flags.ok);
}

TEST(QueuePressureScoreTest, ReturnsZeroWhenHighNotAboveLow)
{
  const auto score =
      starpu_server::congestion::compute_queue_pressure_score_for_test(
          {.fill_high = 0.5, .fill_low = 0.5, .fill_smoothed = 0.9});
  EXPECT_DOUBLE_EQ(score, 0.0);
}

TEST(QueuePressureScoreTest, ReturnsScaledValueWhenThresholdsValid)
{
  const auto score =
      starpu_server::congestion::compute_queue_pressure_score_for_test(
          {.fill_high = 0.8, .fill_low = 0.2, .fill_smoothed = 0.5});
  EXPECT_NEAR(score, 0.5, 1e-9);
}

TEST(LatencyPressureScoreTest, ReturnsScaledValueWhenUpperAboveLower)
{
  const auto score =
      starpu_server::congestion::compute_latency_pressure_score_for_test(
          {.latency_slo_ms = 100.0, .e2e_ok_ratio = 0.8, .e2e_p95 = 95.0});
  EXPECT_NEAR(score, 0.5, 1e-9);
}

TEST(LatencyPressureScoreTest, UsesQueueBudgetPathWhenAvailable)
{
  const auto score = starpu_server::congestion::
      compute_latency_pressure_score_queue_budget_for_test(10.0, 11.0);
  EXPECT_NEAR(score, 0.5, 1e-9);
}

TEST(NormalizeConfigTest, FixesNonPositiveTickInterval)
{
  starpu_server::congestion::Config cfg;
  cfg.tick_interval = 0ms;
  cfg.entry_horizon = 2s;
  cfg.exit_horizon = 3s;

  const auto result = starpu_server::congestion::normalize_config_for_test(cfg);
  EXPECT_EQ(result.tick_interval, 1s);
}

TEST(NormalizeConfigTest, ClampsEntryHorizonToTickInterval)
{
  starpu_server::congestion::Config cfg;
  cfg.tick_interval = 200ms;
  cfg.entry_horizon = 100ms;
  cfg.exit_horizon = 400ms;

  const auto result = starpu_server::congestion::normalize_config_for_test(cfg);
  EXPECT_EQ(result.entry_horizon, 200ms);
  EXPECT_EQ(result.exit_horizon, 400ms);
}

TEST(NormalizeConfigTest, ClampsExitHorizonToDoubleTickInterval)
{
  starpu_server::congestion::Config cfg;
  cfg.tick_interval = 200ms;
  cfg.entry_horizon = 200ms;
  cfg.exit_horizon = 100ms;

  const auto result = starpu_server::congestion::normalize_config_for_test(cfg);
  EXPECT_EQ(result.exit_horizon, 400ms);
}

TEST(NormalizeConfigTest, UsesQueueLatencyBudgetWhenProvided)
{
  starpu_server::congestion::Config cfg;
  cfg.queue_latency_budget_ms = 42.0;
  cfg.latency_slo_ms = 100.0;
  cfg.queue_latency_budget_ratio = 0.5;

  const auto result = starpu_server::congestion::normalize_config_for_test(cfg);
  ASSERT_TRUE(result.queue_budget_ms.has_value());
  EXPECT_DOUBLE_EQ(*result.queue_budget_ms, 42.0);
}

TEST(CapacityPressureScoreTest, ReturnsZeroWhenHighNotAboveLow)
{
  const auto score =
      starpu_server::congestion::compute_capacity_pressure_score_for_test(
          {.rho_high = 1.0, .rho_low = 1.0, .rho_smoothed = 2.0});
  EXPECT_DOUBLE_EQ(score, 0.0);
}

TEST(CapacityPressureScoreTest, ReturnsScaledValueWhenThresholdsValid)
{
  const auto score =
      starpu_server::congestion::compute_capacity_pressure_score_for_test(
          {.rho_high = 1.0, .rho_low = 0.2, .rho_smoothed = 0.6});
  EXPECT_NEAR(score, 0.5, 1e-9);
}

TEST(MonitorTest, ReturnsFallbackValuesWhenImplIsMissing)
{
  starpu_server::congestion::Monitor monitor(nullptr);
  monitor.impl_.reset();

  const auto flags =
      monitor.evaluate_latency_flags_for_test(/*queue_p95=*/1.0, 2.0);
  EXPECT_FALSE(flags.danger);
  EXPECT_TRUE(flags.ok);

  const auto config = monitor.normalized_config_for_test();
  EXPECT_FALSE(config.queue_budget_ms.has_value());
}

}  // namespace
