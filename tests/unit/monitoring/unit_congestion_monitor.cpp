#include <gtest/gtest.h>

#include <chrono>
#include <functional>
#include <memory>
#include <thread>

#include "core/inference_runner.hpp"
#include "monitoring/congestion_monitor.hpp"
#include "starpu_task_worker/inference_queue.hpp"

using namespace std::chrono_literals;

namespace {

auto
wait_until(
    const std::function<bool()>& predicate,
    std::chrono::milliseconds timeout) -> bool
{
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(20ms);
  }
  return predicate();
}

class CongestionMonitorTest : public ::testing::Test {
 protected:
  void TearDown() override { starpu_server::congestion::shutdown(); }
};

TEST_F(CongestionMonitorTest, PanicOnRejectionSetsCongestionFlag)
{
  starpu_server::InferenceQueue queue(8);
  starpu_server::congestion::Config cfg;
  cfg.tick_interval = 20ms;
  cfg.entry_horizon = 40ms;
  cfg.exit_horizon = 60ms;
  ASSERT_TRUE(starpu_server::congestion::start(&queue, cfg));

  starpu_server::congestion::record_arrival(1);
  starpu_server::congestion::record_rejection(1);

  EXPECT_TRUE(wait_until(
      [] { return starpu_server::congestion::is_congested(); }, 500ms));

  const auto snap = starpu_server::congestion::snapshot();
  ASSERT_TRUE(snap.has_value());
  EXPECT_GT(snap->rejection_rps, 0.0);
  EXPECT_TRUE(snap->congestion);
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

  for (int i = 0; i < 8; ++i) {
    queue.push(std::make_shared<starpu_server::InferenceJob>());
  }
  starpu_server::congestion::record_arrival(8);

  EXPECT_TRUE(wait_until(
      [] { return starpu_server::congestion::is_congested(); }, 800ms));

  std::shared_ptr<starpu_server::InferenceJob> drained;
  while (queue.try_pop(drained)) {
  }
  starpu_server::congestion::record_completion(16, 1.0, 10.0);
  starpu_server::congestion::record_completion(16, 1.0, 10.0);

  EXPECT_TRUE(wait_until(
      [] { return !starpu_server::congestion::is_congested(); }, 1200ms));

  const auto snap = starpu_server::congestion::snapshot();
  ASSERT_TRUE(snap.has_value());
  EXPECT_FALSE(snap->congestion);
  EXPECT_LT(snap->fill_ewma, cfg.fill_high);
  EXPECT_LT(snap->rho_ewma, cfg.rho_high);
}

}  // namespace
