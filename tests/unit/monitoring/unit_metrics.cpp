#include <gtest/gtest.h>

#include "monitoring/metrics.hpp"

using namespace starpu_server;

TEST(Metrics, InitializesPointersAndRegistry)
{
  init_metrics(0);

  auto m = metrics.load(std::memory_order_acquire);
  ASSERT_NE(m, nullptr);
  ASSERT_NE(m->registry, nullptr);
  ASSERT_NE(m->requests_total, nullptr);
  ASSERT_NE(m->inference_latency, nullptr);
  ASSERT_NE(m->queue_size_gauge, nullptr);

  bool has_requests = false;
  bool has_latency = false;
  bool has_queue = false;
  for (const auto& family : m->registry->Collect()) {
    if (family.name == "requests_total") {
      has_requests = true;
    } else if (family.name == "inference_latency_ms") {
      has_latency = true;
    } else if (family.name == "inference_queue_size") {
      has_queue = true;
    }
  }
  EXPECT_TRUE(has_requests);
  EXPECT_TRUE(has_latency);
  EXPECT_TRUE(has_queue);

  shutdown_metrics();
  EXPECT_EQ(metrics.load(std::memory_order_acquire), nullptr);
}
