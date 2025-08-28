#include <gtest/gtest.h>

#include "monitoring/metrics.hpp"

using namespace starpu_server;

TEST(Metrics, InitializesPointersAndRegistry)
{
  init_metrics(0);

  ASSERT_NE(metrics, nullptr);
  ASSERT_NE(metrics->registry, nullptr);
  ASSERT_NE(metrics->requests_total, nullptr);
  ASSERT_NE(metrics->inference_latency, nullptr);
  ASSERT_NE(metrics->queue_size_gauge, nullptr);

  bool has_requests = false;
  bool has_latency = false;
  bool has_queue = false;
  for (const auto& family : metrics->registry->Collect()) {
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
  EXPECT_EQ(metrics, nullptr);
}
