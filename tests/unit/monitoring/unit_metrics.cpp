#include <gtest/gtest.h>

#include "monitoring/metrics.hpp"

using namespace starpu_server;

TEST(Metrics, InitializesPointersAndRegistry)
{
  init_metrics(0);

  ASSERT_NE(metrics_registry, nullptr);
  ASSERT_NE(requests_total, nullptr);
  ASSERT_NE(inference_latency, nullptr);
  ASSERT_NE(queue_size_gauge, nullptr);

  bool has_requests = false;
  bool has_latency = false;
  bool has_queue = false;
  for (const auto& family : metrics_registry->Collect()) {
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

  metrics_registry.reset();
  requests_total = nullptr;
  inference_latency = nullptr;
  queue_size_gauge = nullptr;
}
