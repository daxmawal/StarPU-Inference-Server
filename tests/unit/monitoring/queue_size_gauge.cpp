#include <gtest/gtest.h>

#include "core/inference_runner.hpp"
#include "monitoring/metrics.hpp"
#include "starpu_task_worker/inference_queue.hpp"

using namespace starpu_server;

TEST(Metrics, QueueGaugeTracksQueueSize)
{
  ASSERT_TRUE(init_metrics(0));

  InferenceQueue queue;
  EXPECT_DOUBLE_EQ(get_metrics()->queue_size_gauge()->Value(), 0);

  auto job = std::make_shared<InferenceJob>();
  EXPECT_TRUE(queue.push(job));
  EXPECT_DOUBLE_EQ(get_metrics()->queue_size_gauge()->Value(), 1);

  std::shared_ptr<InferenceJob> popped;
  EXPECT_TRUE(queue.wait_and_pop(popped));
  EXPECT_DOUBLE_EQ(get_metrics()->queue_size_gauge()->Value(), 0);

  queue.shutdown();
  EXPECT_FALSE(queue.push(job));

  shutdown_metrics();
}

TEST(Metrics, InflightGaugeTracksCountsAndLimit)
{
  ASSERT_TRUE(init_metrics(0));

  set_max_inflight_tasks(8);
  set_inflight_tasks(3);

  auto metrics = get_metrics();
  ASSERT_NE(metrics, nullptr);
  EXPECT_DOUBLE_EQ(metrics->max_inflight_tasks_gauge()->Value(), 8);
  EXPECT_DOUBLE_EQ(metrics->inflight_tasks_gauge()->Value(), 3);

  set_inflight_tasks(0);
  shutdown_metrics();
}

TEST(Metrics, RejectedRequestsCounterIncrements)
{
  ASSERT_TRUE(init_metrics(0));
  auto metrics = get_metrics();
  ASSERT_NE(metrics, nullptr);
  ASSERT_NE(metrics->requests_rejected_total(), nullptr);

  EXPECT_DOUBLE_EQ(metrics->requests_rejected_total()->Value(), 0);
  increment_rejected_requests();
  EXPECT_DOUBLE_EQ(metrics->requests_rejected_total()->Value(), 1);

  shutdown_metrics();
}
