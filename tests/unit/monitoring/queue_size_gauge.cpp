#include <gtest/gtest.h>

#include "monitoring/metrics.hpp"
#include "starpu_task_worker/inference_queue.hpp"

using namespace starpu_server;

TEST(Metrics, QueueGaugeTracksQueueSize)
{
  ASSERT_TRUE(init_metrics(0));

  InferenceQueue queue;
  EXPECT_DOUBLE_EQ(
      metrics.load(std::memory_order_acquire)->queue_size_gauge->Value(), 0);

  std::shared_ptr<InferenceJob> job;
  EXPECT_TRUE(queue.push(job));
  EXPECT_DOUBLE_EQ(
      metrics.load(std::memory_order_acquire)->queue_size_gauge->Value(), 1);

  std::shared_ptr<InferenceJob> popped;
  EXPECT_TRUE(queue.wait_and_pop(popped));
  EXPECT_DOUBLE_EQ(
      metrics.load(std::memory_order_acquire)->queue_size_gauge->Value(), 0);

  queue.shutdown();
  EXPECT_FALSE(queue.push(job));

  shutdown_metrics();
}
