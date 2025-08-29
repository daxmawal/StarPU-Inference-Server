#include <gtest/gtest.h>

#include <thread>
#include <vector>

#include "core/inference_runner.hpp"
#include "starpu_task_worker/inference_queue.hpp"

// NOLINTBEGIN(readability-function-cognitive-complexity)
TEST(InferenceQueue_Integration, FifoAndShutdown)
{
  starpu_server::InferenceQueue queue;
  std::vector<int> popped_ids;
  {
    std::jthread consumer([&]() {
      for (;;) {
        std::shared_ptr<starpu_server::InferenceJob> job;
        if (!queue.wait_and_pop(job)) {
          popped_ids.push_back(-1);
          break;
        }
        popped_ids.push_back(job->get_job_id());
      }
    });

    for (int i = 0; i < 3; ++i) {
      auto job = std::make_shared<starpu_server::InferenceJob>();
      job->set_job_id(i);
      ASSERT_TRUE(queue.push(job));
    }
    queue.shutdown();
  }
  ASSERT_EQ(popped_ids.size(), 4U);
  EXPECT_EQ(popped_ids[0], 0);
  EXPECT_EQ(popped_ids[1], 1);
  EXPECT_EQ(popped_ids[2], 2);
  EXPECT_EQ(popped_ids[3], -1);

  auto job = std::make_shared<starpu_server::InferenceJob>();
  EXPECT_FALSE(queue.push(job));
}
// NOLINTEND(readability-function-cognitive-complexity)
