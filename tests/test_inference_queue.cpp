#include <gtest/gtest.h>

#include <thread>
#include <vector>

#include "core/inference_runner.hpp"
#include "starpu_task_worker/inference_queue.hpp"

using namespace starpu_server;

TEST(InferenceQueue, FifoAndShutdown)
{
  InferenceQueue queue;
  std::vector<int> popped_ids;

  std::thread consumer([&]() {
    while (true) {
      std::shared_ptr<InferenceJob> job;
      queue.wait_and_pop(job);
      if (job->is_shutdown()) {
        popped_ids.push_back(-1);
        break;
      }
      popped_ids.push_back(job->get_job_id());
    }
  });

  for (int i = 0; i < 3; ++i) {
    auto job = std::make_shared<InferenceJob>();
    job->set_job_id(i);
    queue.push(job);
  }

  queue.shutdown();
  consumer.join();

  ASSERT_EQ(popped_ids.size(), 4u);
  EXPECT_EQ(popped_ids[0], 0);
  EXPECT_EQ(popped_ids[1], 1);
  EXPECT_EQ(popped_ids[2], 2);
  EXPECT_EQ(popped_ids[3], -1);  // shutdown job
}