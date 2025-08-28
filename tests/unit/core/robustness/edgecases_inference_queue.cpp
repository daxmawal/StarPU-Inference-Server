#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <vector>

#include "core/inference_runner.hpp"
#include "starpu_task_worker/inference_queue.hpp"

TEST(InferenceQueue_Robustesse, MultipleConsumersDrainThenShutdownUnblocksAll)
{
  starpu_server::InferenceQueue queue;
  std::mutex mutex;
  std::vector<int> popped_ids;
  std::atomic<int> consumers_done{0};
  auto consume = [&] {
    for (;;) {
      std::shared_ptr<starpu_server::InferenceJob> job;
      if (!queue.wait_and_pop(job)) {
        consumers_done.fetch_add(1);
        break;
      }
      std::scoped_lock lock(mutex);
      popped_ids.push_back(job->get_job_id());
    }
  };
  std::jthread consume_1(consume);
  std::jthread consume_2(consume);
  for (int job_id = 0; job_id < 6; ++job_id) {
    auto job = std::make_shared<starpu_server::InferenceJob>();
    job->set_job_id(job_id);
    queue.push(job);
  }
  queue.shutdown();
  queue.shutdown();
  consume_1.join();
  consume_2.join();
  EXPECT_EQ(consumers_done.load(), 2);
  {
    std::scoped_lock lock(mutex);
    ASSERT_EQ(popped_ids.size(), 6U);
    std::sort(popped_ids.begin(), popped_ids.end());
    for (int i = 0; i < 6; ++i) {
      EXPECT_EQ(popped_ids[i], i);
    }
  }
}
