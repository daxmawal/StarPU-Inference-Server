#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <thread>
#include <vector>

#include "core/inference_runner.hpp"
#include "starpu_task_worker/inference_queue.hpp"

namespace {
constexpr int kNumJobs = 6;

void
ConsumeAll(
    starpu_server::InferenceQueue& queue, std::mutex& mtx,
    std::vector<int>& popped_ids, std::atomic<int>& consumers_done)
{
  for (;;) {
    std::shared_ptr<starpu_server::InferenceJob> job;
    if (!queue.wait_and_pop(job)) {
      consumers_done.fetch_add(1);
      break;
    }
    std::scoped_lock lock(mtx);
    popped_ids.push_back(job->get_job_id());
  }
}

void
PushJobs(starpu_server::InferenceQueue& queue, int count)
{
  for (int job_id = 0; job_id < count; ++job_id) {
    auto job = std::make_shared<starpu_server::InferenceJob>();
    job->set_job_id(job_id);
    ASSERT_TRUE(queue.push(job));
  }
}

void
VerifyPoppedIds(std::mutex& mtx, std::vector<int>& ids, int count)
{
  std::scoped_lock lock(mtx);
  ASSERT_EQ(ids.size(), static_cast<size_t>(count));
  std::sort(ids.begin(), ids.end());
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(ids[static_cast<size_t>(i)], i);
  }
}
}  // namespace

TEST(InferenceQueue_Robustesse, MultipleConsumersDrainThenShutdownUnblocksAll)
{
  starpu_server::InferenceQueue queue;
  std::mutex mutex;
  std::vector<int> popped_ids;
  std::atomic<int> consumers_done{0};

  std::jthread consume_1(
      [&] { ConsumeAll(queue, mutex, popped_ids, consumers_done); });
  std::jthread consume_2(
      [&] { ConsumeAll(queue, mutex, popped_ids, consumers_done); });

  PushJobs(queue, kNumJobs);
  queue.shutdown();
  queue.shutdown();
  consume_1.join();
  consume_2.join();

  EXPECT_FALSE(queue.push(std::make_shared<starpu_server::InferenceJob>()));
  EXPECT_EQ(consumers_done.load(), 2);
  VerifyPoppedIds(mutex, popped_ids, kNumJobs);
}

TEST(InferenceQueue_Robustesse, RejectsNullJob)
{
  starpu_server::InferenceQueue queue;
  std::shared_ptr<starpu_server::InferenceJob> null_job;

  EXPECT_FALSE(queue.push(null_job));
  EXPECT_EQ(queue.size(), 0U);
}
