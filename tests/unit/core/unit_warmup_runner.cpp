#include <gtest/gtest.h>

#include <limits>
#include <map>
#include <vector>

#define private public
#include "core/warmup.hpp"
#undef private

#include "starpu_task_worker/inference_queue.hpp"
#include "test_inference_runner.hpp"
#include "test_warmup_runner.hpp"

TEST_F(WarmupRunnerTest, ClientWorkerPositiveIterations_Unit)
{
  auto device_workers = make_device_workers();
  starpu_server::InferenceQueue queue;

  runner->client_worker(device_workers, queue, 2);

  std::vector<int> job_ids;
  std::vector<int> worker_ids;
  for (;;) {
    std::shared_ptr<starpu_server::InferenceJob> job;
    queue.wait_and_pop(job);
    if (job->is_shutdown()) {
      break;
    }
    job_ids.push_back(job->get_job_id());
    ASSERT_TRUE(job->get_fixed_worker_id().has_value());
    worker_ids.push_back(*job->get_fixed_worker_id());
  }

  ASSERT_EQ(job_ids.size(), 4U);
  EXPECT_EQ(job_ids, (std::vector<int>{0, 1, 2, 3}));
  EXPECT_EQ(worker_ids, (std::vector<int>{1, 1, 2, 2}));
}
