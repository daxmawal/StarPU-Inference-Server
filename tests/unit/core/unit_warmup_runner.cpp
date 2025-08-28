#include <gtest/gtest.h>

#include <limits>
#include <map>
#include <vector>
#include <unordered_set>

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

TEST_F(WarmupRunnerTest, WarmupPregenInputsRespected_Unit)
{
  auto device_workers = make_device_workers();

  // Case 1: only one pre-generated input
  opts.seed = 0;
  opts.warmup_pregen_inputs = 1;
  starpu_server::InferenceQueue queue_single;
  runner->client_worker(device_workers, queue_single, 1);

  std::unordered_set<const void*> unique_single;
  for (;;) {
    std::shared_ptr<starpu_server::InferenceJob> job;
    queue_single.wait_and_pop(job);
    if (job->is_shutdown()) {
      break;
    }
    unique_single.insert(job->get_input_tensors()[0].data_ptr());
  }
  EXPECT_EQ(unique_single.size(), 1U);

  // Case 2: two pre-generated inputs
  opts.seed = 0;
  opts.warmup_pregen_inputs = 2;
  starpu_server::InferenceQueue queue_double;
  runner->client_worker(device_workers, queue_double, 5);

  std::unordered_set<const void*> unique_double;
  for (;;) {
    std::shared_ptr<starpu_server::InferenceJob> job;
    queue_double.wait_and_pop(job);
    if (job->is_shutdown()) {
      break;
    }
    unique_double.insert(job->get_input_tensors()[0].data_ptr());
  }
  EXPECT_EQ(unique_double.size(), 2U);
}
