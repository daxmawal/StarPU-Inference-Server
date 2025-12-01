#include <gtest/gtest.h>

#include <limits>
#include <map>
#include <string>
#include <unordered_set>
#include <vector>

#define private public
#include "core/warmup.hpp"
#undef private

#include "core/inference_runner.hpp"
#include "starpu_task_worker/inference_queue.hpp"
#include "test_inference_runner.hpp"
#include "test_warmup_runner.hpp"

TEST_F(WarmupRunnerTest, ClientWorkerPositiveRequestNb_Unit)
{
  auto device_workers = make_device_workers();
  starpu_server::InferenceQueue queue;

  runner->client_worker(device_workers, queue, 2);

  std::vector<int> request_ids;
  std::vector<int> worker_ids;
  for (;;) {
    std::shared_ptr<starpu_server::InferenceJob> job;
    if (!queue.wait_and_pop(job)) {
      break;
    }
    request_ids.push_back(job->get_request_id());
    const int worker = job->get_fixed_worker_id().value_or(-1);
    ASSERT_NE(worker, -1);
    worker_ids.push_back(worker);
  }

  ASSERT_EQ(request_ids.size(), 4U);
  EXPECT_EQ(request_ids, (std::vector<int>{0, 1, 2, 3}));
  EXPECT_EQ(worker_ids, (std::vector<int>{1, 1, 2, 2}));
}

TEST_F(WarmupRunnerTest, WarmupPregenInputsRespected_Unit)
{
  auto device_workers = make_device_workers();

  opts.seed = 0;
  opts.batching.warmup_pregen_inputs = 1;
  starpu_server::InferenceQueue queue_single;
  runner->client_worker(device_workers, queue_single, 1);

  std::unordered_set<const void*> unique_single;
  for (;;) {
    std::shared_ptr<starpu_server::InferenceJob> job;
    if (!queue_single.wait_and_pop(job)) {
      break;
    }
    unique_single.insert(job->get_input_tensors()[0].data_ptr());
  }
  EXPECT_EQ(unique_single.size(), 1U);

  opts.seed = 0;
  opts.batching.warmup_pregen_inputs = 2;
  starpu_server::InferenceQueue queue_double;
  constexpr int kDoublerequest_nb = 5;
  runner->client_worker(device_workers, queue_double, kDoublerequest_nb);

  std::unordered_set<const void*> unique_double;
  for (;;) {
    std::shared_ptr<starpu_server::InferenceJob> job;
    if (!queue_double.wait_and_pop(job)) {
      break;
    }
    unique_double.insert(job->get_input_tensors()[0].data_ptr());
  }
  EXPECT_EQ(unique_double.size(), 2U);
}

TEST_F(WarmupRunnerTest, ClientWorkerStopsWhenQueuePushFails)
{
  auto device_workers = make_device_workers();
  starpu_server::InferenceQueue queue;
  const int request_nb = 1;

  testing::internal::CaptureStderr();
  queue.shutdown();
  runner->client_worker(device_workers, queue, request_nb);
  const std::string captured = testing::internal::GetCapturedStderr();

  EXPECT_NE(captured.find("Failed to enqueue job"), std::string::npos);

  std::shared_ptr<starpu_server::InferenceJob> job;
  EXPECT_FALSE(queue.wait_and_pop(job));
}

TEST_F(WarmupRunnerTest, ClientWorkerWithCpuOnlyWorkers)
{
  opts.devices.use_cpu = true;
  opts.devices.use_cuda = false;
  std::map<int, std::vector<int32_t>> device_workers;
  device_workers[std::numeric_limits<int>::min()] = {1, 2};
  starpu_server::InferenceQueue queue;
  runner->client_worker(device_workers, queue, 1);
  std::vector<int> worker_ids;
  for (;;) {
    std::shared_ptr<starpu_server::InferenceJob> job;
    if (!queue.wait_and_pop(job)) {
      break;
    }
    const int worker = job->get_fixed_worker_id().value_or(-1);
    if (worker != -1) {
      worker_ids.push_back(worker);
    }
  }
  EXPECT_EQ(worker_ids.size(), 2U);
}

TEST_F(WarmupRunnerTest, ClientWorkerHandlesEmptyWorkerList)
{
  std::map<int, std::vector<int32_t>> device_workers;
  device_workers[0] = {};
  device_workers[1] = {3, 4};
  starpu_server::InferenceQueue queue;
  runner->client_worker(device_workers, queue, 1);
  std::vector<int> worker_ids;
  for (;;) {
    std::shared_ptr<starpu_server::InferenceJob> job;
    if (!queue.wait_and_pop(job)) {
      break;
    }
    const int worker = job->get_fixed_worker_id().value_or(-1);
    if (worker != -1) {
      worker_ids.push_back(worker);
    }
  }
  EXPECT_EQ(worker_ids.size(), 2U);
  EXPECT_EQ(worker_ids, (std::vector<int>{3, 4}));
}

TEST_F(WarmupRunnerTest, ClientWorkerMapAccessForMultipleDevices)
{
  std::map<int, std::vector<int32_t>> device_workers;
  device_workers[0] = {1};
  device_workers[1] = {2};
  device_workers[2] = {3};
  starpu_server::InferenceQueue queue;
  runner->client_worker(device_workers, queue, 1);
  std::vector<int> worker_ids;
  for (;;) {
    std::shared_ptr<starpu_server::InferenceJob> job;
    if (!queue.wait_and_pop(job)) {
      break;
    }
    const int worker = job->get_fixed_worker_id().value_or(-1);
    if (worker != -1) {
      worker_ids.push_back(worker);
    }
  }
  EXPECT_EQ(worker_ids.size(), 3U);
  EXPECT_EQ(worker_ids, (std::vector<int>{1, 2, 3}));
}

TEST_F(WarmupRunnerTest, RunCallsCollectDeviceWorkersWithCpuOnly)
{
  opts.devices.use_cpu = true;
  opts.devices.use_cuda = false;
  opts.batching.warmup_request_nb = 1;
  std::atomic<int> jobs_completed = 0;
  auto completion_observer = [&jobs_completed,
                              this](std::atomic<int>& completed) {
    jobs_completed = completed.load();
  };
  init(false, completion_observer);
  EXPECT_NO_THROW(runner->run(1));
}

TEST_F(WarmupRunnerTest, RunWithNoDevicesConfigured)
{
  opts.devices.use_cpu = false;
  opts.devices.use_cuda = false;
  init(false);
  EXPECT_NO_THROW(runner->run(1));
}

TEST_F(WarmupRunnerTest, RunExercisesDeviceWorkerCollection)
{
  opts.devices.use_cpu = true;
  opts.devices.use_cuda = false;
  opts.batching.warmup_request_nb = 1;
  std::atomic<int> completed_jobs = 0;
  auto observer = [&completed_jobs](std::atomic<int>& jobs) {
    completed_jobs = jobs.load();
  };
  init(false, observer);
  EXPECT_NO_THROW(runner->run(0));
}

TEST_F(WarmupRunnerTest, RunExercisesCudaWorkerCollectionPath)
{
  opts.devices.use_cpu = false;
  opts.devices.use_cuda = true;
  opts.devices.ids = {0};
  opts.batching.warmup_request_nb = 0;
  init(false);
  EXPECT_NO_THROW(runner->run(0));
}

TEST_F(WarmupRunnerTest, RunExercisesDeviceIdMapAccess)
{
  opts.devices.use_cpu = true;
  opts.devices.use_cuda = false;
  opts.batching.warmup_request_nb = 0;
  init(false);
  EXPECT_NO_THROW(runner->run(0));
  opts.devices.use_cpu = false;
  opts.devices.use_cuda = true;
  opts.devices.ids = {0};
  opts.batching.warmup_request_nb = 0;
  runner = std::make_unique<starpu_server::WarmupRunner>(
      opts, *starpu, model_cpu, models_gpu, outputs_ref);
  EXPECT_NO_THROW(runner->run(0));
}
