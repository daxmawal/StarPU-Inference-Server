#include <gtest/gtest.h>

#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#define private public
#include "core/warmup.hpp"
#undef private

#include "inference_runner_test_utils.hpp"
#include "warmup_runner_test_utils.hpp"

namespace {
auto
count_threads() -> int
{
  namespace fs = std::filesystem;
  return std::distance(fs::directory_iterator("/proc/self/task"), {});
}
}  // namespace

TEST_F(WarmupRunnerTest, WarmupRunnerRunNoCuda)
{
  auto elapsed_ms = measure_ms([&]() { runner->run(42); });
  EXPECT_LT(elapsed_ms, 1000);
}

TEST_F(WarmupRunnerTest, ClientWorkerPositiveIterations)
{
  std::map<int, std::vector<int32_t>> device_workers = {{0, {1, 2}}};
  starpu_server::InferenceQueue queue;
  runner->client_worker(device_workers, queue, 2);
  std::vector<int> job_ids;
  std::vector<int> worker_ids;
  while (true) {
    std::shared_ptr<starpu_server::InferenceJob> job;
    queue.wait_and_pop(job);
    if (job->is_shutdown()) {
      break;
    }
    job_ids.push_back(job->get_job_id());
    ASSERT_TRUE(job->get_fixed_worker_id().has_value());
    worker_ids.push_back(*job->get_fixed_worker_id());
  }
  ASSERT_EQ(job_ids.size(), 4u);
  EXPECT_EQ(job_ids, (std::vector<int>{0, 1, 2, 3}));
  EXPECT_EQ(worker_ids, (std::vector<int>{1, 1, 2, 2}));
}

TEST_F(WarmupRunnerTest, RunReturnsImmediatelyWhenCudaDisabled)
{
  auto elapsed_ms = measure_ms([&]() { runner->run(100); });
  EXPECT_LT(elapsed_ms, 100);
}

TEST_F(WarmupRunnerTest, WarmupRunnerRunNegativeIterations)
{
  init(true);
  EXPECT_THROW(runner->run(-1), std::invalid_argument);
}

class WarmupRunnerClientWorkerInvalidIterationsTest
    : public WarmupRunnerTest,
      public ::testing::WithParamInterface<std::pair<int, bool>> {};

TEST_P(WarmupRunnerClientWorkerInvalidIterationsTest, ThrowsOnInvalidIterations)
{
  std::map<int, std::vector<int32_t>> device_workers = {{0, {1, 2}}};
  starpu_server::InferenceQueue queue;
  auto [iterations, expect_overflow] = GetParam();
  if (expect_overflow) {
    EXPECT_THROW(
        runner->client_worker(device_workers, queue, iterations),
        std::overflow_error);
  } else {
    EXPECT_THROW(
        runner->client_worker(device_workers, queue, iterations),
        std::invalid_argument);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ClientWorkerInvalidIterationTests,
    WarmupRunnerClientWorkerInvalidIterationsTest,
    ::testing::Values(
        std::make_pair(-1, false),
        std::make_pair(std::numeric_limits<int>::max(), true)));

TEST_F(WarmupRunnerTest, WarmupRunWithMockedWorkers)
{
  init(true);
  std::map<int, std::vector<int32_t>> device_workers = {{0, {1, 2}}};
  starpu_server::InferenceQueue queue;
  std::atomic<int> completed_jobs{0};
  std::condition_variable cv;
  std::mutex m;
  std::jthread server([&]() {
    while (true) {
      std::shared_ptr<starpu_server::InferenceJob> job;
      queue.wait_and_pop(job);
      if (job->is_shutdown()) {
        break;
      }
      completed_jobs.fetch_add(1);
      cv.notify_one();
    }
  });
  const int iterations_per_worker = 1;
  std::jthread client([&]() {
    runner->client_worker(device_workers, queue, iterations_per_worker);
  });
  size_t total_worker_count = 0;
  for (const auto& [device, workers] : device_workers) {
    (void)device;
    total_worker_count += workers.size();
  }
  const size_t total_jobs =
      static_cast<size_t>(iterations_per_worker) * total_worker_count;

  {
    std::unique_lock lock(m);
    cv.wait(lock, [&]() {
      return static_cast<size_t>(completed_jobs.load()) >= total_jobs;
    });
  }
  EXPECT_EQ(static_cast<size_t>(completed_jobs.load()), total_jobs);
}

TEST_F(WarmupRunnerTest, WarmupRunnerRunZeroIterations)
{
  init(true);
  auto elapsed_ms = measure_ms([&]() { runner->run(0); });
  auto device_workers =
      starpu_server::StarPUSetup::get_cuda_workers_by_device(opts.device_ids);
  EXPECT_TRUE(device_workers.empty());
  EXPECT_LT(elapsed_ms, 100);
}

TEST(WarmupRunnerEdgesTest, RunNoCudaNoThreads)
{
  WarmupRunnerTestFixture fixture;
  fixture.init();
  auto runner = fixture.make_runner();
  const auto threads_before = count_threads();
  const auto elapsed_ms = measure_ms([&]() { runner.run(100); });
  const auto threads_after = count_threads();
  EXPECT_EQ(threads_before, threads_after);
  EXPECT_LT(elapsed_ms, 50);
}

TEST(WarmupRunnerEdgesTest, ClientWorkerThrowsOnWorkerCountOverflow)
{
  WarmupRunnerTestFixture fixture;
  fixture.init();
  auto runner = fixture.make_runner();
  const int iterations = 1000;
  const size_t worker_count =
      static_cast<size_t>(std::numeric_limits<int>::max()) /
          static_cast<size_t>(iterations) +
      1;
  std::vector<int32_t> many_workers(worker_count, 0);
  std::map<int, std::vector<int32_t>> device_workers = {{0, many_workers}};
  starpu_server::InferenceQueue queue;
  EXPECT_THROW(
      runner.client_worker(device_workers, queue, iterations),
      std::overflow_error);
}