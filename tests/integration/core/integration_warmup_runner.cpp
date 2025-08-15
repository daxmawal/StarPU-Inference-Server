#include <gtest/gtest.h>

#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <iterator>
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

#include "starpu_task_worker/inference_queue.hpp"
#include "test_inference_runner.hpp"
#include "test_warmup_runner.hpp"

namespace {
auto
count_threads() -> int
{
  namespace fs = std::filesystem;
  auto distance = std::distance(
      fs::directory_iterator("/proc/self/task"), fs::directory_iterator{});
  if (distance > std::numeric_limits<int>::max())
    throw std::overflow_error("Thread count exceeds int range");
  return static_cast<int>(distance);
}
}  // namespace

TEST_F(WarmupRunnerTest, WarmupRunnerRunNoCuda_Integration)
{
  auto elapsed_ms = measure_ms([&]() { runner->run(42); });
  EXPECT_LT(elapsed_ms, 1000);
}

TEST_F(WarmupRunnerTest, RunReturnsImmediatelyWhenCudaDisabled_Integration)
{
  auto elapsed_ms = measure_ms([&]() { runner->run(100); });
  EXPECT_LT(elapsed_ms, 100);
}

TEST_F(WarmupRunnerTest, WarmupRunWithMockedWorkers_Integration)
{
  init(true);
  auto device_workers = make_device_workers();
  starpu_server::InferenceQueue queue;

  std::atomic<size_t> completed{0};
  std::condition_variable cv;
  std::mutex m;

  std::jthread server([&] {
    for (;;) {
      std::shared_ptr<starpu_server::InferenceJob> job;
      queue.wait_and_pop(job);
      if (job->is_shutdown())
        break;
      completed.fetch_add(1);
      cv.notify_one();
    }
  });

  const int iters = 1;
  std::jthread client(
      [&] { runner->client_worker(device_workers, queue, iters); });

  size_t total_workers = 0;
  for (auto& [dev, ws] : device_workers) (void)dev, total_workers += ws.size();
  const size_t total_jobs = total_workers * static_cast<size_t>(iters);

  {
    std::unique_lock lk(m);
    cv.wait(lk, [&] { return completed.load() >= total_jobs; });
  }
  EXPECT_EQ(completed.load(), total_jobs);
}

TEST_F(WarmupRunnerTest, WarmupRunnerRunZeroIterations_Integration)
{
  init(true);
  auto elapsed_ms = measure_ms([&]() { runner->run(0); });
  auto device_workers =
      starpu_server::StarPUSetup::get_cuda_workers_by_device(opts.device_ids);
  EXPECT_TRUE(device_workers.empty());
  EXPECT_LT(elapsed_ms, 100);
}

TEST(WarmupRunnerEdgesTest, RunNoCudaNoThreads_Integration)
{
  WarmupRunnerTestFixture fixture;
  fixture.init();
  auto runner = fixture.make_runner();
  const auto before = count_threads();
  const auto elapsed_ms = measure_ms([&]() { runner.run(100); });
  const auto after = count_threads();
  EXPECT_EQ(before, after);
  EXPECT_LT(elapsed_ms, 50);
}
