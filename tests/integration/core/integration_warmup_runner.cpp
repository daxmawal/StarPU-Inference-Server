#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "core/inference_runner.hpp"
#include "core/starpu_setup.hpp"
#include "core/warmup.hpp"
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
  if (distance > std::numeric_limits<int>::max()) {
    throw std::overflow_error("Thread count exceeds int range");
  }
  return static_cast<int>(distance);
}
}  // namespace

TEST_F(WarmupRunnerTest, WarmupRunnerRunNoCuda_Integration)
{
  std::atomic<std::size_t> last_observed{0};
  auto local_runner =
      make_runner([&](std::atomic<std::size_t>& completed_jobs) {
        last_observed.store(completed_jobs.load(), std::memory_order_relaxed);
      });

  const auto cpu_workers =
      starpu_server::StarPUSetup::get_worker_ids_by_type(STARPU_CPU_WORKER);
  ASSERT_FALSE(cpu_workers.empty());

  constexpr int kWarmupQuick = 1;
  EXPECT_NO_THROW(local_runner.run(kWarmupQuick));

  const auto expected_jobs =
      cpu_workers.size() * static_cast<std::size_t>(kWarmupQuick);
  EXPECT_EQ(last_observed.load(std::memory_order_relaxed), expected_jobs);
}

TEST_F(WarmupRunnerTest, WarmupRunsOnCpuWhenCudaDisabled_Integration)
{
  std::atomic<std::size_t> last_observed{0};
  auto local_runner =
      make_runner([&](std::atomic<std::size_t>& completed_jobs) {
        last_observed.store(completed_jobs.load(), std::memory_order_relaxed);
      });

  const auto cpu_workers =
      starpu_server::StarPUSetup::get_worker_ids_by_type(STARPU_CPU_WORKER);
  ASSERT_FALSE(cpu_workers.empty());

  constexpr int kWarmupShort = 3;
  local_runner.run(kWarmupShort);

  const auto expected_jobs =
      cpu_workers.size() * static_cast<std::size_t>(kWarmupShort);
  EXPECT_EQ(last_observed.load(), expected_jobs);
}

TEST_F(WarmupRunnerTest, WarmupRunWithMockedWorkers_Integration)
{
  auto device_workers = make_device_workers();
  starpu_server::InferenceQueue queue;

  std::atomic<size_t> completed{0};
  std::condition_variable cond_var;
  std::mutex mutex;

  std::jthread server([&] {
    for (;;) {
      std::shared_ptr<starpu_server::InferenceJob> job;
      if (!queue.wait_and_pop(job)) {
        break;
      }
      completed.fetch_add(1);
      cond_var.notify_one();
    }
  });

  const int iters = 1;
  std::jthread client([&] {
    starpu_server::WarmupRunnerTestHelper::client_worker(
        *runner, device_workers, queue, iters);
  });

  size_t total_workers = 0;
  for (auto& [dev, ws] : device_workers) {
    (void)dev;
    total_workers += ws.size();
  }
  const size_t total_jobs = total_workers * static_cast<size_t>(iters);

  {
    std::unique_lock lock(mutex);
    const auto finished =
        cond_var.wait_for(lock, std::chrono::seconds(10), [&] {
          return completed.load(std::memory_order_relaxed) >= total_jobs;
        });
    if (!finished) {
      queue.shutdown();
    }
    ASSERT_TRUE(finished) << "Warmup mocked workers did not complete in time";
  }
  EXPECT_EQ(completed.load(std::memory_order_relaxed), total_jobs);
}

TEST_F(WarmupRunnerTest, WarmupRunnerRunZeroRequestNb_Integration)
{
  std::atomic<std::size_t> last_observed{0};
  auto local_runner =
      make_runner([&](std::atomic<std::size_t>& completed_jobs) {
        last_observed.store(completed_jobs.load(), std::memory_order_relaxed);
      });
  EXPECT_NO_THROW(local_runner.run(0));
  EXPECT_EQ(last_observed.load(std::memory_order_relaxed), 0U);
}

TEST(WarmupRunnerEdgesTest, RunNoCudaNoThreads_Integration)
{
  WarmupRunnerTestFixture fixture;
  fixture.init();
  fixture.opts.devices.use_cpu = false;
  fixture.opts.devices.use_cuda = false;
  fixture.starpu = std::make_unique<starpu_server::StarPUSetup>(fixture.opts);
  fixture.model_cpu = starpu_server::make_identity_model();
  fixture.models_gpu.clear();
  fixture.outputs_ref = {torch::zeros({1})};
  auto runner = fixture.make_runner();
  const auto before = count_threads();
  constexpr int kWarmupShort = 100;
  EXPECT_NO_THROW(runner.run(kWarmupShort));
  const auto after = count_threads();
  EXPECT_EQ(before, after);
}
