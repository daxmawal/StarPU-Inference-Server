#include <gtest/gtest.h>

#include <filesystem>
#include <limits>
#include <map>
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
