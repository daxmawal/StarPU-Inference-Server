#include <gtest/gtest.h>

#include <limits>
#include <map>
#include <vector>

#define private public
#include "core/warmup.hpp"
#undef private

#include "starpu_task_worker/inference_queue.hpp"
#include "test_warmup_runner.hpp"

TEST_F(WarmupRunnerTest, WarmupRunnerRunNegativeIterations_Robustesse)
{
  init(true);
  EXPECT_THROW(runner->run(-1), std::invalid_argument);
}

class WarmupRunnerClientWorkerInvalidIterations_Robustesse
    : public WarmupRunnerTest,
      public ::testing::WithParamInterface<std::pair<int, bool>> {};

TEST_P(
    WarmupRunnerClientWorkerInvalidIterations_Robustesse,
    ThrowsOnInvalidIterations)
{
  auto device_workers = make_device_workers();
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
    ClientWorkerInvalidIterationTests_Robustesse,
    WarmupRunnerClientWorkerInvalidIterations_Robustesse,
    ::testing::Values(
        std::make_pair(-1, false),
        std::make_pair(std::numeric_limits<int>::max(), true)));

TEST(WarmupRunnerEdgesTest, ClientWorkerThrowsOnWorkerCountOverflow_Robustesse)
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
