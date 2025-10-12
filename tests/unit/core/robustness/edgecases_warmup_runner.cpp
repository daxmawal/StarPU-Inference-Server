#include <gtest/gtest.h>

#include <atomic>
#include <limits>
#include <map>
#include <vector>

#define private public
#include "core/warmup.hpp"
#undef private

#include "exceptions.hpp"
#include "starpu_task_worker/inference_queue.hpp"
#include "test_warmup_runner.hpp"

TEST_F(WarmupRunnerTest, WarmupRunnerRunNegativeRequestNbRobustesse)
{
  init(true);
  EXPECT_THROW(runner->run(-1), std::invalid_argument);
}

TEST_F(
    WarmupRunnerTest, WarmupRunnerRunThrowsOnNegativeCompletedJobs_Robustesse)
{
  init(
      true, [](std::atomic<int>& completed_jobs) { completed_jobs.store(-1); });

  EXPECT_THROW(runner->run(1), starpu_server::InferenceExecutionException);
}

class WarmupRunnerClientWorkerInvalidRequestNb_Robustesse
    : public WarmupRunnerTest,
      public ::testing::WithParamInterface<std::pair<int, bool>> {};

namespace {
void
ExpectClientWorkerThrows(
    starpu_server::WarmupRunner& runner,
    const std::map<int, std::vector<int32_t>>& device_workers,
    starpu_server::InferenceQueue& queue, int request_nb, bool expect_overflow)
{
  try {
    runner.client_worker(device_workers, queue, request_nb);
    ADD_FAILURE() << "Expected exception not thrown";
  }
  catch (const std::overflow_error&) {
    EXPECT_TRUE(expect_overflow);
  }
  catch (const std::invalid_argument&) {
    EXPECT_FALSE(expect_overflow);
  }
}
}  // namespace

TEST_P(
    WarmupRunnerClientWorkerInvalidRequestNb_Robustesse,
    ThrowsOnInvalidRequestNb)
{
  auto device_workers = make_device_workers();
  starpu_server::InferenceQueue queue;
  auto [request_nb, expect_overflow] = GetParam();
  ExpectClientWorkerThrows(
      *runner, device_workers, queue, request_nb, expect_overflow);
}

INSTANTIATE_TEST_SUITE_P(
    ClientWorkerInvalidRequestNbTests_Robustesse,
    WarmupRunnerClientWorkerInvalidRequestNb_Robustesse,
    ::testing::Values(
        std::make_pair(-1, false),
        std::make_pair(std::numeric_limits<int>::max(), true)));

TEST(WarmupRunnerEdgesTest, ClientWorkerThrowsOnWorkerCountOverflow_Robustesse)
{
  WarmupRunnerTestFixture fixture;
  fixture.init();
  auto runner = fixture.make_runner();

  const int request_nb = 1000;
  const size_t worker_count =
      static_cast<size_t>(std::numeric_limits<int>::max()) /
          static_cast<size_t>(request_nb) +
      1;

  std::vector<int32_t> many_workers(worker_count, 0);
  std::map<int, std::vector<int32_t>> device_workers = {{0, many_workers}};
  starpu_server::InferenceQueue queue;

  EXPECT_THROW(
      runner.client_worker(device_workers, queue, request_nb),
      std::overflow_error);
}
