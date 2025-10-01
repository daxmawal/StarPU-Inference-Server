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

namespace {
class WarmupRunnerTestHookGuard {
 public:
  WarmupRunnerTestHookGuard()
  {
    starpu_server::WarmupRunner::clear_test_hook();
  }

  ~WarmupRunnerTestHookGuard()
  {
    starpu_server::WarmupRunner::clear_test_hook();
  }
};
}  // namespace

TEST_F(WarmupRunnerTest, WarmupRunnerRunNegativeIterations_Robustesse)
{
  init(true);
  EXPECT_THROW(runner->run(-1), std::invalid_argument);
}

TEST_F(
    WarmupRunnerTest, WarmupRunnerRunThrowsOnNegativeCompletedJobs_Robustesse)
{
  init(true);
  WarmupRunnerTestHookGuard guard;
  starpu_server::WarmupRunner::set_test_hook(
      [](std::atomic<int>& completed_jobs) { completed_jobs.store(-1); });

  EXPECT_THROW(runner->run(1), starpu_server::InferenceExecutionException);
}

class WarmupRunnerClientWorkerInvalidIterations_Robustesse
    : public WarmupRunnerTest,
      public ::testing::WithParamInterface<std::pair<int, bool>> {};

namespace {
void
ExpectClientWorkerThrows(
    starpu_server::WarmupRunner& runner,
    const std::map<int, std::vector<int32_t>>& device_workers,
    starpu_server::InferenceQueue& queue, int iterations, bool expect_overflow)
{
  try {
    runner.client_worker(device_workers, queue, iterations);
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
    WarmupRunnerClientWorkerInvalidIterations_Robustesse,
    ThrowsOnInvalidIterations)
{
  auto device_workers = make_device_workers();
  starpu_server::InferenceQueue queue;
  auto [iterations, expect_overflow] = GetParam();
  ExpectClientWorkerThrows(
      *runner, device_workers, queue, iterations, expect_overflow);
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
