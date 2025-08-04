#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#define private public
#include "core/warmup.hpp"
#undef private

#include "inference_runner_test_utils.hpp"

template <class F>
static auto
measure_ms(F&& f) -> long
{
  const auto start = std::chrono::steady_clock::now();
  f();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}

class WarmupRunnerTest : public ::testing::Test {
 protected:
  starpu_server::RuntimeConfig opts;

  std::unique_ptr<starpu_server::StarPUSetup> starpu;
  torch::jit::script::Module model_cpu;
  std::vector<torch::jit::script::Module> models_gpu;
  std::vector<torch::Tensor> outputs_ref;
  std::unique_ptr<starpu_server::WarmupRunner> runner;

  void SetUp() override { init(false); }

  void init(bool use_cuda)
  {
    opts = starpu_server::RuntimeConfig{};
    opts.input_shapes = {{1}};
    opts.input_types = {at::kFloat};
    opts.use_cuda = use_cuda;

    starpu = std::make_unique<starpu_server::StarPUSetup>(opts);
    model_cpu = starpu_server::make_identity_model();
    models_gpu.clear();
    outputs_ref = {torch::zeros({1})};
    runner = std::make_unique<starpu_server::WarmupRunner>(
        opts, *starpu, model_cpu, models_gpu, outputs_ref);
  }
};

TEST_F(WarmupRunnerTest, ClientWorkerThrowsOnNegativeIterations)
{
  std::map<int, std::vector<int32_t>> device_workers;
  starpu_server::InferenceQueue queue;
  EXPECT_THROW(
      runner->client_worker(device_workers, queue, -1), std::invalid_argument);
}

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

TEST_F(WarmupRunnerTest, ClientWorkerThrowsOnIterationOverflow)
{
  std::map<int, std::vector<int32_t>> device_workers = {{0, {1, 2}}};
  starpu_server::InferenceQueue queue;
  const int iterations = std::numeric_limits<int>::max();
  EXPECT_THROW(
      runner->client_worker(device_workers, queue, iterations),
      std::overflow_error);
}

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
