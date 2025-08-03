#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

#define private public
#include "core/warmup.hpp"
#undef private

using namespace starpu_server;

static auto
make_identity_model() -> torch::jit::script::Module
{
  torch::jit::script::Module m{"m"};
  m.define(R"JIT(
        def forward(self, x):
            return x
    )JIT");
  return m;
}

TEST(WarmupRunnerTest, ClientWorkerThrowsOnNegativeIterations)
{
  RuntimeConfig opts;
  opts.input_shapes = {{1}};
  opts.input_types = {at::kFloat};
  opts.use_cuda = false;

  StarPUSetup starpu(opts);
  auto model_cpu = make_identity_model();
  std::vector<torch::jit::script::Module> models_gpu;
  std::vector<torch::Tensor> outputs_ref = {torch::zeros({1})};

  WarmupRunner runner(opts, starpu, model_cpu, models_gpu, outputs_ref);

  std::map<int, std::vector<int32_t>> device_workers;
  InferenceQueue queue;

  EXPECT_THROW(
      runner.client_worker(device_workers, queue, -1), std::invalid_argument);
}

TEST(WarmupRunnerTest, WarmupRunnerRunNoCuda)
{
  RuntimeConfig opts;
  opts.input_shapes = {{1}};
  opts.input_types = {at::kFloat};
  opts.use_cuda = false;

  StarPUSetup starpu(opts);
  auto model_cpu = make_identity_model();
  std::vector<torch::jit::script::Module> models_gpu;
  std::vector<torch::Tensor> outputs_ref = {torch::zeros({1})};

  WarmupRunner runner(opts, starpu, model_cpu, models_gpu, outputs_ref);

  const auto start = std::chrono::high_resolution_clock::now();
  runner.run(42);
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::high_resolution_clock::now() - start)
                           .count();

  EXPECT_LT(elapsed, 1000);
}

TEST(WarmupRunnerTest, ClientWorkerPositiveIterations)
{
  RuntimeConfig opts;
  opts.input_shapes = {{1}};
  opts.input_types = {at::kFloat};
  opts.use_cuda = false;

  StarPUSetup starpu(opts);
  auto model_cpu = make_identity_model();
  std::vector<torch::jit::script::Module> models_gpu;
  std::vector<torch::Tensor> outputs_ref = {torch::zeros({1})};

  WarmupRunner runner(opts, starpu, model_cpu, models_gpu, outputs_ref);

  std::map<int, std::vector<int32_t>> device_workers = {{0, {1, 2}}};
  InferenceQueue queue;

  runner.client_worker(device_workers, queue, 2);

  std::vector<int> job_ids;
  std::vector<int> worker_ids;

  while (true) {
    std::shared_ptr<InferenceJob> job;
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

TEST(WarmupRunnerTest, RunReturnsImmediatelyWhenCudaDisabled)
{
  RuntimeConfig opts;
  opts.input_shapes = {{1}};
  opts.input_types = {at::kFloat};
  opts.use_cuda = false;

  StarPUSetup starpu(opts);
  auto model_cpu = make_identity_model();
  std::vector<torch::jit::script::Module> models_gpu;
  std::vector<torch::Tensor> outputs_ref = {torch::zeros({1})};

  WarmupRunner runner(opts, starpu, model_cpu, models_gpu, outputs_ref);

  auto start = std::chrono::steady_clock::now();
  runner.run(100);
  auto end = std::chrono::steady_clock::now();

  auto elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();
  EXPECT_LT(elapsed_ms, 100);
}

TEST(WarmupRunnerTest, ClientWorkerThrowsOnIterationOverflow)
{
  RuntimeConfig opts;
  opts.input_shapes = {{1}};
  opts.input_types = {at::kFloat};
  opts.use_cuda = false;

  StarPUSetup starpu(opts);
  auto model_cpu = make_identity_model();
  std::vector<torch::jit::script::Module> models_gpu;
  std::vector<torch::Tensor> outputs_ref = {torch::zeros({1})};

  WarmupRunner runner(opts, starpu, model_cpu, models_gpu, outputs_ref);

  std::map<int, std::vector<int32_t>> device_workers = {{0, {1, 2}}};
  InferenceQueue queue;

  const int iterations = std::numeric_limits<int>::max();

  EXPECT_THROW(
      runner.client_worker(device_workers, queue, iterations),
      std::overflow_error);
}

TEST(WarmupRunnerTest, WarmupRunWithMockedWorkers)
{
  RuntimeConfig opts;
  opts.input_shapes = {{1}};
  opts.input_types = {at::kFloat};
  opts.use_cuda = true;

  StarPUSetup starpu(opts);
  auto model_cpu = make_identity_model();
  std::vector<torch::jit::script::Module> models_gpu;
  std::vector<torch::Tensor> outputs_ref = {torch::zeros({1})};

  WarmupRunner runner(opts, starpu, model_cpu, models_gpu, outputs_ref);

  std::map<int, std::vector<int32_t>> device_workers = {{0, {1, 2}}};
  InferenceQueue queue;

  std::atomic<int> completed_jobs{0};
  std::condition_variable cv;
  std::mutex m;

  std::jthread server([&]() {
    while (true) {
      std::shared_ptr<InferenceJob> job;
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
    runner.client_worker(device_workers, queue, iterations_per_worker);
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