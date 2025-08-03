#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>

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

static auto
count_threads() -> int
{
  namespace fs = std::filesystem;
  int count = 0;
  for (const auto& entry : fs::directory_iterator("/proc/self/task")) {
    (void)entry;
    ++count;
  }
  return count;
}

TEST(WarmupRunnerEdgesTest, RunNoCudaNoThreads)
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

  const auto threads_before = count_threads();
  const auto start = std::chrono::steady_clock::now();
  runner.run(100);
  const auto end = std::chrono::steady_clock::now();
  const auto threads_after = count_threads();

  const auto elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();

  EXPECT_EQ(threads_before, threads_after);
  EXPECT_LT(elapsed_ms, 50);
}

TEST(WarmupRunnerEdgesTest, ClientWorkerThrowsOnNegativeIterations)
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

TEST(WarmupRunnerEdgesTest, ClientWorkerThrowsOnIterationOverflow)
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

TEST(WarmupRunnerEdgesTest, ClientWorkerThrowsOnWorkerCountOverflow)
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

  const int iterations = 1000;
  const size_t worker_count =
      static_cast<size_t>(std::numeric_limits<int>::max()) /
          static_cast<size_t>(iterations) +
      1;
  std::vector<int32_t> many_workers(worker_count, 0);
  std::map<int, std::vector<int32_t>> device_workers = {{0, many_workers}};
  InferenceQueue queue;

  EXPECT_THROW(
      runner.client_worker(device_workers, queue, iterations),
      std::overflow_error);
}
