#include <gtest/gtest.h>

#include "core/inference_runner.hpp"
#include "starpu_task_worker/starpu_task_worker.hpp"

using namespace starpu_server;

struct SomeException : public std::exception {
  const char* what() const noexcept override { return "SomeException"; }
};

TEST(StarPUTaskRunnerTest, HandleJobExceptionCallback)
{
  auto job = std::make_shared<InferenceJob>();
  bool called = false;
  std::vector<torch::Tensor> results_arg;
  double latency_arg = 0.0;

  job->set_on_complete(
      [&](const std::vector<torch::Tensor>& results, double latency) {
        called = true;
        results_arg = results;
        latency_arg = latency;
      });

  StarPUTaskRunner::handle_job_exception(job, SomeException{});

  EXPECT_TRUE(called);
  EXPECT_TRUE(results_arg.empty());
  EXPECT_EQ(latency_arg, -1);
}

TEST(StarPUTaskRunnerTest, ShouldShutdown)
{
  RuntimeConfig opts;
  StarPUTaskRunnerConfig config{};
  config.opts = &opts;
  StarPUTaskRunner runner(config);

  auto shutdown_job = InferenceJob::make_shutdown_job();
  auto normal_job = std::make_shared<InferenceJob>();

  EXPECT_TRUE(runner.should_shutdown(shutdown_job));
  EXPECT_FALSE(runner.should_shutdown(normal_job));
}

TEST(StarPUTaskRunnerTest, PrepareJobCompletionCallback)
{
  InferenceQueue queue;
  torch::jit::script::Module model_cpu;
  std::vector<torch::jit::script::Module> models_gpu;
  RuntimeConfig opts;
  std::vector<InferenceResult> results;
  std::mutex results_mutex;
  std::atomic<int> completed_jobs{0};
  std::condition_variable cv;

  StarPUTaskRunnerConfig config{};
  config.queue = &queue;
  config.model_cpu = &model_cpu;
  config.models_gpu = &models_gpu;
  config.starpu = nullptr;
  config.opts = &opts;
  config.results = &results;
  config.results_mutex = &results_mutex;
  config.completed_jobs = &completed_jobs;
  config.all_done_cv = &cv;

  StarPUTaskRunner runner(config);

  auto job = std::make_shared<InferenceJob>();
  job->set_job_id(7);
  std::vector<torch::Tensor> inputs = {torch::tensor({1})};
  job->set_input_tensors(inputs);

  bool original_called = false;
  std::vector<torch::Tensor> orig_results;
  double orig_latency = 0.0;

  job->set_on_complete([&](const std::vector<torch::Tensor>& r, double l) {
    original_called = true;
    orig_results = r;
    orig_latency = l;
  });

  runner.prepare_job_completion_callback(job);

  std::vector<torch::Tensor> outputs = {torch::tensor({2})};
  const double latency = 5.0;
  job->get_on_complete()(outputs, latency);

  EXPECT_TRUE(original_called);
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(completed_jobs.load(), 1);
  EXPECT_EQ(results[0].job_id, 7);
  ASSERT_EQ(results[0].results.size(), outputs.size());
  EXPECT_TRUE(torch::equal(results[0].results[0], outputs[0]));
  ASSERT_EQ(orig_results.size(), outputs.size());
  EXPECT_TRUE(torch::equal(orig_results[0], outputs[0]));
  EXPECT_EQ(orig_latency, latency);
}