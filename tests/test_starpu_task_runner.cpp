#include <gtest/gtest.h>

#include "core/inference_params.hpp"
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

TEST(StarPUTaskRunnerTest, RunHandlesShutdownJob)
{
  InferenceQueue queue;
  RuntimeConfig opts;
  StarPUTaskRunnerConfig config{};
  config.queue = &queue;
  config.opts = &opts;

  StarPUTaskRunner runner(config);

  queue.push(InferenceJob::make_shutdown_job());

  testing::internal::CaptureStdout();
  runner.run();
  std::string output = testing::internal::GetCapturedStdout();

  EXPECT_NE(output.find("Received shutdown signal"), std::string::npos);
}

TEST(StarPUTaskRunnerTest, RunHandlesSubmissionException)
{
  InferenceQueue queue;
  torch::jit::script::Module model_cpu;
  std::vector<torch::jit::script::Module> models_gpu(
      InferLimits::MaxModelsGPU + 1);
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
  job->set_job_id(1);
  job->set_input_tensors({torch::tensor({1})});

  bool called = false;
  std::vector<torch::Tensor> cb_results;
  double cb_latency = 0.0;
  job->set_on_complete([&](const std::vector<torch::Tensor>& r, double l) {
    called = true;
    cb_results = r;
    cb_latency = l;
  });

  queue.push(job);
  queue.push(InferenceJob::make_shutdown_job());

  runner.run();

  EXPECT_TRUE(called);
  EXPECT_TRUE(cb_results.empty());
  EXPECT_EQ(cb_latency, -1);
  ASSERT_EQ(results.size(), 1u);
  EXPECT_TRUE(results[0].results.empty());
  EXPECT_EQ(results[0].latency_ms, -1);
  EXPECT_EQ(completed_jobs.load(), 1);
}

TEST(StarPUTaskRunnerTest, LogJobTimingsComputesComponents)
{
  RuntimeConfig opts;
  opts.verbosity = VerbosityLevel::Stats;
  StarPUTaskRunnerConfig config{};
  config.opts = &opts;

  StarPUTaskRunner runner(config);

  starpu_server::detail::TimingInfo t;
  using clock = std::chrono::high_resolution_clock;
  auto base = clock::now();
  t.enqueued_time = base;
  t.dequeued_time = base + std::chrono::milliseconds(10);
  t.before_starpu_submitted_time = base + std::chrono::milliseconds(25);
  t.codelet_start_time = base + std::chrono::milliseconds(40);
  t.codelet_end_time = base + std::chrono::milliseconds(70);
  t.inference_start_time = base + std::chrono::milliseconds(80);
  t.callback_start_time = base + std::chrono::milliseconds(125);
  t.callback_end_time = base + std::chrono::milliseconds(140);

  testing::internal::CaptureStdout();
  runner.log_job_timings(42, 150.0, t);
  std::string output = testing::internal::GetCapturedStdout();

  EXPECT_NE(output.find("Queue = 10.000 ms"), std::string::npos);
  EXPECT_NE(output.find("Submit = 15.000 ms"), std::string::npos);
  EXPECT_NE(output.find("Scheduling = 15.000 ms"), std::string::npos);
  EXPECT_NE(output.find("Codelet = 30.000 ms"), std::string::npos);
  EXPECT_NE(output.find("Inference = 45.000 ms"), std::string::npos);
  EXPECT_NE(output.find("Callback = 15.000 ms"), std::string::npos);
}
