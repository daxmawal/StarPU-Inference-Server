#include <gtest/gtest.h>

#include <atomic>
#include <condition_variable>
#include <mutex>

#include "core/inference_params.hpp"
#include "core/inference_runner.hpp"
#include "starpu_task_worker/starpu_task_worker.hpp"
#include "test_utils.hpp"

struct SomeException : public std::exception {
  [[nodiscard]] auto what() const noexcept -> const char* override
  {
    return "SomeException";
  }
};

class StarPUTaskRunnerFixture : public ::testing::Test {
 protected:
  starpu_server::InferenceQueue queue_;
  torch::jit::script::Module model_cpu_;
  std::vector<torch::jit::script::Module> models_gpu_;
  starpu_server::RuntimeConfig opts_;
  std::vector<starpu_server::InferenceResult> results_;
  std::mutex results_mutex_;
  std::atomic<int> completed_jobs_;
  std::condition_variable cv_;
  starpu_server::StarPUTaskRunnerConfig config_{};
  std::unique_ptr<starpu_server::StarPUTaskRunner> runner_;
  void SetUp() override
  {
    completed_jobs_ = 0;
    config_.queue = &queue_;
    config_.model_cpu = &model_cpu_;
    config_.models_gpu = &models_gpu_;
    config_.starpu = nullptr;
    config_.opts = &opts_;
    config_.results = &results_;
    config_.results_mutex = &results_mutex_;
    config_.completed_jobs = &completed_jobs_;
    config_.all_done_cv = &cv_;
    runner_ = std::make_unique<starpu_server::StarPUTaskRunner>(config_);
  }
};

TEST_F(StarPUTaskRunnerFixture, HandleJobExceptionCallback)
{
  auto probe = starpu_server::make_callback_probe();
  starpu_server::StarPUTaskRunner::handle_job_exception(
      probe.job, SomeException{});
  EXPECT_TRUE(probe.called);
  EXPECT_TRUE(probe.results.empty());
  EXPECT_EQ(probe.latency, -1);
}

TEST_F(StarPUTaskRunnerFixture, ShouldShutdown)
{
  auto shutdown_job = starpu_server::InferenceJob::make_shutdown_job();
  auto normal_job = std::make_shared<starpu_server::InferenceJob>();
  EXPECT_TRUE(runner_->should_shutdown(shutdown_job));
  EXPECT_FALSE(runner_->should_shutdown(normal_job));
}

TEST_F(StarPUTaskRunnerFixture, PrepareJobCompletionCallback)
{
  auto probe = starpu_server::make_callback_probe();
  auto job = probe.job;
  job->set_job_id(7);
  std::vector<torch::Tensor> inputs = {torch::tensor({1})};
  job->set_input_tensors(inputs);
  runner_->prepare_job_completion_callback(job);
  std::vector<torch::Tensor> outputs = {torch::tensor({2})};
  const double latency = 5.0;
  job->get_on_complete()(outputs, latency);
  EXPECT_TRUE(probe.called);
  auto& results = results_;
  const auto& completed_jobs = completed_jobs_;
  ASSERT_EQ(results.size(), 1U);
  EXPECT_EQ(completed_jobs.load(), 1);
  EXPECT_EQ(results[0].job_id, 7);
  ASSERT_EQ(results[0].results.size(), outputs.size());
  EXPECT_TRUE(torch::equal(results[0].results[0], outputs[0]));
  ASSERT_EQ(probe.results.size(), outputs.size());
  EXPECT_TRUE(torch::equal(probe.results[0], outputs[0]));
  EXPECT_EQ(probe.latency, latency);
}

TEST_F(StarPUTaskRunnerFixture, RunHandlesShutdownJob)
{
  queue_.push(starpu_server::InferenceJob::make_shutdown_job());
  std::string output = starpu_server::capture_stdout([&] { runner_->run(); });
  EXPECT_NE(output.find("Received shutdown signal"), std::string::npos);
}

TEST_F(StarPUTaskRunnerFixture, RunHandlesSubmissionException)
{
  auto& models_gpu = models_gpu_;
  models_gpu.resize(starpu_server::InferLimits::MaxModelsGPU + 1);
  auto probe = starpu_server::make_callback_probe();
  auto job = probe.job;
  job->set_job_id(1);
  job->set_input_tensors({torch::tensor({1})});
  queue_.push(job);
  queue_.push(starpu_server::InferenceJob::make_shutdown_job());
  runner_->run();
  EXPECT_TRUE(probe.called);
  EXPECT_TRUE(probe.results.empty());
  EXPECT_EQ(probe.latency, -1);
  auto& results = results_;
  const auto& completed_jobs = completed_jobs_;
  ASSERT_EQ(results.size(), 1U);
  EXPECT_TRUE(results[0].results.empty());
  EXPECT_EQ(results[0].latency_ms, -1);
  EXPECT_EQ(completed_jobs.load(), 1);
}

TEST_F(StarPUTaskRunnerFixture, LogJobTimingsComputesComponents)
{
  opts_.verbosity = starpu_server::VerbosityLevel::Stats;
  starpu_server::detail::TimingInfo time;
  using clock = std::chrono::high_resolution_clock;
  auto base = clock::now();
  time.enqueued_time = base;
  time.dequeued_time = base + std::chrono::milliseconds(10);
  time.before_starpu_submitted_time = base + std::chrono::milliseconds(25);
  time.codelet_start_time = base + std::chrono::milliseconds(40);
  time.codelet_end_time = base + std::chrono::milliseconds(70);
  time.inference_start_time = base + std::chrono::milliseconds(80);
  time.callback_start_time = base + std::chrono::milliseconds(125);
  time.callback_end_time = base + std::chrono::milliseconds(140);
  std::string output = starpu_server::capture_stdout(
      [&] { runner_->log_job_timings(42, 150.0, time); });
  EXPECT_NE(output.find("Queue = 10.000 ms"), std::string::npos);
  EXPECT_NE(output.find("Submit = 15.000 ms"), std::string::npos);
  EXPECT_NE(output.find("Scheduling = 15.000 ms"), std::string::npos);
  EXPECT_NE(output.find("Codelet = 30.000 ms"), std::string::npos);
  EXPECT_NE(output.find("Inference = 45.000 ms"), std::string::npos);
  EXPECT_NE(output.find("Callback = 15.000 ms"), std::string::npos);
}
