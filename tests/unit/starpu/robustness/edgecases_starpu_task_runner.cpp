#include "test_starpu_task_runner.hpp"

#include <functional>
#include <string>

struct SomeException : public std::exception {
  [[nodiscard]] auto what() const noexcept -> const char* override
  {
    return "SomeException";
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

struct InvalidConfigParam {
  std::string name;
  std::function<void(starpu_server::StarPUTaskRunnerConfig&)> nullify;
};

class StarPUTaskRunnerConfigInvalidTest
    : public ::testing::TestWithParam<InvalidConfigParam> {};

TEST_P(StarPUTaskRunnerConfigInvalidTest, NullPointerThrows)
{
  starpu_server::InferenceQueue queue;
  torch::jit::script::Module model_cpu;
  std::vector<torch::jit::script::Module> models_gpu;
  starpu_server::RuntimeConfig opts;
  std::vector<starpu_server::InferenceResult> results;
  std::mutex results_mutex;
  std::atomic<int> completed_jobs{0};
  std::condition_variable cv;
  starpu_server::StarPUSetup starpu(opts);

  starpu_server::StarPUTaskRunnerConfig config{};
  config.queue = &queue;
  config.model_cpu = &model_cpu;
  config.models_gpu = &models_gpu;
  config.starpu = &starpu;
  config.opts = &opts;
  config.results = &results;
  config.results_mutex = &results_mutex;
  config.completed_jobs = &completed_jobs;
  config.all_done_cv = &cv;

  auto param = GetParam();
  param.nullify(config);

  EXPECT_THROW({ starpu_server::StarPUTaskRunner runner(config); }, std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(
    InvalidConfig,
    StarPUTaskRunnerConfigInvalidTest,
    ::testing::Values(
        InvalidConfigParam{"Queue", [](auto& c) { c.queue = nullptr; }},
        InvalidConfigParam{"ModelCpu", [](auto& c) { c.model_cpu = nullptr; }},
        InvalidConfigParam{"ModelsGpu", [](auto& c) { c.models_gpu = nullptr; }},
        InvalidConfigParam{"Starpu", [](auto& c) { c.starpu = nullptr; }},
        InvalidConfigParam{"Opts", [](auto& c) { c.opts = nullptr; }},
        InvalidConfigParam{"Results", [](auto& c) { c.results = nullptr; }},
        InvalidConfigParam{
            "ResultsMutex", [](auto& c) { c.results_mutex = nullptr; }},
        InvalidConfigParam{
            "CompletedJobs", [](auto& c) { c.completed_jobs = nullptr; }},
        InvalidConfigParam{"AllDoneCv", [](auto& c) { c.all_done_cv = nullptr; }}),
    [](const ::testing::TestParamInfo<InvalidConfigParam>& info) {
      return info.param.name;
    });
