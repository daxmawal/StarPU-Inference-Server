#include <format>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

#include "test_starpu_task_runner.hpp"

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

TEST_F(
    StarPUTaskRunnerFixture, HandleJobExceptionCallbackLogsStdExceptionMessage)
{
  auto job = make_job(7, {});
  job->set_on_complete([](const auto&, double) {
    throw std::runtime_error("callback failure");
  });

  starpu_server::CaptureStream capture{std::cerr};
  EXPECT_NO_THROW(starpu_server::StarPUTaskRunner::handle_job_exception(
      job, std::runtime_error{"job failure"}));

  const auto log = capture.str();
  const auto expected = starpu_server::expected_log_line(
      starpu_server::ErrorLevel,
      "Exception in completion callback: callback failure");
  EXPECT_NE(log.find(expected), std::string::npos);
}

TEST_F(
    StarPUTaskRunnerFixture,
    HandleJobExceptionCallbackLogsUnknownNonStdExceptionMessage)
{
  auto job = make_job(8, {});
  job->set_on_complete([](const auto&, double) -> void { throw 42; });

  starpu_server::CaptureStream capture{std::cerr};
  EXPECT_NO_THROW(starpu_server::StarPUTaskRunner::handle_job_exception(
      job, std::runtime_error{"job failure"}));

  const auto log = capture.str();
  const auto expected = starpu_server::expected_log_line(
      starpu_server::ErrorLevel, "Unknown exception in completion callback");
  EXPECT_NE(log.find(expected), std::string::npos);
}

TEST_F(StarPUTaskRunnerFixture, RunHandlesSubmissionException)
{
  auto& models_gpu = models_gpu_;
  models_gpu.resize(starpu_server::InferLimits::MaxModelsGPU + 1);
  auto probe = starpu_server::make_callback_probe();
  auto job = make_job(1, {torch::tensor({1})});
  job->set_on_complete([&probe](const auto& results, double latency) {
    probe.called = true;
    probe.results = results;
    probe.latency = latency;
  });
  probe.job = job;
  ASSERT_TRUE(queue_.push(job));
  ASSERT_TRUE(queue_.push(starpu_server::InferenceJob::make_shutdown_job()));
  runner_->run();
  assert_failure_result(probe);
}

TEST_F(StarPUTaskRunnerFixture, RunHandlesUnexpectedStdException)
{
  auto model_config = make_model_config(
      "test", {make_tensor_config("input0", {3}, at::kFloat)},
      {make_tensor_config("output0", {3}, at::kFloat)});

  reset_runner_with_model(model_config, /*input_slots=*/1);

  auto probe = starpu_server::make_callback_probe();
  auto job = make_job(
      9, {torch::ones({3}), torch::ones({3})}, {at::kFloat, at::kFloat});
  job->set_on_complete([&probe](const auto& results, double latency) {
    probe.called = true;
    probe.results = results;
    probe.latency = latency;
  });
  probe.job = job;

  ASSERT_TRUE(queue_.push(job));
  ASSERT_TRUE(queue_.push(starpu_server::InferenceJob::make_shutdown_job()));

  runner_->run();

  assert_failure_result(probe);
}

struct InvalidConfigParam {
  std::string name;
  std::function<void(starpu_server::StarPUTaskRunnerConfig&)> nullify;
  std::string field;
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
  std::condition_variable completion_cv;
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
  config.all_done_cv = &completion_cv;

  auto param = GetParam();
  param.nullify(config);

  auto assert_throws_with_field = [](auto&& ctor, const std::string& field) {
    try {
      ctor();
      ADD_FAILURE() << "Expected std::invalid_argument to be thrown";
    }
    catch (const std::invalid_argument& e) {
      EXPECT_NE(
          std::string{e.what()}.find(std::format(
              "StarPUTaskRunnerConfig::{} must not be null", field)),
          std::string::npos);
    }
  };

  assert_throws_with_field(
      [&] { starpu_server::StarPUTaskRunner runner(config); }, param.field);
}

INSTANTIATE_TEST_SUITE_P(
    InvalidConfig, StarPUTaskRunnerConfigInvalidTest,
    ::testing::Values(
        InvalidConfigParam{
            "Queue", [](auto& cfg) { cfg.queue = nullptr; }, "queue"},
        InvalidConfigParam{
            "ModelCpu", [](auto& cfg) { cfg.model_cpu = nullptr; },
            "model_cpu"},
        InvalidConfigParam{
            "ModelsGpu", [](auto& cfg) { cfg.models_gpu = nullptr; },
            "models_gpu"},
        InvalidConfigParam{
            "Starpu", [](auto& cfg) { cfg.starpu = nullptr; }, "starpu"},
        InvalidConfigParam{
            "Opts", [](auto& cfg) { cfg.opts = nullptr; }, "opts"},
        InvalidConfigParam{
            "Results", [](auto& cfg) { cfg.results = nullptr; }, "results"},
        InvalidConfigParam{
            "ResultsMutex", [](auto& cfg) { cfg.results_mutex = nullptr; },
            "results_mutex"},
        InvalidConfigParam{
            "CompletedJobs", [](auto& cfg) { cfg.completed_jobs = nullptr; },
            "completed_jobs"},
        InvalidConfigParam{
            "AllDoneCv", [](auto& cfg) { cfg.all_done_cv = nullptr; },
            "all_done_cv"}),
    [](const ::testing::TestParamInfo<InvalidConfigParam>& info) {
      return info.param.name;
    });
