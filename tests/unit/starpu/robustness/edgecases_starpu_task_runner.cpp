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
  auto job = std::make_shared<starpu_server::InferenceJob>();
  job->set_job_id(7);
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
  auto job = std::make_shared<starpu_server::InferenceJob>();
  job->set_job_id(8);
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
  auto job = probe.job;
  job->set_job_id(1);
  job->set_input_tensors({torch::tensor({1})});
  ASSERT_TRUE(queue_.push(job));
  ASSERT_TRUE(queue_.push(starpu_server::InferenceJob::make_shutdown_job()));
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

TEST_F(StarPUTaskRunnerFixture, RunHandlesUnexpectedStdException)
{
  runner_.reset();
  starpu_setup_.reset();

  starpu_server::ModelConfig model_config{};
  model_config.name = "test";
  starpu_server::TensorConfig input0{};
  input0.name = "input0";
  input0.dims = {3};
  input0.type = at::kFloat;
  model_config.inputs = {input0};

  starpu_server::TensorConfig output0{};
  output0.name = "output0";
  output0.dims = {3};
  output0.type = at::kFloat;
  model_config.outputs = {output0};

  opts_.models = {model_config};
  opts_.input_slots = 1;

  starpu_setup_ = std::make_unique<starpu_server::StarPUSetup>(opts_);
  config_.starpu = starpu_setup_.get();
  config_.opts = &opts_;
  runner_ = std::make_unique<starpu_server::StarPUTaskRunner>(config_);

  auto probe = starpu_server::make_callback_probe();
  auto job = probe.job;
  job->set_job_id(9);
  job->set_input_tensors({torch::ones({3}), torch::ones({3})});
  job->set_input_types({at::kFloat, at::kFloat});

  ASSERT_TRUE(queue_.push(job));
  ASSERT_TRUE(queue_.push(starpu_server::InferenceJob::make_shutdown_job()));

  runner_->run();

  EXPECT_TRUE(probe.called);
  EXPECT_TRUE(probe.results.empty());
  EXPECT_EQ(probe.latency, -1);

  ASSERT_EQ(results_.size(), 1U);
  EXPECT_TRUE(results_[0].results.empty());
  EXPECT_EQ(results_[0].latency_ms, -1);
  EXPECT_EQ(completed_jobs_.load(), 1);
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
