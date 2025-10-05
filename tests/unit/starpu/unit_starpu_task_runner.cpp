#include "core/inference_task.hpp"
#include "test_starpu_task_runner.hpp"
#include "utils/perf_observer.hpp"

namespace starpu_server {
class StarPUTaskRunnerTestAdapter {
 public:
  static void handle_submission_failure(
      InputSlotPool* input_pool, int input_slot, OutputSlotPool* output_pool,
      int output_slot, const std::shared_ptr<InferenceCallbackContext>& ctx,
      int submit_code)
  {
    StarPUTaskRunner::PoolResources pools{};
    pools.input_pool = input_pool;
    pools.input_slot = input_slot;
    pools.output_pool = output_pool;
    pools.output_slot = output_slot;
    StarPUTaskRunner::handle_submission_failure(pools, ctx, submit_code);
  }
};
}  // namespace starpu_server

TEST_F(StarPUTaskRunnerFixture, ShouldShutdown)
{
  auto shutdown_job = starpu_server::InferenceJob::make_shutdown_job();
  auto normal_job = make_job(0, {});
  EXPECT_TRUE(runner_->should_shutdown(shutdown_job));
  EXPECT_FALSE(runner_->should_shutdown(normal_job));
}

TEST_F(StarPUTaskRunnerFixture, PrepareJobCompletionCallback)
{
  auto probe = starpu_server::make_callback_probe();
  auto job = probe.job;
  constexpr int kJobId = 7;
  job->set_job_id(kJobId);
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
  EXPECT_EQ(results[0].job_id, kJobId);
  ASSERT_EQ(results[0].results.size(), outputs.size());
  EXPECT_TRUE(torch::equal(results[0].results[0], outputs[0]));
  ASSERT_EQ(probe.results.size(), outputs.size());
  EXPECT_TRUE(torch::equal(probe.results[0], outputs[0]));
  EXPECT_EQ(probe.latency, latency);
}

TEST_F(
    StarPUTaskRunnerFixture,
    PrepareJobCompletionCallbackRecordsInferenceWithoutInputs)
{
  starpu_server::perf_observer::reset();

  auto probe = starpu_server::make_callback_probe();
  auto job = probe.job;
  job->set_input_tensors({});

  runner_->prepare_job_completion_callback(job);

  const double latency = 3.0;
  job->get_on_complete()(std::vector<torch::Tensor>{}, latency);

  const auto stats = starpu_server::perf_observer::snapshot();
  ASSERT_TRUE(stats.has_value());
  EXPECT_EQ(stats->total_inferences, 1U);

  starpu_server::perf_observer::reset();
}

TEST_F(
    StarPUTaskRunnerFixture,
    PrepareJobCompletionCallbackRecordsInferenceWithScalarInput)
{
  starpu_server::perf_observer::reset();

  auto probe = starpu_server::make_callback_probe();
  auto job = probe.job;
  job->set_input_tensors({torch::tensor(1)});

  runner_->prepare_job_completion_callback(job);

  const double latency = 4.0;
  job->get_on_complete()(std::vector<torch::Tensor>{}, latency);

  const auto stats = starpu_server::perf_observer::snapshot();
  ASSERT_TRUE(stats.has_value());
  EXPECT_EQ(stats->total_inferences, 1U);

  starpu_server::perf_observer::reset();
}

TEST_F(StarPUTaskRunnerFixture, LogJobTimingsComputesComponents)
{
  opts_.verbosity = starpu_server::VerbosityLevel::Stats;
  starpu_server::detail::TimingInfo time;
  using clock = std::chrono::high_resolution_clock;
  auto base = clock::now();
  time.enqueued_time = base;
  constexpr int kQueueMs = 10;
  constexpr int kSubmitDeltaMs = 15;
  constexpr int kScheduleDeltaMs = 15;
  constexpr int kCodeletMs = 30;
  constexpr int kInferenceMs = 45;
  constexpr int kCallbackMs = 15;
  constexpr int kDequeuedMs = kQueueMs;
  constexpr int kBeforeSubmitMs = kDequeuedMs + kSubmitDeltaMs;
  constexpr int kCodeletStartMs = kBeforeSubmitMs + kScheduleDeltaMs;
  constexpr int kCodeletEndMs = kCodeletStartMs + kCodeletMs;
  constexpr int kInferenceStartMs = 80;
  constexpr int kCallbackStartMs = 125;
  constexpr int kCallbackEndMs = 140;
  time.dequeued_time = base + std::chrono::milliseconds(kDequeuedMs);
  time.before_starpu_submitted_time =
      base + std::chrono::milliseconds(kBeforeSubmitMs);
  time.codelet_start_time = base + std::chrono::milliseconds(kCodeletStartMs);
  time.codelet_end_time = base + std::chrono::milliseconds(kCodeletEndMs);
  time.inference_start_time =
      base + std::chrono::milliseconds(kInferenceStartMs);
  time.callback_start_time = base + std::chrono::milliseconds(kCallbackStartMs);
  time.callback_end_time = base + std::chrono::milliseconds(kCallbackEndMs);
  constexpr int kLogJobId = 42;
  constexpr double kTotalLatencyMs = 150.0;
  std::string output = starpu_server::capture_stdout(
      [&] { runner_->log_job_timings(kLogJobId, kTotalLatencyMs, time); });
  EXPECT_NE(output.find("Queue = 10.000 ms"), std::string::npos);
  EXPECT_NE(output.find("Submit = 15.000 ms"), std::string::npos);
  EXPECT_NE(output.find("Scheduling = 15.000 ms"), std::string::npos);
  EXPECT_NE(output.find("Codelet = 30.000 ms"), std::string::npos);
  EXPECT_NE(output.find("Inference = 45.000 ms"), std::string::npos);
  EXPECT_NE(output.find("Callback = 15.000 ms"), std::string::npos);
}

TEST_F(
    StarPUTaskRunnerFixture,
    SubmitInferenceTaskWithoutPoolsPropagatesExceptions)
{
  opts_.max_models_gpu = 0;
  models_gpu_.resize(1);

  auto job = make_job(42, {torch::ones({1})}, {at::kFloat});
  job->set_output_tensors({torch::zeros({1})});

  EXPECT_THROW(
      runner_->submit_inference_task(job),
      starpu_server::TooManyGpuModelsException);
}

TEST_F(
    StarPUTaskRunnerFixture, SubmitInferenceTaskWithPoolsReleasesSlotsOnFailure)
{
  auto model_config = make_model_config(
      "test",
      {make_tensor_config("input0", {3}, at::kFloat),
       make_tensor_config("input1", {3}, at::kFloat)},
      {make_tensor_config("output0", {3}, at::kFloat)});

  reset_runner_with_model(model_config, /*input_slots=*/1);

  auto job = make_job(7, {torch::ones({3})}, {at::kFloat});

  EXPECT_THROW(runner_->submit_inference_task(job), std::runtime_error);

  constexpr int kExpectedSlotId = 0;

  auto maybe_input_slot = starpu_setup_->input_pool().try_acquire();
  ASSERT_TRUE(maybe_input_slot.has_value());
  EXPECT_EQ(*maybe_input_slot, kExpectedSlotId);
  starpu_setup_->input_pool().release(*maybe_input_slot);

  auto maybe_output_slot = starpu_setup_->output_pool().try_acquire();
  ASSERT_TRUE(maybe_output_slot.has_value());
  EXPECT_EQ(*maybe_output_slot, kExpectedSlotId);
  starpu_setup_->output_pool().release(*maybe_output_slot);
}

TEST_F(
    StarPUTaskRunnerFixture,
    SubmitInferenceTaskWithOutputPoolReleasesSlotOnInputRegistrationFailure)
{
  auto model_config = make_model_config(
      "output_only", {}, {make_tensor_config("output0", {3}, at::kFloat)});

  reset_runner_with_model(model_config, /*input_slots=*/1);
  ASSERT_FALSE(starpu_setup_->has_input_pool());
  ASSERT_TRUE(starpu_setup_->has_output_pool());

  auto job = make_job(13, {torch::ones({2, 2})}, {at::kFloat});

  auto& stored_inputs =
      const_cast<std::vector<torch::Tensor>&>(job->get_input_tensors());
  stored_inputs[0] = stored_inputs[0].transpose(0, 1);
  ASSERT_FALSE(stored_inputs[0].is_contiguous());

  EXPECT_THROW(
      runner_->submit_inference_task(job),
      starpu_server::StarPURegistrationException);

  constexpr int kExpectedSlotId = 0;
  auto maybe_output_slot = starpu_setup_->output_pool().try_acquire();
  ASSERT_TRUE(maybe_output_slot.has_value());
  EXPECT_EQ(*maybe_output_slot, kExpectedSlotId);
  starpu_setup_->output_pool().release(*maybe_output_slot);
}

TEST_F(
    StarPUTaskRunnerFixture,
    SubmitInferenceTaskWithOutputPoolReleasesSlotOnTaskCreationFailure)
{
  auto model_config = make_model_config(
      "output_only", {}, {make_tensor_config("output0", {3}, at::kFloat)});

  auto deps = starpu_server::kDefaultInferenceTaskDependencies;
  deps.task_create_fn = []() -> starpu_task* { return nullptr; };
  reset_runner_with_model(model_config, /*input_slots=*/1, deps);
  ASSERT_FALSE(starpu_setup_->has_input_pool());
  ASSERT_TRUE(starpu_setup_->has_output_pool());

  auto job = make_job(17, {});

  EXPECT_THROW(
      runner_->submit_inference_task(job),
      starpu_server::StarPUTaskCreationException);

  constexpr int kExpectedSlotId = 0;
  auto maybe_output_slot = starpu_setup_->output_pool().try_acquire();
  ASSERT_TRUE(maybe_output_slot.has_value());
  EXPECT_EQ(*maybe_output_slot, kExpectedSlotId);
  starpu_setup_->output_pool().release(*maybe_output_slot);
}

TEST_F(
    StarPUTaskRunnerFixture,
    HandleSubmissionFailureReleasesSlotsThroughTestHook)
{
  auto model_config = make_model_config(
      "test", {make_tensor_config("input0", {3}, at::kFloat)},
      {make_tensor_config("output0", {3}, at::kFloat)});

  reset_runner_with_model(model_config, /*input_slots=*/1);

  auto& input_pool = starpu_setup_->input_pool();
  auto& output_pool = starpu_setup_->output_pool();

  const int input_slot = input_pool.acquire();
  const int output_slot = output_pool.acquire();

  EXPECT_THROW(
      starpu_server::StarPUTaskRunnerTestAdapter::handle_submission_failure(
          &input_pool, input_slot, &output_pool, output_slot, nullptr, -1),
      starpu_server::StarPUTaskSubmissionException);

  auto reacquired_input = input_pool.try_acquire();
  ASSERT_TRUE(reacquired_input.has_value());
  EXPECT_EQ(*reacquired_input, input_slot);
  input_pool.release(*reacquired_input);

  auto reacquired_output = output_pool.try_acquire();
  ASSERT_TRUE(reacquired_output.has_value());
  EXPECT_EQ(*reacquired_output, output_slot);
  output_pool.release(*reacquired_output);
}
