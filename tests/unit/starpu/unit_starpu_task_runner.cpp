#include <chrono>

#include "core/inference_task.hpp"
#include "exceptions.hpp"
#include "starpu_task_worker/task_runner_internal.hpp"
#include "test_starpu_task_runner.hpp"
#include "utils/perf_observer.hpp"

using starpu_server::CaptureStream;
using starpu_server::ErrorLevel;
using starpu_server::expected_log_line;

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

  static void propagate_completion_to_sub_jobs(
      const std::shared_ptr<InferenceJob>& aggregated_job,
      const std::vector<torch::Tensor>& aggregated_outputs, double latency_ms)
  {
    StarPUTaskRunner::propagate_completion_to_sub_jobs(
        aggregated_job, aggregated_outputs, latency_ms);
  }

  static auto maybe_build_batched_job(
      StarPUTaskRunner* runner,
      std::vector<std::shared_ptr<InferenceJob>>& jobs)
      -> std::shared_ptr<InferenceJob>
  {
    return runner->maybe_build_batched_job(jobs);
  }

  static auto can_merge_jobs(
      const std::shared_ptr<InferenceJob>& lhs,
      const std::shared_ptr<InferenceJob>& rhs) -> bool
  {
    return StarPUTaskRunner::can_merge_jobs(lhs, rhs);
  }

  static auto collect_batch(
      StarPUTaskRunner* runner, const std::shared_ptr<InferenceJob>& first_job)
      -> std::vector<std::shared_ptr<InferenceJob>>
  {
    return runner->collect_batch(first_job);
  }

  static void set_submit_hook(std::function<void()> hook)
  {
    task_runner_internal::set_submit_inference_task_hook(std::move(hook));
  }

  static void reset_submit_hook()
  {
    task_runner_internal::reset_submit_inference_task_hook();
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
  constexpr int kSubmissionId = 42;
  job->set_request_id(kJobId);
  job->set_submission_id(kSubmissionId);
  job->timing_info().submission_id = kSubmissionId;
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
  EXPECT_EQ(results[0].request_id, kJobId);
  EXPECT_EQ(results[0].submission_id, kSubmissionId);
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
  constexpr int kBatchMs = 5;
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
  time.batch_collect_start_time = time.dequeued_time;
  time.batch_collect_end_time =
      time.batch_collect_start_time + std::chrono::milliseconds(kBatchMs);
  time.before_starpu_submitted_time =
      base + std::chrono::milliseconds(kBeforeSubmitMs);
  time.codelet_start_time = base + std::chrono::milliseconds(kCodeletStartMs);
  time.codelet_end_time = base + std::chrono::milliseconds(kCodeletEndMs);
  time.inference_start_time =
      base + std::chrono::milliseconds(kInferenceStartMs);
  time.callback_start_time = base + std::chrono::milliseconds(kCallbackStartMs);
  time.callback_end_time = base + std::chrono::milliseconds(kCallbackEndMs);
  constexpr int kLogJobId = 42;
  constexpr auto kTotalLatencyMs =
      starpu_server::StarPUTaskRunner::DurationMs{150.0};
  std::string output = starpu_server::capture_stdout(
      [&] { runner_->log_job_timings(kLogJobId, kTotalLatencyMs, time); });
  EXPECT_NE(output.find("Queue = 10.000 ms"), std::string::npos);
  EXPECT_NE(output.find("Batch = 5.000 ms"), std::string::npos);
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
  opts_.limits.max_models_gpu = 0;
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

  reset_runner_with_model(model_config, /*pool_size=*/1);

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

  reset_runner_with_model(model_config, /*pool_size=*/1);
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
  reset_runner_with_model(model_config, /*pool_size=*/1, deps);
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

  reset_runner_with_model(model_config, /*pool_size=*/1);

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

TEST_F(
    StarPUTaskRunnerFixture,
    MaybeBuildBatchedJobAggregatesInputsAndPropagatesCallbacks)
{
  namespace internal = starpu_server::task_runner_internal;
  const auto base = internal::Clock::now();

  auto make_input = [](float a, float b) {
    return torch::tensor({{a, b}}, torch::TensorOptions().dtype(torch::kFloat));
  };

  auto job0_input = make_input(1.0F, 2.0F);
  auto job1_input = make_input(3.0F, 4.0F);
  auto job2_input = make_input(5.0F, 6.0F);

  auto job0 = make_job(0, {job0_input});
  auto job1 = make_job(1, {job1_input});
  auto job2 = make_job(2, {job2_input});

  job0->set_input_types({at::kFloat});
  job1->set_input_types({at::kFloat});
  job2->set_input_types({at::kFloat});

  job0->set_output_tensors(
      {torch::zeros({1, 2}, torch::TensorOptions().dtype(torch::kFloat))});
  job1->set_output_tensors(
      {torch::zeros({1, 2}, torch::TensorOptions().dtype(torch::kFloat))});
  job2->set_output_tensors(
      {torch::zeros({1, 2}, torch::TensorOptions().dtype(torch::kFloat))});

  job0->set_start_time(base + std::chrono::milliseconds(6));
  job1->set_start_time(base + std::chrono::milliseconds(3));
  job2->set_start_time(base + std::chrono::milliseconds(4));

  job0->timing_info().enqueued_time = base + std::chrono::milliseconds(6);
  job1->timing_info().enqueued_time = base + std::chrono::milliseconds(2);
  job2->timing_info().enqueued_time = base + std::chrono::milliseconds(4);

  job0->timing_info().batch_collect_start_time =
      base + std::chrono::milliseconds(7);
  job1->timing_info().batch_collect_start_time =
      base + std::chrono::milliseconds(2);
  job2->timing_info().batch_collect_start_time =
      base + std::chrono::milliseconds(1);
  job0->timing_info().dequeued_time = base + std::chrono::milliseconds(5);

  bool master_called = false;
  double master_latency = 0.0;
  std::vector<torch::Tensor> master_outputs;
  job0->set_on_complete(
      [&](const std::vector<torch::Tensor>& outputs, double latency_ms) {
        master_called = true;
        master_latency = latency_ms;
        master_outputs = outputs;
      });

  bool job1_called = false;
  double job1_latency = 0.0;
  std::vector<torch::Tensor> job1_outputs;
  job1->set_on_complete(
      [&](const std::vector<torch::Tensor>& outputs, double latency_ms) {
        job1_called = true;
        job1_latency = latency_ms;
        job1_outputs = outputs;
      });

  bool job2_called = false;
  double job2_latency = 0.0;
  std::vector<torch::Tensor> job2_outputs;
  job2->set_on_complete(
      [&](const std::vector<torch::Tensor>& outputs, double latency_ms) {
        job2_called = true;
        job2_latency = latency_ms;
        job2_outputs = outputs;
      });

  std::vector<std::shared_ptr<starpu_server::InferenceJob>> jobs{
      job0, job1, job2};

  auto master =
      starpu_server::StarPUTaskRunnerTestAdapter::maybe_build_batched_job(
          runner_.get(), jobs);

  ASSERT_EQ(master, job0);
  EXPECT_EQ(master->logical_job_count(), 3);
  const auto& aggregated_sub_jobs = master->aggregated_sub_jobs();
  ASSERT_EQ(aggregated_sub_jobs.size(), 3U);

  ASSERT_EQ(master->get_input_tensors().size(), 1U);
  auto expected_inputs =
      torch::cat({job0_input, job1_input, job2_input}, /*dim=*/0);
  EXPECT_TRUE(torch::equal(master->get_input_tensors()[0], expected_inputs));

  ASSERT_EQ(master->get_output_tensors().size(), 1U);
  EXPECT_EQ(master->get_output_tensors()[0].sizes()[0], 3);
  EXPECT_EQ(master->get_output_tensors()[0].sizes()[1], 2);

  EXPECT_EQ(master->get_start_time(), job1->get_start_time());
  EXPECT_EQ(
      master->timing_info().enqueued_time, job1->timing_info().enqueued_time);
  EXPECT_EQ(
      master->timing_info().batch_collect_start_time,
      job2->timing_info().batch_collect_start_time);

  EXPECT_TRUE(job1->get_input_tensors().empty());
  EXPECT_TRUE(job2->get_input_tensors().empty());

  master->get_device_id() = 11;
  master->get_worker_id() = 13;
  master->get_executed_on() = starpu_server::DeviceType::CUDA;
  master->set_submission_id(99);
  master->timing_info().submission_id = 99;

  const auto aggregated_primary = torch::tensor(
      {{11.0F, 12.0F}, {21.0F, 22.0F}, {31.0F, 32.0F}},
      torch::TensorOptions().dtype(torch::kFloat));
  const auto aggregated_aux = torch::tensor(
      {{1.0F}, {2.0F}, {3.0F}}, torch::TensorOptions().dtype(torch::kFloat));
  const double latency_ms = 9.5;

  master->get_on_complete()({aggregated_primary, aggregated_aux}, latency_ms);

  EXPECT_TRUE(master_called);
  EXPECT_TRUE(job1_called);
  EXPECT_TRUE(job2_called);
  EXPECT_EQ(master_latency, latency_ms);
  EXPECT_EQ(job1_latency, latency_ms);
  EXPECT_EQ(job2_latency, latency_ms);

  ASSERT_EQ(master_outputs.size(), 2U);
  EXPECT_TRUE(torch::equal(
      master_outputs[0], aggregated_primary.narrow(/*dim=*/0, /*start=*/0, 1)));
  EXPECT_TRUE(torch::equal(
      master_outputs[1], aggregated_aux.narrow(/*dim=*/0, /*start=*/0, 1)));

  ASSERT_EQ(job1_outputs.size(), 2U);
  EXPECT_TRUE(torch::equal(
      job1_outputs[0], aggregated_primary.narrow(/*dim=*/0, /*start=*/1, 1)));
  EXPECT_TRUE(torch::equal(
      job1_outputs[1], aggregated_aux.narrow(/*dim=*/0, /*start=*/1, 1)));

  ASSERT_EQ(job2_outputs.size(), 2U);
  EXPECT_TRUE(torch::equal(
      job2_outputs[0], aggregated_primary.narrow(/*dim=*/0, /*start=*/2, 1)));
  EXPECT_TRUE(torch::equal(
      job2_outputs[1], aggregated_aux.narrow(/*dim=*/0, /*start=*/2, 1)));

  EXPECT_EQ(job1->timing_info().submission_id, master->submission_id());
  EXPECT_EQ(job2->timing_info().submission_id, master->submission_id());
  EXPECT_EQ(job1->get_device_id(), master->get_device_id());
  EXPECT_EQ(job2->get_device_id(), master->get_device_id());
  EXPECT_EQ(job1->get_worker_id(), master->get_worker_id());
  EXPECT_EQ(job2->get_worker_id(), master->get_worker_id());
  EXPECT_EQ(job1->get_executed_on(), master->get_executed_on());
  EXPECT_EQ(job2->get_executed_on(), master->get_executed_on());
  EXPECT_EQ(
      job1->timing_info().enqueued_time, master->timing_info().enqueued_time);
  EXPECT_EQ(
      job2->timing_info().batch_collect_start_time,
      master->timing_info().batch_collect_start_time);
}

TEST(
    StarPUTaskRunnerTestAdapter,
    PropagateCompletionToSubJobsDistributesSlicesAndMetadata)
{
  namespace internal = starpu_server::task_runner_internal;
  auto aggregated = std::make_shared<starpu_server::InferenceJob>();
  auto job_one = std::make_shared<starpu_server::InferenceJob>();
  auto job_two = std::make_shared<starpu_server::InferenceJob>();

  aggregated->set_submission_id(77);
  aggregated->timing_info().submission_id = 77;
  aggregated->get_device_id() = 5;
  aggregated->get_worker_id() = 7;
  aggregated->get_executed_on() = starpu_server::DeviceType::CUDA;

  const auto base = internal::Clock::now();
  aggregated->set_start_time(base + std::chrono::milliseconds(10));
  aggregated->timing_info().enqueued_time = base + std::chrono::milliseconds(2);
  aggregated->timing_info().batch_collect_start_time =
      base + std::chrono::milliseconds(3);

  bool job_one_called = false;
  double job_one_latency = 0.0;
  std::vector<torch::Tensor> job_one_outputs;
  job_one->set_on_complete(
      [&](const std::vector<torch::Tensor>& outputs, double latency_ms) {
        job_one_called = true;
        job_one_latency = latency_ms;
        job_one_outputs = outputs;
      });

  bool job_two_called = false;
  double job_two_latency = 0.0;
  std::vector<torch::Tensor> job_two_outputs;
  job_two->set_on_complete(
      [&](const std::vector<torch::Tensor>& outputs, double latency_ms) {
        job_two_called = true;
        job_two_latency = latency_ms;
        job_two_outputs = outputs;
      });

  std::vector<starpu_server::InferenceJob::AggregatedSubJob> sub_jobs;
  sub_jobs.emplace_back(job_one, job_one->get_on_complete(), 1);
  sub_jobs.emplace_back(job_two, job_two->get_on_complete(), 2);
  aggregated->set_aggregated_sub_jobs(std::move(sub_jobs));

  const auto primary = torch::tensor(
      {{1.0F, 2.0F}, {3.0F, 4.0F}, {5.0F, 6.0F}},
      torch::TensorOptions().dtype(torch::kFloat));
  const auto secondary = torch::tensor(
      {{10.0F}, {20.0F}, {30.0F}}, torch::TensorOptions().dtype(torch::kFloat));
  const double latency_ms = 12.5;

  starpu_server::StarPUTaskRunnerTestAdapter::propagate_completion_to_sub_jobs(
      aggregated, {primary, secondary}, latency_ms);

  EXPECT_TRUE(job_one_called);
  EXPECT_TRUE(job_two_called);
  EXPECT_EQ(job_one_latency, latency_ms);
  EXPECT_EQ(job_two_latency, latency_ms);

  ASSERT_EQ(job_one_outputs.size(), 2U);
  EXPECT_TRUE(torch::equal(
      job_one_outputs[0], primary.narrow(/*dim=*/0, /*start=*/0, 1)));
  EXPECT_TRUE(torch::equal(
      job_one_outputs[1], secondary.narrow(/*dim=*/0, /*start=*/0, 1)));

  ASSERT_EQ(job_two_outputs.size(), 2U);
  EXPECT_TRUE(torch::equal(
      job_two_outputs[0], primary.narrow(/*dim=*/0, /*start=*/1, 2)));
  EXPECT_TRUE(torch::equal(
      job_two_outputs[1], secondary.narrow(/*dim=*/0, /*start=*/1, 2)));

  EXPECT_EQ(job_one->timing_info().submission_id, aggregated->submission_id());
  EXPECT_EQ(job_two->timing_info().submission_id, aggregated->submission_id());
  EXPECT_EQ(job_one->get_device_id(), aggregated->get_device_id());
  EXPECT_EQ(job_two->get_device_id(), aggregated->get_device_id());
  EXPECT_EQ(job_one->get_worker_id(), aggregated->get_worker_id());
  EXPECT_EQ(job_two->get_worker_id(), aggregated->get_worker_id());
  EXPECT_EQ(job_one->get_executed_on(), aggregated->get_executed_on());
  EXPECT_EQ(job_two->get_executed_on(), aggregated->get_executed_on());
  EXPECT_EQ(
      job_one->timing_info().enqueued_time,
      aggregated->timing_info().enqueued_time);
  EXPECT_EQ(
      job_two->timing_info().batch_collect_start_time,
      aggregated->timing_info().batch_collect_start_time);
}

TEST(TaskRunnerInternal, SliceOutputsForSubJobReturnsDefaultLengthWhenEmpty)
{
  namespace internal = starpu_server::task_runner_internal;

  const auto result = internal::slice_outputs_for_sub_job(
      {}, internal::SubJobSliceOptions{0, 3});

  EXPECT_TRUE(result.outputs.empty());
  EXPECT_EQ(result.processed_length, 3);
}

TEST(TaskRunnerInternal, SliceOutputsForSubJobExtractsContiguousRows)
{
  namespace internal = starpu_server::task_runner_internal;

  auto first = torch::arange(0, 12, torch::TensorOptions().dtype(torch::kInt64))
                   .reshape({4, 3});
  auto second =
      torch::arange(100, 112, torch::TensorOptions().dtype(torch::kInt64))
          .reshape({4, 3});
  std::vector<torch::Tensor> aggregated{first, second};

  const auto result = internal::slice_outputs_for_sub_job(
      aggregated, internal::SubJobSliceOptions{1, 2});

  ASSERT_EQ(result.outputs.size(), 2U);
  EXPECT_EQ(result.processed_length, 2);
  auto expected_first = first.narrow(0, 1, 2).contiguous();
  auto expected_second = second.narrow(0, 1, 2).contiguous();
  EXPECT_TRUE(torch::equal(result.outputs[0], expected_first));
  EXPECT_TRUE(result.outputs[0].is_contiguous());
  EXPECT_TRUE(torch::equal(result.outputs[1], expected_second));
  EXPECT_TRUE(result.outputs[1].is_contiguous());
}

TEST(
    TaskRunnerInternal,
    SliceOutputsForSubJobPreservesUndefinedAndZeroDimTensors)
{
  namespace internal = starpu_server::task_runner_internal;

  torch::Tensor undefined_tensor;
  auto scalar_tensor =
      torch::tensor(42, torch::TensorOptions().dtype(torch::kInt64));
  auto matrix = torch::arange(0, 6, torch::TensorOptions().dtype(torch::kInt64))
                    .reshape({3, 2});
  std::vector<torch::Tensor> aggregated{
      undefined_tensor, scalar_tensor, matrix};

  const auto result = internal::slice_outputs_for_sub_job(
      aggregated, internal::SubJobSliceOptions{0, 1});

  ASSERT_EQ(result.outputs.size(), aggregated.size());
  EXPECT_FALSE(result.outputs[0].defined());
  ASSERT_TRUE(result.outputs[1].defined());
  EXPECT_EQ(result.outputs[1].dim(), 0);
  EXPECT_EQ(result.outputs[1].item<int64_t>(), scalar_tensor.item<int64_t>());
  auto expected_matrix_slice = matrix.narrow(0, 0, 1).contiguous();
  EXPECT_TRUE(torch::equal(result.outputs[2], expected_matrix_slice));
  EXPECT_EQ(result.processed_length, 1);
}

TEST(
    TaskRunnerInternal,
    SliceOutputsForSubJobYieldsEmptySliceWhenOffsetExceedsData)
{
  namespace internal = starpu_server::task_runner_internal;

  auto tensor = torch::arange(0, 6, torch::TensorOptions().dtype(torch::kInt64))
                    .reshape({3, 2});
  std::vector<torch::Tensor> aggregated{tensor};

  const auto result = internal::slice_outputs_for_sub_job(
      aggregated, internal::SubJobSliceOptions{5, 2});

  ASSERT_EQ(result.outputs.size(), 1U);
  EXPECT_FALSE(result.outputs[0].defined());
  EXPECT_EQ(result.processed_length, 2);
}

TEST(TaskRunnerInternal, AggregateBatchMetadataReturnsDefaultsForEmptyJobs)
{
  namespace internal = starpu_server::task_runner_internal;

  const auto info = internal::aggregate_batch_metadata({});

  EXPECT_TRUE(info.sub_jobs.empty());
  EXPECT_EQ(info.logical_jobs, 0);
  EXPECT_EQ(info.total_samples, 0);
  EXPECT_EQ(info.earliest_start, internal::Clock::time_point{});
  EXPECT_EQ(info.earliest_enqueued, internal::Clock::time_point{});
  EXPECT_EQ(info.earliest_batch_collect_start, internal::Clock::time_point{});
}

TEST(TaskRunnerInternal, AggregateBatchMetadataCollectsEarliestTimings)
{
  namespace internal = starpu_server::task_runner_internal;

  auto job_one = std::make_shared<starpu_server::InferenceJob>();
  auto job_two = std::make_shared<starpu_server::InferenceJob>();

  job_one->set_input_tensors({torch::ones({2, 1})});
  job_two->set_input_tensors({torch::ones({3, 1})});
  job_one->set_logical_job_count(2);
  job_two->set_logical_job_count(0);

  bool first_called = false;
  bool second_called = false;
  job_one->set_on_complete(
      [&first_called](const std::vector<torch::Tensor>&, double) {
        first_called = true;
      });
  job_two->set_on_complete(
      [&second_called](const std::vector<torch::Tensor>&, double) {
        second_called = true;
      });

  const auto base = internal::Clock::now();
  job_one->set_start_time(base + std::chrono::milliseconds(5));
  job_two->set_start_time(base + std::chrono::milliseconds(3));
  job_one->timing_info().enqueued_time = base + std::chrono::milliseconds(6);
  job_two->timing_info().enqueued_time = base + std::chrono::milliseconds(4);
  job_one->timing_info().batch_collect_start_time =
      base + std::chrono::milliseconds(7);
  job_two->timing_info().batch_collect_start_time =
      base + std::chrono::milliseconds(2);

  const auto info = internal::aggregate_batch_metadata({job_one, job_two});

  EXPECT_EQ(info.total_samples, 5);
  EXPECT_EQ(info.logical_jobs, 3);
  EXPECT_EQ(info.sub_jobs.size(), 2U);
  EXPECT_EQ(info.sub_jobs[0].batch_size, 2);
  EXPECT_EQ(info.sub_jobs[1].batch_size, 3);
  EXPECT_EQ(info.earliest_start, job_two->get_start_time());
  EXPECT_EQ(info.earliest_enqueued, job_two->timing_info().enqueued_time);
  EXPECT_EQ(
      info.earliest_batch_collect_start,
      job_two->timing_info().batch_collect_start_time);

  auto locked_one = info.sub_jobs[0].job.lock();
  auto locked_two = info.sub_jobs[1].job.lock();
  ASSERT_TRUE(locked_one);
  ASSERT_TRUE(locked_two);
  EXPECT_EQ(locked_one, job_one);
  EXPECT_EQ(locked_two, job_two);

  ASSERT_TRUE(info.sub_jobs[0].callback);
  ASSERT_TRUE(info.sub_jobs[1].callback);
  info.sub_jobs[0].callback({}, 0.0);
  info.sub_jobs[1].callback({}, 0.0);
  EXPECT_TRUE(first_called);
  EXPECT_TRUE(second_called);
}

TEST(
    TaskRunnerInternal,
    AggregateBatchMetadataRetainsExistingTimesWhenCandidateUnset)
{
  namespace internal = starpu_server::task_runner_internal;

  auto job_one = std::make_shared<starpu_server::InferenceJob>();
  auto job_two = std::make_shared<starpu_server::InferenceJob>();

  job_one->set_input_tensors({torch::ones({2, 1})});
  job_two->set_input_tensors({torch::ones({1, 1})});

  const auto base = internal::Clock::now();
  job_one->set_start_time(base + std::chrono::milliseconds(5));
  job_one->timing_info().enqueued_time = base + std::chrono::milliseconds(6);
  job_one->timing_info().batch_collect_start_time =
      base + std::chrono::milliseconds(7);

  const auto info = internal::aggregate_batch_metadata({job_one, job_two});

  EXPECT_EQ(info.earliest_start, job_one->get_start_time());
  EXPECT_EQ(info.earliest_enqueued, job_one->timing_info().enqueued_time);
  EXPECT_EQ(
      info.earliest_batch_collect_start,
      job_one->timing_info().batch_collect_start_time);
}

TEST(TaskRunnerInternal, RunWithLoggedExceptionsLogsInferenceEngineException)
{
  auto job = std::make_shared<starpu_server::InferenceJob>();
  job->set_on_complete([](const std::vector<torch::Tensor>&, double) {
    throw starpu_server::InferenceEngineException("inference failure");
  });

  CaptureStream capture{std::cerr};
  starpu_server::StarPUTaskRunner::handle_job_exception(
      job, std::runtime_error("outer failure"));

  const auto logs = capture.str();
  const auto expected = expected_log_line(
      ErrorLevel, "Exception in completion callback: inference failure");
  EXPECT_NE(logs.find(expected), std::string::npos);
}

TEST(TaskRunnerInternal, RunWithLoggedExceptionsLogsLogicError)
{
  auto job = std::make_shared<starpu_server::InferenceJob>();
  job->set_on_complete([](const std::vector<torch::Tensor>&, double) {
    throw std::logic_error("logic failure");
  });

  CaptureStream capture{std::cerr};
  starpu_server::StarPUTaskRunner::handle_job_exception(
      job, std::runtime_error("outer failure"));

  const auto logs = capture.str();
  const auto expected = expected_log_line(
      ErrorLevel, "Exception in completion callback: logic failure");
  EXPECT_NE(logs.find(expected), std::string::npos);
}

TEST(TaskRunnerInternal, RunWithLoggedExceptionsLogsBadAlloc)
{
  auto job = std::make_shared<starpu_server::InferenceJob>();
  job->set_on_complete([](const std::vector<torch::Tensor>&, double) {
    throw std::bad_alloc();
  });

  CaptureStream capture{std::cerr};
  starpu_server::StarPUTaskRunner::handle_job_exception(
      job, std::runtime_error("outer failure"));

  const auto logs = capture.str();
  const auto expected = expected_log_line(
      ErrorLevel, "Exception in completion callback: std::bad_alloc");
  const auto alt_expected = expected_log_line(
      ErrorLevel, "Exception in completion callback: bad_alloc");
  EXPECT_TRUE(
      logs.find(expected) != std::string::npos ||
      logs.find(alt_expected) != std::string::npos);
}

TEST(TaskRunnerInternal, ResizeOutputsForBatchRespectsPrototypeLayout)
{
  namespace internal = starpu_server::task_runner_internal;

  torch::Tensor defined =
      torch::ones({2, 4}, torch::TensorOptions().dtype(torch::kFloat));
  torch::Tensor scalar = torch::tensor(42.0);
  torch::Tensor undefined;

  const auto resized =
      internal::resize_outputs_for_batch({defined, scalar, undefined}, 5);

  ASSERT_EQ(resized.size(), 3U);
  EXPECT_TRUE(resized[0].is_contiguous());
  EXPECT_EQ(resized[0].sizes()[0], 5);
  EXPECT_EQ(resized[0].sizes()[1], 4);
  EXPECT_EQ(resized[0].dtype(), defined.dtype());
  EXPECT_EQ(resized[1].dim(), 0);
  EXPECT_FALSE(resized[2].defined());
}

TEST(TaskRunnerInternal, ReleaseInputsFromAdditionalJobsClearsExtraEntries)
{
  namespace internal = starpu_server::task_runner_internal;

  auto make_job = [](int id, int value) {
    auto job = std::make_shared<starpu_server::InferenceJob>();
    job->set_request_id(id);
    job->set_input_tensors(
        {torch::full({1}, value, torch::TensorOptions().dtype(torch::kInt32))});
    job->set_input_memory_holders(
        {std::shared_ptr<const void>{nullptr, [](const void*) {}}});
    return job;
  };

  auto job_zero = make_job(0, 1);
  auto job_one = make_job(1, 2);
  auto job_two = make_job(2, 3);

  std::vector<std::shared_ptr<starpu_server::InferenceJob>> jobs = {
      job_zero, job_one, job_two};

  internal::release_inputs_from_additional_jobs(jobs);

  EXPECT_FALSE(job_zero->get_input_tensors().empty());
  EXPECT_EQ(job_one->get_input_tensors().size(), 0U);
  EXPECT_EQ(job_two->get_input_tensors().size(), 0U);
  EXPECT_TRUE(job_one->get_input_memory_holders().empty());
  EXPECT_TRUE(job_two->get_input_memory_holders().empty());
  EXPECT_FALSE(job_zero->get_input_memory_holders().empty());
}

TEST_F(
    StarPUTaskRunnerFixture,
    CollectBatchReturnsOnlyFirstWhenDynamicBatchingDisabled)
{
  opts_.batching.dynamic_batching = false;

  auto first = make_job(
      0, {torch::ones({1, 2}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});

  auto collected = starpu_server::StarPUTaskRunnerTestAdapter::collect_batch(
      runner_.get(), first);

  ASSERT_EQ(collected.size(), 1U);
  EXPECT_EQ(collected.front(), first);
  EXPECT_EQ(queue_.size(), 0U);
}

TEST_F(StarPUTaskRunnerFixture, CollectBatchAggregatesCompatibleQueuedJobs)
{
  opts_.batching.dynamic_batching = true;
  opts_.batching.max_batch_size = 4;
  opts_.batching.batch_coalesce_timeout_ms = 0;

  auto first = make_job(
      0, {torch::ones({1, 2}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  auto second = make_job(
      1, {torch::ones({1, 2}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});

  ASSERT_TRUE(queue_.push(second));

  auto collected = starpu_server::StarPUTaskRunnerTestAdapter::collect_batch(
      runner_.get(), first);

  ASSERT_EQ(collected.size(), 2U);
  EXPECT_EQ(collected[0], first);
  EXPECT_EQ(collected[1], second);
  EXPECT_EQ(queue_.size(), 0U);
}

TEST_F(StarPUTaskRunnerFixture, CollectBatchStoresNonMergeableJobAsPending)
{
  opts_.batching.dynamic_batching = true;
  opts_.batching.max_batch_size = 4;
  opts_.batching.batch_coalesce_timeout_ms = 0;

  auto first = make_job(
      0, {torch::ones({1, 2}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  auto incompatible = make_job(
      1, {torch::ones({1, 3}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});

  ASSERT_TRUE(queue_.push(incompatible));

  auto collected = starpu_server::StarPUTaskRunnerTestAdapter::collect_batch(
      runner_.get(), first);

  ASSERT_EQ(collected.size(), 1U);
  EXPECT_EQ(collected.front(), first);
  EXPECT_EQ(queue_.size(), 0U);

  auto pending = runner_->wait_for_next_job();
  ASSERT_EQ(pending, incompatible);
}

TEST_F(StarPUTaskRunnerFixture, CollectBatchRespectsConfiguredMaximumBatchSize)
{
  opts_.batching.dynamic_batching = true;
  opts_.batching.max_batch_size = 2;
  opts_.batching.batch_coalesce_timeout_ms = 0;

  auto first = make_job(
      0, {torch::ones({1, 2}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  auto second = make_job(
      1, {torch::ones({1, 2}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  auto third = make_job(
      2, {torch::ones({1, 2}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});

  ASSERT_TRUE(queue_.push(second));
  ASSERT_TRUE(queue_.push(third));

  auto collected = starpu_server::StarPUTaskRunnerTestAdapter::collect_batch(
      runner_.get(), first);

  ASSERT_EQ(collected.size(), 2U);
  EXPECT_EQ(collected[0], first);
  EXPECT_EQ(collected[1], second);
  EXPECT_EQ(queue_.size(), 1U);
}

TEST_F(StarPUTaskRunnerFixture, RunCatchesInferenceEngineException)
{
  opts_.batching.dynamic_batching = false;

  auto probe = starpu_server::make_callback_probe();
  auto job = probe.job;
  job->set_input_tensors(
      {torch::ones({1}, torch::TensorOptions().dtype(torch::kFloat))});
  job->set_input_types({at::kFloat});

  ASSERT_TRUE(queue_.push(job));
  ASSERT_TRUE(queue_.push(starpu_server::InferenceJob::make_shutdown_job()));

  starpu_server::StarPUTaskRunnerTestAdapter::set_submit_hook([&]() {
    starpu_server::StarPUTaskRunnerTestAdapter::reset_submit_hook();
    throw starpu_server::InferenceEngineException("test inference failure");
  });

  runner_->run();
  starpu_server::StarPUTaskRunnerTestAdapter::reset_submit_hook();

  EXPECT_TRUE(probe.called);
  ASSERT_EQ(results_.size(), 1U);
  EXPECT_EQ(results_[0].latency_ms, -1);
  EXPECT_EQ(completed_jobs_.load(), 1);
  EXPECT_EQ(queue_.size(), 0U);
}

TEST_F(StarPUTaskRunnerFixture, RunCatchesRuntimeError)
{
  opts_.batching.dynamic_batching = false;

  auto probe = starpu_server::make_callback_probe();
  auto job = probe.job;
  job->set_input_tensors(
      {torch::ones({1}, torch::TensorOptions().dtype(torch::kFloat))});
  job->set_input_types({at::kFloat});

  ASSERT_TRUE(queue_.push(job));
  ASSERT_TRUE(queue_.push(starpu_server::InferenceJob::make_shutdown_job()));

  starpu_server::StarPUTaskRunnerTestAdapter::set_submit_hook([&]() {
    starpu_server::StarPUTaskRunnerTestAdapter::reset_submit_hook();
    throw std::runtime_error("runtime failure");
  });

  runner_->run();
  starpu_server::StarPUTaskRunnerTestAdapter::reset_submit_hook();

  EXPECT_TRUE(probe.called);
  ASSERT_EQ(results_.size(), 1U);
  EXPECT_EQ(results_[0].latency_ms, -1);
  EXPECT_EQ(completed_jobs_.load(), 1);
  EXPECT_EQ(queue_.size(), 0U);
}

TEST_F(StarPUTaskRunnerFixture, RunCatchesLogicError)
{
  opts_.batching.dynamic_batching = false;

  auto probe = starpu_server::make_callback_probe();
  auto job = probe.job;
  job->set_input_tensors(
      {torch::ones({1}, torch::TensorOptions().dtype(torch::kFloat))});
  job->set_input_types({at::kFloat});

  ASSERT_TRUE(queue_.push(job));
  ASSERT_TRUE(queue_.push(starpu_server::InferenceJob::make_shutdown_job()));

  starpu_server::StarPUTaskRunnerTestAdapter::set_submit_hook([&]() {
    starpu_server::StarPUTaskRunnerTestAdapter::reset_submit_hook();
    throw std::logic_error("logic failure");
  });

  runner_->run();
  starpu_server::StarPUTaskRunnerTestAdapter::reset_submit_hook();

  EXPECT_TRUE(probe.called);
  ASSERT_EQ(results_.size(), 1U);
  EXPECT_EQ(results_[0].latency_ms, -1);
  EXPECT_EQ(completed_jobs_.load(), 1);
  EXPECT_EQ(queue_.size(), 0U);
}

TEST_F(StarPUTaskRunnerFixture, RunCatchesBadAlloc)
{
  opts_.batching.dynamic_batching = false;

  auto probe = starpu_server::make_callback_probe();
  auto job = probe.job;
  job->set_input_tensors(
      {torch::ones({1}, torch::TensorOptions().dtype(torch::kFloat))});
  job->set_input_types({at::kFloat});

  ASSERT_TRUE(queue_.push(job));
  ASSERT_TRUE(queue_.push(starpu_server::InferenceJob::make_shutdown_job()));

  starpu_server::StarPUTaskRunnerTestAdapter::set_submit_hook([&]() {
    starpu_server::StarPUTaskRunnerTestAdapter::reset_submit_hook();
    throw std::bad_alloc();
  });

  runner_->run();
  starpu_server::StarPUTaskRunnerTestAdapter::reset_submit_hook();

  EXPECT_TRUE(probe.called);
  ASSERT_EQ(results_.size(), 1U);
  EXPECT_EQ(results_[0].latency_ms, -1);
  EXPECT_EQ(completed_jobs_.load(), 1);
  EXPECT_EQ(queue_.size(), 0U);
}

TEST(
    StarPUTaskRunnerTestAdapter,
    CanMergeJobsReturnsTrueForCompatibleInputsAndTypes)
{
  auto make_job = [] {
    auto job = std::make_shared<starpu_server::InferenceJob>();
    job->set_input_tensors(
        {torch::ones({2, 3}, torch::TensorOptions().dtype(torch::kFloat)),
         torch::ones({2}, torch::TensorOptions().dtype(torch::kFloat))});
    job->set_input_types({at::kFloat, at::kFloat});
    return job;
  };

  auto lhs = make_job();
  auto rhs = make_job();

  EXPECT_TRUE(
      starpu_server::StarPUTaskRunnerTestAdapter::can_merge_jobs(lhs, rhs));
}

TEST(StarPUTaskRunnerTestAdapter, CanMergeJobsRejectsMismatchedShapesOrTypes)
{
  auto base_job = std::make_shared<starpu_server::InferenceJob>();
  base_job->set_input_tensors(
      {torch::ones({2, 3}, torch::TensorOptions().dtype(torch::kFloat)),
       torch::ones({2}, torch::TensorOptions().dtype(torch::kFloat))});
  base_job->set_input_types({at::kFloat, at::kFloat});

  auto shape_mismatch = std::make_shared<starpu_server::InferenceJob>();
  shape_mismatch->set_input_tensors(
      {torch::ones({2, 4}, torch::TensorOptions().dtype(torch::kFloat)),
       torch::ones({2}, torch::TensorOptions().dtype(torch::kFloat))});
  shape_mismatch->set_input_types({at::kFloat, at::kFloat});

  auto type_mismatch = std::make_shared<starpu_server::InferenceJob>();
  type_mismatch->set_input_tensors(
      {torch::ones({2, 3}, torch::TensorOptions().dtype(torch::kFloat)),
       torch::ones({2}, torch::TensorOptions().dtype(torch::kFloat))});
  type_mismatch->set_input_types({at::kFloat, at::kHalf});

  EXPECT_FALSE(starpu_server::StarPUTaskRunnerTestAdapter::can_merge_jobs(
      base_job, shape_mismatch));
  EXPECT_FALSE(starpu_server::StarPUTaskRunnerTestAdapter::can_merge_jobs(
      base_job, type_mismatch));
}
