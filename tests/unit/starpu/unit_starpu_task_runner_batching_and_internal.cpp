#include "unit_starpu_task_runner_support.hpp"

#define private public
#include "starpu_task_worker/result_dispatcher_component.hpp"
#undef private

TEST_F(StarPUTaskRunnerFixture, MaybeBuildBatchedJobReturnsNullWhenNoJobs)
{
  std::vector<std::shared_ptr<starpu_server::InferenceJob>> jobs;
  auto master =
      starpu_server::StarPUTaskRunnerTestAdapter::maybe_build_batched_job(
          runner_.get(), jobs);
  EXPECT_EQ(master, nullptr);
}

TEST_F(StarPUTaskRunnerFixture, MaybeBuildBatchedJobSingleJobResetsState)
{
  auto job = make_job(
      5, {torch::ones({1, 2}, torch::TensorOptions().dtype(torch::kFloat))});
  job->set_logical_job_count(4);
  starpu_server::InferenceJob::AggregatedSubJob sub_job{};
  sub_job.job = job;
  sub_job.callback = [](const std::vector<torch::Tensor>&, double) {};
  sub_job.batch_size = 3;
  sub_job.request_id = job->get_request_id();
  job->set_aggregated_sub_jobs({sub_job});

  bool callback_called = false;
  job->set_on_complete([&](const std::vector<torch::Tensor>&, double) {
    callback_called = true;
  });

  std::vector<std::shared_ptr<starpu_server::InferenceJob>> jobs{job};

  auto master =
      starpu_server::StarPUTaskRunnerTestAdapter::maybe_build_batched_job(
          runner_.get(), jobs);

  ASSERT_EQ(master, job);
  EXPECT_EQ(master->logical_job_count(), 1);
  EXPECT_TRUE(master->aggregated_sub_jobs().empty());

  master->get_on_complete()({}, 0.0);
  EXPECT_TRUE(callback_called);
}

TEST_F(
    StarPUTaskRunnerFixture,
    MaybeBuildBatchedJobFallsBackToEarliestTimesAndMergesMemory)
{
  namespace internal = starpu_server::task_runner_internal;
  const auto base = internal::Clock::now();

  auto job0 = make_job(
      0, {torch::tensor(
             {{1.0F, 2.0F}}, torch::TensorOptions().dtype(torch::kFloat))});
  auto job1 = make_job(
      1, {torch::tensor(
             {{3.0F, 4.0F}}, torch::TensorOptions().dtype(torch::kFloat))});

  job0->set_input_types({at::kFloat});
  job1->set_input_types({at::kFloat});

  job0->set_output_tensors(
      {torch::zeros({1, 2}, torch::TensorOptions().dtype(torch::kFloat))});
  job1->set_output_tensors(
      {torch::zeros({1, 2}, torch::TensorOptions().dtype(torch::kFloat))});

  auto holder0 = std::make_shared<int>(1);
  auto holder1 = std::make_shared<int>(2);
  job0->set_input_memory_holders(
      {std::shared_ptr<const void>(holder0, holder0.get())});
  job1->set_input_memory_holders(
      {std::shared_ptr<const void>(holder1, holder1.get())});

  job0->timing_info().dequeued_time = base + std::chrono::milliseconds(5);
  job0->timing_info().enqueued_time = internal::Clock::time_point{};
  job0->timing_info().last_enqueued_time = internal::Clock::time_point{};
  job0->timing_info().batch_collect_start_time = internal::Clock::time_point{};
  job1->timing_info().enqueued_time = base + std::chrono::milliseconds(8);
  job1->timing_info().last_enqueued_time = job1->timing_info().enqueued_time;
  job1->timing_info().batch_collect_start_time = internal::Clock::time_point{};

  bool job1_called = false;
  job1->set_on_complete(
      [&](const std::vector<torch::Tensor>&, double) { job1_called = true; });

  std::vector<std::shared_ptr<starpu_server::InferenceJob>> jobs{job0, job1};

  auto master =
      starpu_server::StarPUTaskRunnerTestAdapter::maybe_build_batched_job(
          runner_.get(), jobs);

  ASSERT_EQ(master, job0);
  EXPECT_EQ(master->logical_job_count(), 2);
  EXPECT_EQ(master->aggregated_sub_jobs().size(), 2U);

  EXPECT_EQ(master->get_start_time(), job1->timing_info().enqueued_time);
  EXPECT_EQ(
      master->timing_info().enqueued_time, job1->timing_info().enqueued_time);
  EXPECT_EQ(
      master->timing_info().batch_collect_start_time,
      job0->timing_info().dequeued_time);

  const auto& holders = master->get_input_memory_holders();
  ASSERT_EQ(holders.size(), 2U);
  EXPECT_EQ(holders.front().get(), static_cast<const void*>(holder0.get()));
  EXPECT_EQ(holders.back().get(), static_cast<const void*>(holder1.get()));

  EXPECT_TRUE(job1->get_input_memory_holders().empty());
  EXPECT_TRUE(job1->get_input_tensors().empty());

  ASSERT_EQ(master->get_output_tensors().size(), 1U);
  EXPECT_EQ(master->get_output_tensors()[0].size(0), 2);

  const std::vector<torch::Tensor> aggregated_outputs = {
      torch::zeros({2, 2}, torch::TensorOptions().dtype(torch::kFloat))};
  master->get_on_complete()(aggregated_outputs, 3.4);
  EXPECT_TRUE(job1_called);
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
  job0->timing_info().last_enqueued_time = job0->timing_info().enqueued_time;
  job1->timing_info().enqueued_time = base + std::chrono::milliseconds(2);
  job1->timing_info().last_enqueued_time = job1->timing_info().enqueued_time;
  job2->timing_info().enqueued_time = base + std::chrono::milliseconds(4);
  job2->timing_info().last_enqueued_time = job2->timing_info().enqueued_time;

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

  master->set_executed_on(starpu_server::DeviceType::CUDA);
  master->set_device_id(11);
  master->set_worker_id(13);
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

TEST_F(
    StarPUTaskRunnerFixture,
    MaybeBuildBatchedJobPreservesEffectiveBatchSizeAfterMergingInputs)
{
  auto make_input = [](float a, float b) {
    return torch::tensor({{a, b}}, torch::TensorOptions().dtype(torch::kFloat));
  };

  auto job0 = make_job(0, {make_input(1.0F, 2.0F)});
  auto job1 = make_job(1, {make_input(3.0F, 4.0F)});
  auto job2 = make_job(2, {make_input(5.0F, 6.0F)});

  job0->set_output_tensors(
      {torch::zeros({1, 2}, torch::TensorOptions().dtype(torch::kFloat))});
  job1->set_output_tensors(
      {torch::zeros({1, 2}, torch::TensorOptions().dtype(torch::kFloat))});
  job2->set_output_tensors(
      {torch::zeros({1, 2}, torch::TensorOptions().dtype(torch::kFloat))});

  std::vector<std::shared_ptr<starpu_server::InferenceJob>> jobs{
      job0, job1, job2};

  auto master =
      starpu_server::StarPUTaskRunnerTestAdapter::maybe_build_batched_job(
          runner_.get(), jobs);

  ASSERT_EQ(master, job0);
  ASSERT_TRUE(master->effective_batch_size().has_value());
  EXPECT_EQ(*master->effective_batch_size(), 3);
}

TEST_F(
    StarPUTaskRunnerFixture,
    MaybeBuildBatchedJobStoresPendingSubJobsWhenInputPoolAvailable)
{
  auto model_config = make_model_config(
      "pending_pool_model", {make_tensor_config("input0", {1, 2}, at::kFloat)},
      {make_tensor_config("output0", {1, 2}, at::kFloat)});
  reset_runner_with_model(model_config, /*pool_size=*/2);

  auto make_input = [](float a, float b) {
    return torch::tensor({{a, b}}, torch::TensorOptions().dtype(torch::kFloat));
  };

  auto job0 = make_job(0, {make_input(1.0F, 2.0F)});
  auto job1 = make_job(1, {make_input(3.0F, 4.0F)});
  auto job2 = make_job(2, {make_input(5.0F, 6.0F)});

  std::vector<std::shared_ptr<starpu_server::InferenceJob>> jobs{
      job0, job1, job2};

  auto master =
      starpu_server::StarPUTaskRunnerTestAdapter::maybe_build_batched_job(
          runner_.get(), jobs);

  ASSERT_EQ(master, job0);
  const auto& pending = master->pending_sub_jobs();
  ASSERT_EQ(pending.size(), 2U);
  EXPECT_EQ(pending[0], job1);
  EXPECT_EQ(pending[1], job2);

  EXPECT_FALSE(job1->get_input_tensors().empty());
  EXPECT_FALSE(job2->get_input_tensors().empty());
}

TEST_F(
    StarPUTaskRunnerFixture, MaybeBuildBatchedJobLogsTraceWhenVerbosityEnabled)
{
  opts_.verbosity = starpu_server::VerbosityLevel::Trace;
  auto model_config = make_model_config(
      "trace_batch", {make_tensor_config("input0", {1, 2}, at::kFloat)},
      {make_tensor_config("output0", {1, 2}, at::kFloat)});
  reset_runner_with_model(model_config, /*pool_size=*/1);

  auto make_input = [](float a, float b) {
    return torch::tensor({{a, b}}, torch::TensorOptions().dtype(torch::kFloat));
  };

  auto job0 = make_job(10, {make_input(1.0F, 2.0F)});
  auto job1 = make_job(11, {make_input(3.0F, 4.0F)});
  job0->set_output_tensors(
      {torch::zeros({1, 2}, torch::TensorOptions().dtype(torch::kFloat))});
  job1->set_output_tensors(
      {torch::zeros({1, 2}, torch::TensorOptions().dtype(torch::kFloat))});

  std::vector<std::shared_ptr<starpu_server::InferenceJob>> jobs{job0, job1};

  CaptureStream capture{std::cout};
  auto master =
      starpu_server::StarPUTaskRunnerTestAdapter::maybe_build_batched_job(
          runner_.get(), jobs);
  static_cast<void>(master);

  const auto logs = capture.str();
  EXPECT_NE(logs.find("Formed batch for job ID"), std::string::npos) << logs;
  EXPECT_NE(logs.find("requests (2 samples)"), std::string::npos) << logs;
}

TEST_F(StarPUTaskRunnerFixture, SampleLimitPerBatchRespectsInputPoolCapacity)
{
  auto model_config = make_model_config(
      "pooled_model", {make_tensor_config("input0", {1, 1}, at::kFloat)},
      {make_tensor_config("output0", {1, 1}, at::kFloat)});
  constexpr int kPoolSize = 1;
  reset_runner_with_model(model_config, /*pool_size=*/kPoolSize);

  opts_.batching.dynamic_batching = true;
  opts_.batching.max_batch_size = 4;
  opts_.batching.batch_coalesce_timeout_ms = 0;

  auto first = make_job(
      0, {torch::ones({1, 1}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  auto second = make_job(
      1, {torch::ones({1, 1}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});

  ASSERT_TRUE(queue_.push(second));

  auto collected = starpu_server::StarPUTaskRunnerTestAdapter::collect_batch(
      runner_.get(), first);

  ASSERT_EQ(collected.size(), static_cast<std::size_t>(kPoolSize));
  EXPECT_EQ(collected[0], first);
  EXPECT_EQ(queue_.size(), 0U);

  auto pending = runner_->wait_for_next_job();
  ASSERT_EQ(pending, second);
}

TEST_F(
    StarPUTaskRunnerFixture, ExceedsSampleLimitReturnsFalseWhenCapNonPositive)
{
  auto job = make_job(
      42, {torch::ones({2, 3}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  ASSERT_NE(runner_, nullptr);

  EXPECT_FALSE(starpu_server::StarPUTaskRunnerTestAdapter::exceeds_sample_limit(
      runner_.get(), /*accumulated_samples=*/5, job,
      /*max_samples_cap=*/0));
  EXPECT_FALSE(starpu_server::StarPUTaskRunnerTestAdapter::exceeds_sample_limit(
      runner_.get(), /*accumulated_samples=*/5, job,
      /*max_samples_cap=*/-7));
}

TEST_F(StarPUTaskRunnerFixture, CollectBatchLimitsUsingEffectiveBatchSize)
{
  auto model_config = make_model_config("eff_model", {}, {});
  reset_runner_with_model(model_config, /*pool_size=*/0);

  opts_.batching.dynamic_batching = true;
  opts_.batching.max_batch_size = 5;
  opts_.batching.batch_coalesce_timeout_ms = 0;

  auto first = make_job(
      10, {torch::ones({1}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  first->set_effective_batch_size(5);
  auto second = make_job(
      11, {torch::ones({1}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  ASSERT_TRUE(queue_.push(second));

  auto collected = starpu_server::StarPUTaskRunnerTestAdapter::collect_batch(
      runner_.get(), first);
  ASSERT_EQ(collected.size(), 1U);
  EXPECT_EQ(collected.front(), first);
  auto pending = runner_->wait_for_next_job();
  ASSERT_EQ(pending, second);
}

TEST_F(StarPUTaskRunnerFixture, CollectBatchDefaultsSampleSizeToOneWhenNoInputs)
{
  auto model_config = make_model_config("no_input_model", {}, {});
  reset_runner_with_model(model_config, /*pool_size=*/0);

  opts_.batching.dynamic_batching = true;
  opts_.batching.max_batch_size = 1;
  opts_.batching.batch_coalesce_timeout_ms = 0;

  auto first = make_job(20, {});
  auto second = make_job(21, {});
  ASSERT_TRUE(queue_.push(second));

  auto collected = starpu_server::StarPUTaskRunnerTestAdapter::collect_batch(
      runner_.get(), first);
  ASSERT_EQ(collected.size(), 1U);
  EXPECT_EQ(collected.front(), first);
  auto pending = runner_->wait_for_next_job();
  ASSERT_EQ(pending, second);
}

TEST_F(
    StarPUTaskRunnerFixture,
    CollectBatchBatchesMultipleJobsWhenInputsAreMissing)
{
  auto model_config = make_model_config("no_input_model", {}, {});
  reset_runner_with_model(model_config, /*pool_size=*/0);

  opts_.batching.dynamic_batching = true;
  opts_.batching.max_batch_size = 2;
  opts_.batching.batch_coalesce_timeout_ms = 0;

  auto first = make_job(22, {});
  auto second = make_job(23, {});
  ASSERT_TRUE(queue_.push(second));

  auto collected = starpu_server::StarPUTaskRunnerTestAdapter::collect_batch(
      runner_.get(), first);

  ASSERT_EQ(collected.size(), 2U);
  EXPECT_EQ(collected[0], first);
  EXPECT_EQ(collected[1], second);
  EXPECT_EQ(queue_.size(), 0U);
}

TEST_F(
    StarPUTaskRunnerFixture, CollectBatchReturnsAggregatedJobWithoutCoalescing)
{
  opts_.batching.dynamic_batching = true;
  opts_.batching.max_batch_size = 4;
  opts_.batching.batch_coalesce_timeout_ms = 0;

  auto aggregated = make_job(
      24, {torch::ones({1, 2}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  auto sub_job = make_job(
      25, {torch::ones({1, 2}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  std::vector<starpu_server::InferenceJob::AggregatedSubJob> sub_jobs;
  sub_jobs.push_back(
      make_aggregated_sub_job(sub_job, sub_job->get_request_id()));
  aggregated->set_aggregated_sub_jobs(std::move(sub_jobs));

  auto queued = make_job(
      26, {torch::ones({1, 2}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  ASSERT_TRUE(queue_.push(queued));

  auto collected = starpu_server::StarPUTaskRunnerTestAdapter::collect_batch(
      runner_.get(), aggregated);

  ASSERT_EQ(collected.size(), 1U);
  EXPECT_EQ(collected.front(), aggregated);
  EXPECT_EQ(queue_.size(), 1U);
}

TEST_F(
    StarPUTaskRunnerFixture,
    CollectBatchReturnsFirstWhenLogicalJobCountExceedsOne)
{
  opts_.batching.dynamic_batching = true;
  opts_.batching.max_batch_size = 4;
  opts_.batching.batch_coalesce_timeout_ms = 0;

  auto first = make_job(
      40, {torch::ones({1, 2}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  first->set_logical_job_count(2);

  auto queued = make_job(
      41, {torch::ones({1, 2}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  ASSERT_TRUE(queue_.push(queued));

  auto collected = starpu_server::StarPUTaskRunnerTestAdapter::collect_batch(
      runner_.get(), first);

  ASSERT_EQ(collected.size(), 1U);
  EXPECT_EQ(collected.front(), first);
  EXPECT_EQ(queue_.size(), 1U);
}

TEST_F(
    StarPUTaskRunnerFixture,
    WaitForNextJobDeliversAggregatedJobWithoutCoalescing)
{
  opts_.batching.dynamic_batching = true;
  opts_.batching.max_batch_size = 4;
  opts_.batching.batch_coalesce_timeout_ms = 10;

  auto aggregated = make_job(
      27, {torch::ones({1, 2}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  auto sub_job = make_job(
      28, {torch::ones({1, 2}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});

  std::vector<starpu_server::InferenceJob::AggregatedSubJob> sub_jobs;
  sub_jobs.push_back(
      make_aggregated_sub_job(sub_job, sub_job->get_request_id()));
  aggregated->set_aggregated_sub_jobs(std::move(sub_jobs));
  ASSERT_TRUE(aggregated->has_aggregated_sub_jobs());

  auto follower = make_job(
      29, {torch::ones({1, 2}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  ASSERT_TRUE(queue_.push(aggregated));
  ASSERT_TRUE(queue_.push(follower));

  auto first = runner_->wait_for_next_job();
  ASSERT_EQ(first, aggregated);
  ASSERT_TRUE(first->has_aggregated_sub_jobs());

  auto collected = starpu_server::StarPUTaskRunnerTestAdapter::collect_batch(
      runner_.get(), first);

  ASSERT_EQ(collected.size(), 1U);
  EXPECT_EQ(collected.front(), aggregated);
  EXPECT_EQ(queue_.size(), 1U);
  auto remaining = runner_->wait_for_next_job();
  ASSERT_EQ(remaining, follower);
}

TEST_F(StarPUTaskRunnerFixture, JobSampleSizeTreatsNullJobAsZeroSamples)
{
  auto model_config = make_model_config("null_job_model", {}, {});
  reset_runner_with_model(model_config, /*pool_size=*/0);

  const int64_t samples =
      starpu_server::StarPUTaskRunnerTestAdapter::job_sample_size(
          runner_.get(), nullptr);

  EXPECT_EQ(samples, 0);
}

TEST_F(StarPUTaskRunnerFixture, CollectBatchInfersSampleCountFromInputRank)
{
  auto model_config = make_model_config(
      "rank_model", {make_tensor_config("input0", {1, 2}, at::kFloat)},
      {make_tensor_config("output0", {1, 2}, at::kFloat)});
  reset_runner_with_model(model_config, /*pool_size=*/4);

  opts_.batching.dynamic_batching = true;
  opts_.batching.max_batch_size = 3;
  opts_.batching.batch_coalesce_timeout_ms = 0;

  auto first = make_job(
      30, {torch::ones({3, 1, 2}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  auto second = make_job(
      31, {torch::ones({1, 1, 2}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  ASSERT_TRUE(queue_.push(second));

  auto collected = starpu_server::StarPUTaskRunnerTestAdapter::collect_batch(
      runner_.get(), first);
  ASSERT_EQ(collected.size(), 1U);
  EXPECT_EQ(collected.front(), first);
  auto pending = runner_->wait_for_next_job();
  ASSERT_EQ(pending, second);
}

TEST_F(StarPUTaskRunnerFixture, CollectBatchIgnoresNullFirstJob)
{
  auto collected = starpu_server::StarPUTaskRunnerTestAdapter::collect_batch(
      runner_.get(), nullptr);
  EXPECT_TRUE(collected.empty());
}

TEST(
    StarPUTaskRunnerTestAdapter,
    PropagateCompletionToSubJobsNoopsWhenAggregatedJobNull)
{
  const auto outputs = std::vector<torch::Tensor>{
      torch::tensor({1.0F}, torch::TensorOptions().dtype(torch::kFloat))};
  EXPECT_NO_THROW(starpu_server::StarPUTaskRunnerTestAdapter::
                      propagate_completion_to_sub_jobs(nullptr, outputs, 5.0));
}

TEST(
    StarPUTaskRunnerTestAdapter, PropagateCompletionToSubJobsNoopsWhenNoSubJobs)
{
  auto aggregated = std::make_shared<starpu_server::InferenceJob>();
  aggregated->set_aggregated_sub_jobs({});

  const auto outputs = std::vector<torch::Tensor>{
      torch::tensor({2.0F}, torch::TensorOptions().dtype(torch::kFloat))};

  starpu_server::StarPUTaskRunnerTestAdapter::propagate_completion_to_sub_jobs(
      aggregated, outputs, 3.5);

  EXPECT_TRUE(aggregated->aggregated_sub_jobs().empty());
}

TEST(
    StarPUTaskRunnerTestAdapter,
    PropagateCompletionToSubJobsSkipsExpiredSubJobPointers)
{
  auto aggregated = std::make_shared<starpu_server::InferenceJob>();
  bool callback_invoked = false;
  std::vector<starpu_server::InferenceJob::AggregatedSubJob> sub_jobs;
  {
    auto expired = std::make_shared<starpu_server::InferenceJob>();
    starpu_server::InferenceJob::AggregatedSubJob entry{};
    entry.job = expired;
    entry.callback = [&](const std::vector<torch::Tensor>&, double) {
      callback_invoked = true;
    };
    entry.batch_size = 2;
    sub_jobs.push_back(entry);
  }  // expired shared_ptr goes out of scope and invalidates weak_ptr
  aggregated->set_aggregated_sub_jobs(std::move(sub_jobs));

  const auto outputs = std::vector<torch::Tensor>{
      torch::tensor({3.0F, 4.0F}, torch::TensorOptions().dtype(torch::kFloat))};

  starpu_server::StarPUTaskRunnerTestAdapter::propagate_completion_to_sub_jobs(
      aggregated, outputs, 4.5);

  EXPECT_FALSE(callback_invoked);
}

TEST(ResultDispatcher, CleanupTerminalJobPayloadNoopsWhenJobIsNull)
{
  const std::shared_ptr<starpu_server::InferenceJob> job;
  EXPECT_NO_THROW(
      starpu_server::ResultDispatcher::cleanup_terminal_job_payload(job));
}

TEST(ResultDispatcher, ClearBatchingStateNoopsWhenJobIsNull)
{
  const std::shared_ptr<starpu_server::InferenceJob> job;
  EXPECT_NO_THROW(starpu_server::ResultDispatcher::clear_batching_state(job));
}

TEST(ResultDispatcher, ClearPendingSubJobCallbacksNoopsWhenJobIsNull)
{
  const std::shared_ptr<starpu_server::InferenceJob> job;
  EXPECT_NO_THROW(
      starpu_server::ResultDispatcher::clear_pending_sub_job_callbacks(job));
}

TEST(ResultDispatcher, FinalizeJobCompletionNoopsWhenJobIsNull)
{
  std::atomic<std::size_t> completed_jobs{0};
  std::condition_variable all_done_cv;
  starpu_server::ResultDispatcher dispatcher{
      nullptr, &completed_jobs, &all_done_cv};

  const std::shared_ptr<starpu_server::InferenceJob> job;
  EXPECT_NO_THROW(dispatcher.finalize_job_completion(job));
  EXPECT_EQ(completed_jobs.load(std::memory_order_acquire), 0U);
}

TEST(ResultDispatcher, FinalizeJobCompletionNoopsWhenCompletedJobsIsNull)
{
  std::condition_variable all_done_cv;
  starpu_server::ResultDispatcher dispatcher{nullptr, nullptr, &all_done_cv};

  auto job = std::make_shared<starpu_server::InferenceJob>();
  job->set_logical_job_count(3);
  EXPECT_NO_THROW(dispatcher.finalize_job_completion(job));
}

TEST(ResultDispatcher, FinalizeJobCompletionNoopsWhenAllDoneCvIsNull)
{
  std::atomic<std::size_t> completed_jobs{0};
  starpu_server::ResultDispatcher dispatcher{nullptr, &completed_jobs, nullptr};

  auto job = std::make_shared<starpu_server::InferenceJob>();
  job->set_logical_job_count(3);
  EXPECT_NO_THROW(dispatcher.finalize_job_completion(job));
  EXPECT_EQ(completed_jobs.load(std::memory_order_acquire), 0U);
}

TEST_F(
    StarPUTaskRunnerFixture,
    ResultDispatcherReleaseInflightSlotReturnsWhenCountIsZero)
{
  opts_.batching.max_inflight_tasks = 10;
  runner_.reset();
  starpu_setup_.reset();
  starpu_setup_ = std::make_unique<starpu_server::StarPUSetup>(opts_);
  config_.starpu = starpu_setup_.get();
  config_.opts = &opts_;
  runner_ = std::make_unique<starpu_server::StarPUTaskRunner>(config_);

  ASSERT_TRUE(starpu_server::StarPUTaskRunnerTestAdapter::has_inflight_limit(
      runner_.get()));
  ASSERT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_max_inflight_tasks(
          runner_.get()),
      10U);
  ASSERT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      0U);

  starpu_server::StarPUTaskRunnerTestAdapter::
      release_inflight_slot_via_result_dispatcher(runner_.get());

  EXPECT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      0U);
}

TEST(ResultDispatcher, HandleJobCompletionNoopsWhenJobIsNull)
{
  std::atomic<std::size_t> completed_jobs{0};
  std::condition_variable all_done_cv;
  starpu_server::ResultDispatcher dispatcher{
      nullptr, &completed_jobs, &all_done_cv};

  bool callback_invoked = false;
  const starpu_server::InferenceJob::CompletionCallback callback =
      [&callback_invoked](const std::vector<torch::Tensor>&, double) {
        callback_invoked = true;
      };
  auto results = std::vector<torch::Tensor>{
      torch::tensor({1.0F}, torch::TensorOptions().dtype(torch::kFloat))};

  const std::shared_ptr<starpu_server::InferenceJob> job;
  EXPECT_NO_THROW(
      dispatcher.handle_job_completion(job, callback, results, 1.5));
  EXPECT_FALSE(callback_invoked);
  EXPECT_EQ(completed_jobs.load(std::memory_order_acquire), 0U);
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
  aggregated->set_executed_on(starpu_server::DeviceType::CUDA);
  aggregated->set_device_id(5);
  aggregated->set_worker_id(7);

  const auto base = internal::Clock::now();
  aggregated->set_start_time(base + std::chrono::milliseconds(10));
  aggregated->timing_info().enqueued_time = base + std::chrono::milliseconds(2);
  aggregated->timing_info().last_enqueued_time =
      aggregated->timing_info().enqueued_time;
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
  sub_jobs.back().request_id = job_one->get_request_id();
  sub_jobs.emplace_back(job_two, job_two->get_on_complete(), 2);
  sub_jobs.back().request_id = job_two->get_request_id();
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

TEST(
    StarPUTaskRunnerTestAdapter,
    PropagateCompletionToSubJobsCleansEachSubJobWhenCallbackThrows)
{
  auto aggregated = std::make_shared<starpu_server::InferenceJob>();
  auto job_one = std::make_shared<starpu_server::InferenceJob>();
  auto job_two = std::make_shared<starpu_server::InferenceJob>();
  auto pending_job = std::make_shared<starpu_server::InferenceJob>();

  aggregated->set_request_id(100);
  job_one->set_request_id(101);
  job_two->set_request_id(102);

  auto make_holder = [](int value) {
    auto holder = std::make_shared<int>(value);
    return std::shared_ptr<const void>(holder, holder.get());
  };

  aggregated->set_input_tensors({torch::tensor({9})});
  aggregated->set_input_memory_holders({make_holder(9)});
  aggregated->set_output_tensors(
      {torch::tensor({9.0F}, torch::TensorOptions().dtype(torch::kFloat))});

  job_one->set_input_tensors({torch::tensor({1})});
  job_one->set_input_memory_holders({make_holder(1)});
  job_one->set_output_tensors(
      {torch::tensor({1.0F}, torch::TensorOptions().dtype(torch::kFloat))});

  job_two->set_input_tensors({torch::tensor({2})});
  job_two->set_input_memory_holders({make_holder(2)});
  job_two->set_output_tensors(
      {torch::tensor({2.0F}, torch::TensorOptions().dtype(torch::kFloat))});

  bool pending_callback_invoked = false;
  pending_job->set_on_complete(
      [&pending_callback_invoked](const std::vector<torch::Tensor>&, double) {
        pending_callback_invoked = true;
      });
  aggregated->set_pending_sub_jobs({pending_job});

  bool job_one_called = false;
  double job_one_latency = 0.0;
  std::vector<torch::Tensor> job_one_outputs;
  job_one->set_on_complete(
      [&](const std::vector<torch::Tensor>& outputs, double latency_ms) {
        job_one_called = true;
        job_one_latency = latency_ms;
        job_one_outputs = outputs;
        throw std::runtime_error("sub-job callback failure");
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
  sub_jobs.back().request_id = job_one->get_request_id();
  sub_jobs.emplace_back(job_two, job_two->get_on_complete(), 1);
  sub_jobs.back().request_id = job_two->get_request_id();
  aggregated->set_aggregated_sub_jobs(std::move(sub_jobs));

  const auto primary = torch::tensor(
      {{1.0F, 2.0F}, {3.0F, 4.0F}},
      torch::TensorOptions().dtype(torch::kFloat));
  const auto secondary = torch::tensor(
      {{10.0F}, {20.0F}}, torch::TensorOptions().dtype(torch::kFloat));
  const double latency_ms = 8.5;

  EXPECT_NO_THROW(starpu_server::StarPUTaskRunnerTestAdapter::
                      propagate_completion_to_sub_jobs(
                          aggregated, {primary, secondary}, latency_ms));

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
      job_two_outputs[0], primary.narrow(/*dim=*/0, /*start=*/1, 1)));
  EXPECT_TRUE(torch::equal(
      job_two_outputs[1], secondary.narrow(/*dim=*/0, /*start=*/1, 1)));

  EXPECT_TRUE(job_one->get_input_tensors().empty());
  EXPECT_TRUE(job_one->get_input_memory_holders().empty());
  EXPECT_TRUE(job_one->get_output_tensors().empty());
  EXPECT_TRUE(job_two->get_input_tensors().empty());
  EXPECT_TRUE(job_two->get_input_memory_holders().empty());
  EXPECT_TRUE(job_two->get_output_tensors().empty());
  EXPECT_TRUE(aggregated->get_input_tensors().empty());
  EXPECT_TRUE(aggregated->get_input_memory_holders().empty());
  EXPECT_TRUE(aggregated->get_output_tensors().empty());

  EXPECT_FALSE(pending_callback_invoked);
  EXPECT_FALSE(pending_job->has_on_complete());
  EXPECT_TRUE(aggregated->pending_sub_jobs().empty());
  EXPECT_TRUE(aggregated->aggregated_sub_jobs().empty());
}

TEST(
    StarPUTaskRunnerTestAdapter, PropagateCompletionReleasesAggregatedJobInputs)
{
  auto aggregated = std::make_shared<starpu_server::InferenceJob>();
  auto sub_job = std::make_shared<starpu_server::InferenceJob>();

  std::vector<starpu_server::InferenceJob::AggregatedSubJob> sub_jobs;
  sub_jobs.push_back(
      {sub_job,
       std::function<void(const std::vector<torch::Tensor>&, double)>{}, 1});
  aggregated->set_aggregated_sub_jobs(std::move(sub_jobs));

  aggregated->set_input_tensors({torch::tensor({1})});
  auto holder = std::make_shared<int>(12);
  aggregated->set_input_memory_holders(
      {std::shared_ptr<const void>(holder, holder.get())});

  ASSERT_FALSE(aggregated->get_input_tensors().empty());
  ASSERT_FALSE(aggregated->get_input_memory_holders().empty());

  const auto outputs = std::vector<torch::Tensor>{
      torch::tensor({2.0F}, torch::TensorOptions().dtype(torch::kFloat))};
  starpu_server::StarPUTaskRunnerTestAdapter::propagate_completion_to_sub_jobs(
      aggregated, outputs, 5.0);

  EXPECT_TRUE(aggregated->get_input_tensors().empty());
  EXPECT_TRUE(aggregated->get_input_memory_holders().empty());
}

TEST(
    StarPUTaskRunnerTestAdapter,
    PropagateCompletionClearsPendingSubJobsWithOnComplete)
{
  auto aggregated = std::make_shared<starpu_server::InferenceJob>();

  auto aggregated_sub_job = std::make_shared<starpu_server::InferenceJob>();
  std::vector<starpu_server::InferenceJob::AggregatedSubJob> sub_jobs;
  sub_jobs.push_back(
      {aggregated_sub_job,
       std::function<void(const std::vector<torch::Tensor>&, double)>{}, 1});
  aggregated->set_aggregated_sub_jobs(std::move(sub_jobs));

  auto pending_job1 = std::make_shared<starpu_server::InferenceJob>();
  bool callback1_invoked = false;
  pending_job1->set_on_complete(
      [&callback1_invoked](const std::vector<torch::Tensor>&, double) {
        callback1_invoked = true;
      });

  auto pending_job2 = std::make_shared<starpu_server::InferenceJob>();
  bool callback2_invoked = false;
  pending_job2->set_on_complete(
      [&callback2_invoked](const std::vector<torch::Tensor>&, double) {
        callback2_invoked = true;
      });

  std::vector<std::shared_ptr<starpu_server::InferenceJob>> pending_jobs;
  pending_jobs.push_back(pending_job1);
  pending_jobs.push_back(pending_job2);
  aggregated->set_pending_sub_jobs(std::move(pending_jobs));

  ASSERT_TRUE(pending_job1->has_on_complete());
  ASSERT_TRUE(pending_job2->has_on_complete());

  const auto outputs = std::vector<torch::Tensor>{
      torch::tensor({1.0F}, torch::TensorOptions().dtype(torch::kFloat))};
  starpu_server::StarPUTaskRunnerTestAdapter::propagate_completion_to_sub_jobs(
      aggregated, outputs, 5.0);

  EXPECT_FALSE(pending_job1->has_on_complete());
  EXPECT_FALSE(pending_job2->has_on_complete());
  EXPECT_FALSE(callback1_invoked);
  EXPECT_FALSE(callback2_invoked);
  EXPECT_TRUE(aggregated->pending_sub_jobs().empty());
}

TEST(StarPUTaskRunnerTestAdapter, PropagateCompletionSkipsNullPendingSubJobs)
{
  auto aggregated = std::make_shared<starpu_server::InferenceJob>();

  auto aggregated_sub_job = std::make_shared<starpu_server::InferenceJob>();
  std::vector<starpu_server::InferenceJob::AggregatedSubJob> sub_jobs;
  sub_jobs.push_back(
      {aggregated_sub_job,
       std::function<void(const std::vector<torch::Tensor>&, double)>{}, 1});
  aggregated->set_aggregated_sub_jobs(std::move(sub_jobs));

  auto valid_pending_job = std::make_shared<starpu_server::InferenceJob>();
  bool callback_invoked = false;
  valid_pending_job->set_on_complete(
      [&callback_invoked](const std::vector<torch::Tensor>&, double) {
        callback_invoked = true;
      });

  std::vector<std::shared_ptr<starpu_server::InferenceJob>> pending_jobs;
  pending_jobs.push_back(nullptr);
  pending_jobs.push_back(valid_pending_job);
  pending_jobs.push_back(nullptr);
  aggregated->set_pending_sub_jobs(std::move(pending_jobs));

  ASSERT_TRUE(valid_pending_job->has_on_complete());

  const auto outputs = std::vector<torch::Tensor>{};
  starpu_server::StarPUTaskRunnerTestAdapter::propagate_completion_to_sub_jobs(
      aggregated, outputs, 5.0);

  EXPECT_FALSE(valid_pending_job->has_on_complete());
  EXPECT_FALSE(callback_invoked);
  EXPECT_TRUE(aggregated->pending_sub_jobs().empty());
}

TEST(
    StarPUTaskRunnerTestAdapter,
    PropagateCompletionSkipsPendingSubJobsWithoutOnComplete)
{
  auto aggregated = std::make_shared<starpu_server::InferenceJob>();

  auto aggregated_sub_job = std::make_shared<starpu_server::InferenceJob>();
  std::vector<starpu_server::InferenceJob::AggregatedSubJob> sub_jobs;
  sub_jobs.push_back(
      {aggregated_sub_job,
       std::function<void(const std::vector<torch::Tensor>&, double)>{}, 1});
  aggregated->set_aggregated_sub_jobs(std::move(sub_jobs));

  auto job_without_callback = std::make_shared<starpu_server::InferenceJob>();
  auto job_with_callback = std::make_shared<starpu_server::InferenceJob>();
  bool callback_invoked = false;
  job_with_callback->set_on_complete(
      [&callback_invoked](const std::vector<torch::Tensor>&, double) {
        callback_invoked = true;
      });

  std::vector<std::shared_ptr<starpu_server::InferenceJob>> pending_jobs;
  pending_jobs.push_back(job_without_callback);
  pending_jobs.push_back(job_with_callback);
  aggregated->set_pending_sub_jobs(std::move(pending_jobs));

  ASSERT_FALSE(job_without_callback->has_on_complete());
  ASSERT_TRUE(job_with_callback->has_on_complete());

  const auto outputs = std::vector<torch::Tensor>{};
  starpu_server::StarPUTaskRunnerTestAdapter::propagate_completion_to_sub_jobs(
      aggregated, outputs, 5.0);

  EXPECT_FALSE(job_without_callback->has_on_complete());
  EXPECT_FALSE(job_with_callback->has_on_complete());
  EXPECT_FALSE(callback_invoked);
  EXPECT_TRUE(aggregated->pending_sub_jobs().empty());
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
  job_one->timing_info().last_enqueued_time =
      job_one->timing_info().enqueued_time;
  job_two->timing_info().enqueued_time = base + std::chrono::milliseconds(4);
  job_two->timing_info().last_enqueued_time =
      job_two->timing_info().enqueued_time;
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
  job_one->timing_info().last_enqueued_time =
      job_one->timing_info().enqueued_time;
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

TEST(TaskRunnerInternal, HandleJobExceptionWithoutJobReturnsFalse)
{
  CaptureStream capture{std::cerr};
  const auto handled = starpu_server::StarPUTaskRunner::handle_job_exception(
      std::shared_ptr<starpu_server::InferenceJob>{},
      std::runtime_error("outer failure"));

  EXPECT_FALSE(handled);

  const auto logs = capture.str();
  const auto expected =
      expected_log_line(ErrorLevel, "[Exception] Job -1: outer failure");
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

TEST(TaskRunnerInternal, BuildRequestArrivalUsForTraceReturnsEmptyWhenJobIsNull)
{
  namespace internal = starpu_server::task_runner_internal;

  const auto result = internal::build_request_arrival_us_for_trace(nullptr);

  EXPECT_TRUE(result.empty());
}

TEST(
    TaskRunnerInternal,
    BuildRequestArrivalUsForTraceReturnsSingleEntryForNonAggregatedJob)
{
  namespace internal = starpu_server::task_runner_internal;
  using Clock = starpu_server::MonotonicClock;

  auto job = std::make_shared<starpu_server::InferenceJob>();
  const auto arrival_time =
      Clock::time_point{} + std::chrono::microseconds(123456);
  job->timing_info().enqueued_time = arrival_time;

  const auto result = internal::build_request_arrival_us_for_trace(job);

  ASSERT_EQ(result.size(), 1U);
  EXPECT_EQ(result[0], 123456);
}

TEST(
    TaskRunnerInternal,
    BuildRequestArrivalUsForTraceReturnsZeroForDefaultTimePoint)
{
  namespace internal = starpu_server::task_runner_internal;
  using Clock = starpu_server::MonotonicClock;

  auto job = std::make_shared<starpu_server::InferenceJob>();
  job->timing_info().enqueued_time = Clock::time_point{};

  const auto result = internal::build_request_arrival_us_for_trace(job);

  ASSERT_EQ(result.size(), 1U);
  EXPECT_EQ(result[0], 0);
}

TEST(
    TaskRunnerInternal,
    BuildRequestArrivalUsForTraceUsesArrivalTimeFromAggregatedSubJobs)
{
  namespace internal = starpu_server::task_runner_internal;
  using Clock = starpu_server::MonotonicClock;

  auto aggregated = std::make_shared<starpu_server::InferenceJob>();
  auto sub_job1 = std::make_shared<starpu_server::InferenceJob>();
  auto sub_job2 = std::make_shared<starpu_server::InferenceJob>();

  const auto arrival1 = Clock::time_point{} + std::chrono::microseconds(100);
  const auto arrival2 = Clock::time_point{} + std::chrono::microseconds(200);

  std::vector<starpu_server::InferenceJob::AggregatedSubJob> sub_jobs;
  sub_jobs.push_back(
      {sub_job1,
       std::function<void(const std::vector<torch::Tensor>&, double)>{}, 1});
  sub_jobs.back().arrival_time = arrival1;

  sub_jobs.push_back(
      {sub_job2,
       std::function<void(const std::vector<torch::Tensor>&, double)>{}, 1});
  sub_jobs.back().arrival_time = arrival2;

  aggregated->set_aggregated_sub_jobs(std::move(sub_jobs));

  const auto result = internal::build_request_arrival_us_for_trace(aggregated);

  ASSERT_EQ(result.size(), 2U);
  EXPECT_EQ(result[0], 100);
  EXPECT_EQ(result[1], 200);
}

TEST(
    TaskRunnerInternal,
    BuildRequestArrivalUsForTraceFallsBackToJobEnqueuedTimeWhenArrivalIsDefault)
{
  namespace internal = starpu_server::task_runner_internal;
  using Clock = starpu_server::MonotonicClock;

  auto aggregated = std::make_shared<starpu_server::InferenceJob>();
  auto sub_job = std::make_shared<starpu_server::InferenceJob>();

  const auto enqueued_time =
      Clock::time_point{} + std::chrono::microseconds(300);
  sub_job->timing_info().enqueued_time = enqueued_time;

  std::vector<starpu_server::InferenceJob::AggregatedSubJob> sub_jobs;
  sub_jobs.push_back(
      {sub_job,
       std::function<void(const std::vector<torch::Tensor>&, double)>{}, 1});
  sub_jobs.back().arrival_time = Clock::time_point{};

  aggregated->set_aggregated_sub_jobs(std::move(sub_jobs));

  const auto result = internal::build_request_arrival_us_for_trace(aggregated);

  ASSERT_EQ(result.size(), 1U);
  EXPECT_EQ(result[0], 300);
}
