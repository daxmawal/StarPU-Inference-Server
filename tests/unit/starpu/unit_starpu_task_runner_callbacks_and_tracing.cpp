#include "monitoring/congestion_monitor.hpp"
#include "monitoring/runtime_observability.hpp"
#include "starpu_task_worker/result_dispatcher_component.hpp"
#include "unit_starpu_task_runner_support.hpp"

namespace {

struct PrepareJobCompletionCallbackHooksGuard {
  explicit PrepareJobCompletionCallbackHooksGuard(
      starpu_server::ResultDispatcher::PrepareJobCompletionCallbackTestHooks
          hooks)
  {
    starpu_server::ResultDispatcher::SetPrepareJobCompletionCallbackTestHooks(
        std::move(hooks));
  }

  ~PrepareJobCompletionCallbackHooksGuard()
  {
    starpu_server::ResultDispatcher::
        ClearPrepareJobCompletionCallbackTestHooks();
  }
};

auto
find_family(
    const std::vector<prometheus::MetricFamily>& families,
    std::string_view name) -> const prometheus::MetricFamily*
{
  for (const auto& family : families) {
    if (family.name == name) {
      return &family;
    }
  }
  return nullptr;
}

auto
labels_match(
    const prometheus::ClientMetric& metric,
    const std::vector<std::pair<std::string_view, std::string_view>>& labels)
    -> bool
{
  for (const auto& [name, value] : labels) {
    bool matched = false;
    for (const auto& label : metric.label) {
      if (label.name == name && label.value == value) {
        matched = true;
        break;
      }
    }
    if (!matched) {
      return false;
    }
  }
  return true;
}

auto
find_metric(
    const std::vector<prometheus::MetricFamily>& families,
    std::string_view family_name,
    const std::vector<std::pair<std::string_view, std::string_view>>& labels)
    -> const prometheus::ClientMetric*
{
  const auto* family = find_family(families, family_name);
  if (family == nullptr) {
    return nullptr;
  }
  for (const auto& metric : family->metric) {
    if (labels_match(metric, labels)) {
      return &metric;
    }
  }
  return nullptr;
}

auto
find_gauge_value(
    const std::vector<prometheus::MetricFamily>& families,
    std::string_view family_name,
    const std::vector<std::pair<std::string_view, std::string_view>>& labels =
        {}) -> std::optional<double>
{
  const auto* metric = find_metric(families, family_name, labels);
  if (metric == nullptr) {
    return std::nullopt;
  }
  return metric->gauge.value;
}

auto
find_counter_value(
    const std::vector<prometheus::MetricFamily>& families,
    std::string_view family_name,
    const std::vector<std::pair<std::string_view, std::string_view>>& labels =
        {}) -> std::optional<double>
{
  const auto* metric = find_metric(families, family_name, labels);
  if (metric == nullptr) {
    return std::nullopt;
  }
  return metric->counter.value;
}

}  // namespace

TEST(TaskRunnerInternal, ReleaseInputsFromAdditionalJobsSkipsNullEntries)
{
  namespace internal = starpu_server::task_runner_internal;

  auto job0 = std::make_shared<starpu_server::InferenceJob>();
  job0->set_input_tensors({torch::tensor({1})});
  auto job0_holder = std::make_shared<int>(7);
  job0->set_input_memory_holders(
      {std::shared_ptr<const void>(job0_holder, job0_holder.get())});

  auto job2 = std::make_shared<starpu_server::InferenceJob>();
  job2->set_input_tensors({torch::tensor({2})});
  auto job2_holder = std::make_shared<int>(9);
  job2->set_input_memory_holders(
      {std::shared_ptr<const void>(job2_holder, job2_holder.get())});

  std::vector<std::shared_ptr<starpu_server::InferenceJob>> jobs{
      job0, nullptr, job2};

  internal::release_inputs_from_additional_jobs(jobs);

  ASSERT_EQ(job0->get_input_tensors().size(), 1U);
  EXPECT_FALSE(job0->get_input_tensors().empty());
  EXPECT_EQ(job0->get_input_memory_holders().size(), 1U);

  EXPECT_TRUE(job2->get_input_tensors().empty());
  EXPECT_TRUE(job2->get_input_memory_holders().empty());
}

TEST(
    StarPUTaskWorkerInternals,
    ResizeVectorInterfaceReturnsEarlyWhenInterfaceMissing)
{
  constexpr test_api::VectorResizeSpecShim spec{5, 20};

  EXPECT_NO_THROW(
      test_api::resize_starpu_vector_interface(nullptr, spec, true));
}

TEST_F(
    StarPUTaskRunnerFixture, ResolveBatchSizeWithoutJobDefaultsToSingleSample)
{
  std::shared_ptr<starpu_server::InferenceJob> missing_job;
  EXPECT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::resolve_batch_size(
          runner_.get(), missing_job),
      1);
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
  job->completion().get_on_complete()(outputs, latency);
  EXPECT_TRUE(probe.called);
  const auto& completed_jobs = completed_jobs_;
  EXPECT_EQ(completed_jobs.load(), 1);
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
  job->completion().get_on_complete()(std::vector<torch::Tensor>{}, latency);

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
  job->completion().get_on_complete()(std::vector<torch::Tensor>{}, latency);

  const auto stats = starpu_server::perf_observer::snapshot();
  ASSERT_TRUE(stats.has_value());
  EXPECT_EQ(stats->total_inferences, 1U);

  starpu_server::perf_observer::reset();
}

TEST_F(
    StarPUTaskRunnerFixture,
    PrepareJobCompletionCallbackLogsTraceWhenTracerEnabled)
{
  starpu_server::perf_observer::reset();
  TraceLoggerSession session;

  auto probe = starpu_server::make_callback_probe();
  auto job = probe.job;
  job->set_request_id(501);
  job->set_submission_id(901);
  job->completion().set_model_name("trace_model");
  job->set_input_tensors({torch::tensor({1})});
  job->set_executed_on(starpu_server::DeviceType::CPU);
  job->set_device_id(0);
  job->set_worker_id(4);
  populate_trace_timing(*job);

  using clock = starpu_server::MonotonicClock;
  const auto base = clock::now();
  auto& timing = job->timing_info();
  timing.codelet_start_time = base - std::chrono::microseconds(400);
  timing.inference_start_time =
      timing.codelet_start_time + std::chrono::microseconds(50);
  timing.codelet_end_time =
      timing.codelet_start_time + std::chrono::microseconds(300);
  timing.callback_start_time =
      timing.codelet_end_time + std::chrono::microseconds(50);
  timing.callback_end_time =
      timing.callback_start_time + std::chrono::microseconds(25);

  runner_->prepare_job_completion_callback(job);

  const double latency = 2.0;
  auto outputs = std::vector<torch::Tensor>{torch::tensor({2})};
  job->completion().get_on_complete()(outputs, latency);

  EXPECT_TRUE(probe.called);

  session.close();
  const auto trace_content = read_trace_file(session.path());
  const auto expected_fragment =
      "\"batch_id\":" + std::to_string(job->submission_id());
  EXPECT_NE(trace_content.find(expected_fragment), std::string::npos)
      << trace_content;

  starpu_server::perf_observer::reset();
}

TEST_F(
    StarPUTaskRunnerFixture,
    PrepareJobCompletionCallbackClampsInvalidComputeEndBeforeTracing)
{
  starpu_server::perf_observer::reset();
  TraceLoggerSession session;

  auto probe = starpu_server::make_callback_probe();
  auto job = probe.job;
  job->set_request_id(777);
  job->set_submission_id(902);
  job->completion().set_model_name("trace_fallback_model");
  job->set_input_tensors({torch::tensor({1})});
  job->set_executed_on(starpu_server::DeviceType::CPU);
  job->set_device_id(0);
  job->set_worker_id(6);
  populate_trace_timing(*job);

  using clock = starpu_server::MonotonicClock;
  const auto base = clock::now();
  auto& timing = job->timing_info();
  timing.codelet_start_time = base - std::chrono::microseconds(500);
  timing.codelet_end_time =
      timing.codelet_start_time + std::chrono::microseconds(200);
  timing.inference_start_time =
      timing.codelet_start_time + std::chrono::microseconds(150);
  timing.callback_start_time =
      timing.codelet_start_time + std::chrono::microseconds(100);
  timing.callback_end_time =
      timing.callback_start_time + std::chrono::microseconds(10);

  runner_->prepare_job_completion_callback(job);

  job->completion().get_on_complete()(
      std::vector<torch::Tensor>{torch::tensor({3})}, 3.0);

  EXPECT_TRUE(probe.called);

  session.close();
  const auto trace_content = read_trace_file(session.path());
  const auto expected_fragment =
      "\"batch_id\":" + std::to_string(job->submission_id());
  EXPECT_NE(trace_content.find(expected_fragment), std::string::npos)
      << trace_content;

  starpu_server::perf_observer::reset();
}

TEST_F(
    StarPUTaskRunnerFixture,
    PrepareJobCompletionCallbackUsesCodeletStartWhenInferenceStartMissing)
{
  starpu_server::perf_observer::reset();
  TraceLoggerSession session;

  auto probe = starpu_server::make_callback_probe();
  auto job = probe.job;
  job->set_request_id(808);
  job->set_submission_id(903);
  job->completion().set_model_name("trace_missing_inference_start");
  job->set_input_tensors({torch::tensor({1})});
  job->set_executed_on(starpu_server::DeviceType::CPU);
  job->set_device_id(0);
  job->set_worker_id(2);
  populate_trace_timing(*job);

  using clock = starpu_server::MonotonicClock;
  const auto base = clock::now();
  auto& timing = job->timing_info();
  timing.inference_start_time = clock::time_point{};
  timing.codelet_start_time = base - std::chrono::microseconds(600);
  timing.codelet_end_time =
      timing.codelet_start_time + std::chrono::microseconds(250);
  timing.callback_start_time =
      timing.codelet_end_time + std::chrono::microseconds(75);
  timing.callback_end_time =
      timing.callback_start_time + std::chrono::microseconds(25);

  runner_->prepare_job_completion_callback(job);

  auto outputs = std::vector<torch::Tensor>{torch::tensor({4})};
  job->completion().get_on_complete()(outputs, 1.0);

  EXPECT_TRUE(probe.called);

  session.close();
  const auto trace_content = read_trace_file(session.path());
  const auto expected_event_fragment = std::format(
      "\"name\":\"{}\",\"cat\":\"batching\"", job->completion().model_name());
  EXPECT_NE(trace_content.find(expected_event_fragment), std::string::npos)
      << trace_content;

  starpu_server::perf_observer::reset();
}

TEST_F(
    StarPUTaskRunnerFixture,
    PrepareJobCompletionCallbackReleasesInflightSlotWhenLimitSet)
{
  opts_.batching.max_inflight_tasks = 5;
  reset_runner_with_model(
      make_model_config(
          "test_model", {make_tensor_config("input", {1}, c10::kFloat)},
          {make_tensor_config("output", {1}, c10::kFloat)}),
      1);

  auto job = make_job(1, {torch::tensor({1.0F})});
  bool callback_invoked = false;
  job->completion().set_on_complete(
      [&callback_invoked](const std::vector<torch::Tensor>&, double) {
        callback_invoked = true;
      });

  starpu_server::StarPUTaskRunnerTestAdapter::reserve_inflight_slot(
      runner_.get());
  ASSERT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      1U);

  runner_->prepare_job_completion_callback(job);

  job->completion().get_on_complete()(
      std::vector<torch::Tensor>{torch::tensor({2.0F})}, 5.0);

  EXPECT_TRUE(callback_invoked);
  EXPECT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      0U);
}

TEST_F(
    StarPUTaskRunnerFixture,
    PrepareJobCompletionCallbackDoesNotReleaseInflightSlotWhenNoLimit)
{
  opts_.batching.max_inflight_tasks = 0;
  reset_runner_with_model(
      make_model_config(
          "test_model", {make_tensor_config("input", {1}, c10::kFloat)},
          {make_tensor_config("output", {1}, c10::kFloat)}),
      1);

  auto job = make_job(1, {torch::tensor({1.0F})});
  bool callback_invoked = false;
  job->completion().set_on_complete(
      [&callback_invoked](const std::vector<torch::Tensor>&, double) {
        callback_invoked = true;
      });

  ASSERT_FALSE(starpu_server::StarPUTaskRunnerTestAdapter::has_inflight_limit(
      runner_.get()));

  runner_->prepare_job_completion_callback(job);

  job->completion().get_on_complete()(
      std::vector<torch::Tensor>{torch::tensor({2.0F})}, 5.0);

  EXPECT_TRUE(callback_invoked);
  EXPECT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      0U);
}

TEST_F(
    StarPUTaskRunnerFixture,
    PrepareJobCompletionCallbackUsesObservabilityMetricsWhenConfigured)
{
  TraceLoggerSession session;

  auto observability = std::make_shared<starpu_server::RuntimeObservability>();
  observability->metrics = starpu_server::create_metrics_recorder(0);
  ASSERT_NE(observability->metrics, nullptr);
  observability->congestion_monitor =
      std::make_shared<starpu_server::congestion::Monitor>(nullptr);

  opts_.batching.max_inflight_tasks = 4;
  config_.observability = observability;
  reset_runner_with_model(
      make_model_config(
          "metrics_model", {make_tensor_config("input", {1}, c10::kFloat)},
          {make_tensor_config("output", {1}, c10::kFloat)}),
      1);

  auto job = make_job(44, {torch::tensor({1.0F})});
  job->completion().set_model_name("metrics_model");
  job->set_submission_id(444);
  job->timing_info().submission_id = 444;
  job->set_executed_on(starpu_server::DeviceType::CPU);
  job->set_worker_id(3);
  job->set_device_id(0);
  populate_trace_timing(*job);

  using clock = starpu_server::MonotonicClock;
  const auto base = clock::now();
  auto& timing = job->timing_info();
  timing.enqueued_time = base - std::chrono::milliseconds(6);
  timing.last_enqueued_time = timing.enqueued_time;
  timing.batch_collect_start_time = base - std::chrono::milliseconds(5);
  timing.batch_collect_end_time = base - std::chrono::milliseconds(4);
  timing.codelet_start_time = base - std::chrono::milliseconds(3);
  timing.inference_start_time = base - std::chrono::milliseconds(2);
  timing.codelet_end_time = base - std::chrono::milliseconds(1);
  timing.callback_start_time = base;
  timing.callback_end_time = base + std::chrono::milliseconds(1);

  bool callback_invoked = false;
  job->completion().set_on_complete(
      [&callback_invoked](const std::vector<torch::Tensor>&, double) {
        callback_invoked = true;
      });

  starpu_server::StarPUTaskRunnerTestAdapter::reserve_inflight_slot(
      runner_.get());
  ASSERT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      1U);

  runner_->prepare_job_completion_callback(job);
  job->completion().get_on_complete()(
      std::vector<torch::Tensor>{torch::tensor({2.0F})}, 7.5);

  EXPECT_TRUE(callback_invoked);
  EXPECT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      0U);

  const auto families =
      observability->metrics->registry()->registry()->Collect();
  const auto batch_metric =
      find_metric(families, "inference_batch_size", /*labels=*/{});
  ASSERT_NE(batch_metric, nullptr);
  EXPECT_EQ(batch_metric->histogram.sample_count, 1);

  const auto logical_batch_metric =
      find_metric(families, "inference_logical_batch_size", /*labels=*/{});
  ASSERT_NE(logical_batch_metric, nullptr);
  EXPECT_EQ(logical_batch_metric->histogram.sample_count, 1);

  const auto completed = find_counter_value(
      families, "inference_completed_total", {{"model", "metrics_model"}});
  ASSERT_TRUE(completed.has_value());
  EXPECT_DOUBLE_EQ(*completed, 1.0);

  const auto* runtime_family =
      find_family(families, "starpu_task_runtime_ms_by_worker");
  ASSERT_NE(runtime_family, nullptr);
  EXPECT_TRUE(std::ranges::any_of(
      runtime_family->metric, [](const prometheus::ClientMetric& metric) {
        return metric.histogram.sample_count > 0;
      }));

  const auto* compute_family =
      find_family(families, "inference_compute_latency_ms_by_worker");
  ASSERT_NE(compute_family, nullptr);
  EXPECT_TRUE(std::ranges::any_of(
      compute_family->metric, [](const prometheus::ClientMetric& metric) {
        return metric.histogram.sample_count > 0;
      }));

  const auto inflight =
      find_gauge_value(families, "inference_inflight_tasks", {});
  ASSERT_TRUE(inflight.has_value());
  EXPECT_DOUBLE_EQ(*inflight, 0.0);

  const auto busy_ratio =
      find_gauge_value(families, "starpu_worker_busy_ratio", {});
  ASSERT_TRUE(busy_ratio.has_value());
  EXPECT_DOUBLE_EQ(*busy_ratio, 0.0);
}

TEST_F(
    StarPUTaskRunnerFixture,
    PrepareJobCompletionCallbackRemainsValidAfterRunnerDestruction)
{
  opts_.batching.max_inflight_tasks = 2;
  reset_runner_with_model(
      make_model_config(
          "test_model", {make_tensor_config("input", {1}, c10::kFloat)},
          {make_tensor_config("output", {1}, c10::kFloat)}),
      1);

  auto job = make_job(1, {torch::tensor({1.0F})});
  bool callback_invoked = false;
  job->completion().set_on_complete(
      [&callback_invoked](const std::vector<torch::Tensor>&, double) {
        callback_invoked = true;
      });

  starpu_server::StarPUTaskRunnerTestAdapter::reserve_inflight_slot(
      runner_.get());
  runner_->prepare_job_completion_callback(job);
  auto completion = job->completion().get_on_complete();

  runner_.reset();

  EXPECT_NO_THROW(
      completion(std::vector<torch::Tensor>{torch::tensor({2.0F})}, 5.0));
  EXPECT_TRUE(callback_invoked);
  EXPECT_EQ(completed_jobs_.load(std::memory_order_acquire), 1U);
}

TEST_F(
    StarPUTaskRunnerFixture,
    PrepareJobCompletionCallbackReturnsWhenTerminalAlreadyHandled)
{
  auto job = make_job(2, {torch::tensor({1.0F})});
  std::atomic<int> callback_count{0};
  job->completion().set_on_complete(
      [&callback_count](const std::vector<torch::Tensor>&, double) {
        callback_count.fetch_add(1, std::memory_order_acq_rel);
      });

  runner_->prepare_job_completion_callback(job);
  auto outputs = std::vector<torch::Tensor>{torch::tensor({2.0F})};

  job->completion().get_on_complete()(outputs, 5.0);
  job->completion().get_on_complete()(outputs, 5.0);

  EXPECT_EQ(callback_count.load(std::memory_order_acquire), 1);
  EXPECT_EQ(completed_jobs_.load(std::memory_order_acquire), 1U);
}

TEST_F(
    StarPUTaskRunnerFixture,
    PrepareJobCompletionCallbackCleansPayloadWhenDispatcherMissing)
{
  completed_jobs_.store(0, std::memory_order_release);
  auto job = make_job(3, {torch::tensor({1.0F})});
  bool callback_invoked = false;
  job->completion().set_on_complete(
      [&callback_invoked](const std::vector<torch::Tensor>&, double) {
        callback_invoked = true;
      });
  auto holder = std::make_shared<int>(7);
  job->set_input_memory_holders(
      {std::shared_ptr<const void>(holder, holder.get())});
  job->set_output_tensors(
      {torch::tensor({9.0F}, torch::TensorOptions().dtype(torch::kFloat))});

  CaptureStream capture{std::cerr};
  starpu_server::StarPUTaskRunnerTestAdapter::set_result_dispatcher(
      runner_.get(), nullptr);
  runner_->prepare_job_completion_callback(job);

  auto outputs = std::vector<torch::Tensor>{torch::tensor({2.0F})};
  job->completion().get_on_complete()(outputs, 4.0);

  EXPECT_FALSE(callback_invoked);
  EXPECT_TRUE(job->get_input_tensors().empty());
  EXPECT_TRUE(job->get_input_memory_holders().empty());
  EXPECT_TRUE(job->get_output_tensors().empty());
  EXPECT_EQ(completed_jobs_.load(std::memory_order_acquire), 0U);

  const auto logs = capture.str();
  const auto expected = expected_log_line(
      ErrorLevel,
      "Missing ResultDispatcher in terminal completion path; completion "
      "counter may be inconsistent");
  EXPECT_NE(logs.find(expected), std::string::npos);
}

TEST_F(
    StarPUTaskRunnerFixture,
    PrepareJobCompletionCallbackLogsAndCleansPayloadOnStdException)
{
  completed_jobs_.store(0, std::memory_order_release);
  auto job = make_job(4, {torch::tensor({1.0F})});
  bool callback_invoked = false;
  job->completion().set_on_complete(
      [&callback_invoked](const std::vector<torch::Tensor>&, double) {
        callback_invoked = true;
      });
  auto holder = std::make_shared<int>(9);
  job->set_input_memory_holders(
      {std::shared_ptr<const void>(holder, holder.get())});
  job->set_output_tensors(
      {torch::tensor({5.0F}, torch::TensorOptions().dtype(torch::kFloat))});

  starpu_server::ResultDispatcher::PrepareJobCompletionCallbackTestHooks hooks;
  hooks.before_dispatch = []() {
    throw std::runtime_error("forced std completion exception");
  };
  PrepareJobCompletionCallbackHooksGuard guard{std::move(hooks)};

  CaptureStream capture{std::cerr};
  runner_->prepare_job_completion_callback(job);
  auto outputs = std::vector<torch::Tensor>{torch::tensor({2.0F})};
  job->completion().get_on_complete()(outputs, 3.0);

  EXPECT_FALSE(callback_invoked);
  EXPECT_TRUE(job->get_input_tensors().empty());
  EXPECT_TRUE(job->get_input_memory_holders().empty());
  EXPECT_TRUE(job->get_output_tensors().empty());
  EXPECT_EQ(completed_jobs_.load(std::memory_order_acquire), 1U);

  const auto logs = capture.str();
  const auto expected = expected_log_line(
      ErrorLevel,
      "Unhandled exception in terminal completion path: forced std completion "
      "exception");
  EXPECT_NE(logs.find(expected), std::string::npos);
}

TEST_F(
    StarPUTaskRunnerFixture,
    PrepareJobCompletionCallbackLogsAndCleansPayloadOnUnknownException)
{
  completed_jobs_.store(0, std::memory_order_release);
  auto job = make_job(5, {torch::tensor({1.0F})});
  bool callback_invoked = false;
  job->completion().set_on_complete(
      [&callback_invoked](const std::vector<torch::Tensor>&, double) {
        callback_invoked = true;
      });
  auto holder = std::make_shared<int>(11);
  job->set_input_memory_holders(
      {std::shared_ptr<const void>(holder, holder.get())});
  job->set_output_tensors(
      {torch::tensor({6.0F}, torch::TensorOptions().dtype(torch::kFloat))});

  starpu_server::ResultDispatcher::PrepareJobCompletionCallbackTestHooks hooks;
  hooks.before_dispatch = []() { throw 42; };
  PrepareJobCompletionCallbackHooksGuard guard{std::move(hooks)};

  CaptureStream capture{std::cerr};
  runner_->prepare_job_completion_callback(job);
  auto outputs = std::vector<torch::Tensor>{torch::tensor({2.0F})};
  job->completion().get_on_complete()(outputs, 3.0);

  EXPECT_FALSE(callback_invoked);
  EXPECT_TRUE(job->get_input_tensors().empty());
  EXPECT_TRUE(job->get_input_memory_holders().empty());
  EXPECT_TRUE(job->get_output_tensors().empty());
  EXPECT_EQ(completed_jobs_.load(std::memory_order_acquire), 1U);

  const auto logs = capture.str();
  const auto expected = expected_log_line(
      ErrorLevel, "Unhandled non-std exception in terminal completion path");
  EXPECT_NE(logs.find(expected), std::string::npos);
}

TEST_F(StarPUTaskRunnerFixture, LogJobTimingsComputesComponents)
{
  opts_.verbosity = starpu_server::VerbosityLevel::Stats;
  starpu_server::detail::TimingInfo time;
  using clock = starpu_server::MonotonicClock;
  auto base = clock::now();
  time.enqueued_time = base;
  time.last_enqueued_time = base;
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
    EnsureCallbackTimingFillsMissingAndInvalidTimestamps)
{
  using Clock = starpu_server::task_runner_internal::Clock;
  using namespace std::chrono_literals;

  starpu_server::detail::TimingInfo missing{};
  starpu_server::StarPUTaskRunnerTestAdapter::ensure_callback_timing(missing);
  EXPECT_NE(missing.callback_start_time, Clock::time_point{});
  EXPECT_NE(missing.callback_end_time, Clock::time_point{});
  EXPECT_EQ(missing.enqueued_time, missing.callback_start_time);
  EXPECT_EQ(missing.last_enqueued_time, missing.enqueued_time);

  starpu_server::detail::TimingInfo inconsistent{};
  inconsistent.callback_start_time = Clock::now();
  inconsistent.callback_end_time = inconsistent.callback_start_time - 1ms;
  inconsistent.enqueued_time = inconsistent.callback_start_time + 2ms;
  inconsistent.last_enqueued_time = inconsistent.enqueued_time - 5ms;
  const auto original_start = inconsistent.callback_start_time;

  starpu_server::StarPUTaskRunnerTestAdapter::ensure_callback_timing(
      inconsistent);

  EXPECT_EQ(inconsistent.callback_start_time, original_start);
  EXPECT_GT(inconsistent.callback_end_time, inconsistent.callback_start_time);
  EXPECT_EQ(inconsistent.enqueued_time, inconsistent.callback_start_time);
  EXPECT_EQ(inconsistent.last_enqueued_time, inconsistent.enqueued_time);
}

TEST_F(StarPUTaskRunnerFixture, RecordJobMetricsUpdatesPerfObserver)
{
  using Clock = starpu_server::task_runner_internal::Clock;
  using namespace std::chrono_literals;

  starpu_server::perf_observer::reset();

  auto job = make_job(21, {torch::tensor({3})});
  const auto enqueue_time = Clock::now();
  const auto completion_time = enqueue_time + 2ms;
  job->timing_info().enqueued_time = enqueue_time;
  job->timing_info().callback_end_time = completion_time;
  job->set_submission_id(55);
  job->timing_info().submission_id = -1;

  constexpr std::size_t kBatchSize = 4;
  const auto latency = starpu_server::StarPUTaskRunner::DurationMs{5.0};

  starpu_server::StarPUTaskRunnerTestAdapter::record_job_metrics(
      runner_.get(), job, latency, kBatchSize);

  const auto snapshot = starpu_server::perf_observer::snapshot();
  ASSERT_TRUE(snapshot.has_value());
  const double expected_duration =
      std::chrono::duration<double>(completion_time - enqueue_time).count();
  EXPECT_EQ(snapshot->total_inferences, kBatchSize);
  EXPECT_DOUBLE_EQ(snapshot->duration_seconds, expected_duration);
  EXPECT_DOUBLE_EQ(
      snapshot->throughput,
      static_cast<double>(kBatchSize) / expected_duration);
  EXPECT_EQ(job->timing_info().submission_id, job->submission_id());

  starpu_server::perf_observer::reset();
}

TEST_F(StarPUTaskRunnerFixture, FinalizeJobCompletionCountsLogicalJobs)
{
  completed_jobs_.store(0);

  auto job = make_job(31, {});
  job->batch().set_logical_job_count(3);

  starpu_server::StarPUTaskRunnerTestAdapter::finalize_job_completion(
      runner_.get(), job);
  EXPECT_EQ(completed_jobs_.load(), 3);

  job->batch().set_logical_job_count(0);
  starpu_server::StarPUTaskRunnerTestAdapter::finalize_job_completion(
      runner_.get(), job);
  EXPECT_EQ(completed_jobs_.load(), 4);
}

TEST_F(
    StarPUTaskRunnerFixture,
    FinalizeJobAfterExceptionCompletesJobWhenCallbackMissing)
{
  completed_jobs_.store(0);
  auto job = make_job(88, {});
  const std::runtime_error error("runtime failure");

  starpu_server::StarPUTaskRunnerTestAdapter::finalize_job_after_exception(
      runner_.get(), job, error, "runtime failure", job->get_request_id());

  EXPECT_EQ(completed_jobs_.load(), 1);
}

TEST_F(StarPUTaskRunnerFixture, FinalizeJobAfterExceptionReturnsWhenJobMissing)
{
  opts_.batching.max_inflight_tasks = 5;
  reset_runner_with_model(
      make_model_config(
          "test_model", {make_tensor_config("input", {1}, c10::kFloat)},
          {make_tensor_config("output", {1}, c10::kFloat)}),
      1);
  completed_jobs_.store(0, std::memory_order_release);

  starpu_server::StarPUTaskRunnerTestAdapter::reserve_inflight_slot(
      runner_.get());
  ASSERT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      1U);

  std::shared_ptr<starpu_server::InferenceJob> missing_job;
  const std::runtime_error error("runtime failure");

  EXPECT_NO_THROW(
      starpu_server::StarPUTaskRunnerTestAdapter::finalize_job_after_exception(
          runner_.get(), missing_job, error, "runtime failure", -1));

  EXPECT_EQ(completed_jobs_.load(std::memory_order_acquire), 0U);
  EXPECT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      1U);
}

TEST_F(StarPUTaskRunnerFixture, FinalizeJobAfterExceptionUsesGenericReason)
{
  starpu_server::shutdown_metrics();
  ASSERT_TRUE(starpu_server::init_metrics(0));
  struct MetricsGuard {
    ~MetricsGuard() { starpu_server::shutdown_metrics(); }
  } guard;

  auto job = make_job(89, {});
  job->completion().set_model_name("demo_model");

  struct CustomException final : std::exception {
    const char* what() const noexcept override { return "custom failure"; }
  };
  const CustomException error;

  starpu_server::StarPUTaskRunnerTestAdapter::finalize_job_after_exception(
      runner_.get(), job, error, "", job->get_request_id());

  const auto metrics = starpu_server::get_metrics();
  ASSERT_NE(metrics, nullptr);
  const auto families = metrics->registry()->Collect();
  auto find_failure_value =
      [](const std::vector<prometheus::MetricFamily>& items,
         std::string_view stage, std::string_view reason,
         std::string_view model) -> std::optional<double> {
    for (const auto& family : items) {
      if (family.name != "inference_failures_total") {
        continue;
      }
      for (const auto& metric : family.metric) {
        bool stage_match = false;
        bool reason_match = false;
        bool model_match = false;
        for (const auto& label : metric.label) {
          if (label.name == "stage" && label.value == stage) {
            stage_match = true;
          } else if (label.name == "reason" && label.value == reason) {
            reason_match = true;
          } else if (label.name == "model" && label.value == model) {
            model_match = true;
          }
        }
        if (stage_match && reason_match && model_match) {
          return metric.counter.value;
        }
      }
    }
    return std::nullopt;
  };

  const auto value =
      find_failure_value(families, "execution", "exception", "demo_model");
  ASSERT_TRUE(value.has_value());
  EXPECT_DOUBLE_EQ(*value, 1.0);
}

TEST_F(
    StarPUTaskRunnerFixture,
    FinalizeJobAfterExceptionUsesObservabilityMetricsRecorder)
{
  auto observability = std::make_shared<starpu_server::RuntimeObservability>();
  observability->metrics = starpu_server::create_metrics_recorder(0);
  ASSERT_NE(observability->metrics, nullptr);

  config_.observability = observability;
  reset_runner_with_model(
      make_model_config(
          "failure_model", {make_tensor_config("input", {1}, c10::kFloat)},
          {make_tensor_config("output", {1}, c10::kFloat)}),
      1);

  auto probe = starpu_server::make_callback_probe();
  auto job = probe.job;
  job->completion().set_model_name("failure_model");
  const std::logic_error error("logic failure");

  starpu_server::StarPUTaskRunnerTestAdapter::finalize_job_after_exception(
      runner_.get(), job, error, "logic prefix", job->get_request_id());

  assert_failure_result(probe);

  const auto families =
      observability->metrics->registry()->registry()->Collect();
  const auto failures = find_counter_value(
      families, "inference_failures_total",
      {{"stage", "execution"},
       {"reason", "logic_error"},
       {"model", "failure_model"}});
  ASSERT_TRUE(failures.has_value());
  EXPECT_DOUBLE_EQ(*failures, 1.0);
}

TEST(ResultDispatcher, FinalizeJobAfterExceptionLogsWhenDispatcherMissing)
{
  auto probe = starpu_server::make_callback_probe();
  auto job = probe.job;
  const std::logic_error error("logic failure");

  CaptureStream capture{std::cerr};
  starpu_server::ResultDispatcher::finalize_job_after_exception(
      /*dispatcher=*/nullptr, /*inflight_state=*/nullptr, job, error, "",
      job->get_request_id());

  EXPECT_TRUE(probe.called);
  const auto logs = capture.str();
  const auto expected = expected_log_line(
      ErrorLevel,
      "Missing ResultDispatcher in terminal completion path; completion "
      "counter may be inconsistent");
  EXPECT_NE(logs.find(expected), std::string::npos);
}

TEST_F(
    StarPUTaskRunnerFixture,
    FinalizeJobAfterUnknownExceptionSetsFailureInfoAndCompletesJob)
{
  completed_jobs_.store(0, std::memory_order_release);
  auto probe = starpu_server::make_callback_probe();
  auto job = probe.job;
  job->completion().set_model_name("unknown_exception_model");

  starpu_server::StarPUTaskRunnerTestAdapter::
      finalize_job_after_unknown_exception(
          runner_.get(), job, "Unexpected non-standard exception",
          job->get_request_id());

  assert_failure_result(probe);
  const auto failure = job->completion().failure_info();
  ASSERT_TRUE(failure.has_value());
  EXPECT_EQ(failure->stage, "execution");
  EXPECT_EQ(failure->reason, "exception");
  EXPECT_EQ(
      failure->message,
      "Unexpected non-standard exception: Unknown non-standard exception");
  EXPECT_TRUE(failure->metrics_reported);
}

TEST_F(
    StarPUTaskRunnerFixture,
    FinalizeJobAfterUnknownExceptionReturnsWhenJobMissing)
{
  opts_.batching.max_inflight_tasks = 5;
  reset_runner_with_model(
      make_model_config(
          "test_model", {make_tensor_config("input", {1}, c10::kFloat)},
          {make_tensor_config("output", {1}, c10::kFloat)}),
      1);
  completed_jobs_.store(0, std::memory_order_release);

  starpu_server::StarPUTaskRunnerTestAdapter::reserve_inflight_slot(
      runner_.get());
  ASSERT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      1U);

  std::shared_ptr<starpu_server::InferenceJob> missing_job;

  EXPECT_NO_THROW(starpu_server::StarPUTaskRunnerTestAdapter::
                      finalize_job_after_unknown_exception(
                          runner_.get(), missing_job,
                          "Unexpected non-standard exception", -1));

  EXPECT_EQ(completed_jobs_.load(std::memory_order_acquire), 0U);
  EXPECT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      1U);
}

TEST_F(StarPUTaskRunnerFixture, TraceBatchIfEnabledLogsAggregatedRequestIds)
{
  TraceLoggerSession session;

  auto aggregated_job = std::make_shared<starpu_server::InferenceJob>();
  aggregated_job->completion().set_model_name("demo_model");
  aggregated_job->set_submission_id(11);
  aggregated_job->batch().set_effective_batch_size(2);
  populate_trace_timing(*aggregated_job);

  auto sub_job_a = std::make_shared<starpu_server::InferenceJob>();
  sub_job_a->set_request_id(41);
  auto sub_job_b = std::make_shared<starpu_server::InferenceJob>();
  sub_job_b->set_request_id(42);

  aggregated_job->batch().set_aggregated_sub_jobs(
      {make_aggregated_sub_job(sub_job_a, 41),
       make_aggregated_sub_job(sub_job_b, 42)});
  aggregated_job->batch().set_logical_job_count(2);

  starpu_server::StarPUTaskRunnerTestAdapter::trace_batch_if_enabled(
      runner_.get(), aggregated_job, /*warmup_job=*/false,
      aggregated_job->submission_id());

  session.close();
  const auto trace_content = read_trace_file(session.path());
  ASSERT_FALSE(trace_content.empty());
  EXPECT_NE(trace_content.find("\"request_ids\":[41,42]"), std::string::npos)
      << trace_content;
}

TEST_F(StarPUTaskRunnerFixture, TraceBatchIfEnabledFallsBackToSubJobRequestIds)
{
  TraceLoggerSession session;

  auto aggregated_job = std::make_shared<starpu_server::InferenceJob>();
  aggregated_job->completion().set_model_name("demo_model");
  aggregated_job->set_submission_id(15);
  aggregated_job->batch().set_effective_batch_size(2);
  populate_trace_timing(*aggregated_job);

  auto explicit_job = std::make_shared<starpu_server::InferenceJob>();
  explicit_job->set_request_id(77);
  auto inferred_job = std::make_shared<starpu_server::InferenceJob>();
  inferred_job->set_request_id(88);

  aggregated_job->batch().set_aggregated_sub_jobs(
      {make_aggregated_sub_job(explicit_job, 77),
       make_aggregated_sub_job(inferred_job, -1)});
  aggregated_job->batch().set_logical_job_count(2);

  starpu_server::StarPUTaskRunnerTestAdapter::trace_batch_if_enabled(
      runner_.get(), aggregated_job, /*warmup_job=*/false,
      aggregated_job->submission_id());

  session.close();
  const auto trace_content = read_trace_file(session.path());
  ASSERT_FALSE(trace_content.empty());
  EXPECT_NE(trace_content.find("\"request_ids\":[77,88]"), std::string::npos)
      << trace_content;
}

TEST_F(
    StarPUTaskRunnerFixture,
    TraceBatchIfEnabledUsesJobRequestIdWhenNotAggregated)
{
  TraceLoggerSession session;

  auto job = make_job(901, {torch::ones({1})}, {at::kFloat});
  job->completion().set_model_name("demo_model");
  job->set_submission_id(21);
  job->batch().set_effective_batch_size(1);
  populate_trace_timing(*job);

  starpu_server::StarPUTaskRunnerTestAdapter::trace_batch_if_enabled(
      runner_.get(), job, /*warmup_job=*/false, job->submission_id());

  session.close();
  const auto trace_content = read_trace_file(session.path());
  ASSERT_FALSE(trace_content.empty());
  EXPECT_NE(trace_content.find("\"request_ids\":[901]"), std::string::npos)
      << trace_content;
}

TEST_F(
    StarPUTaskRunnerFixture,
    TraceBatchIfEnabledHandlesMissingLastEnqueuedTimestamp)
{
  TraceLoggerSession session;

  auto job = make_job(777, {torch::ones({1})}, {at::kFloat});
  job->completion().set_model_name("demo_trace");
  job->set_submission_id(31);
  job->batch().set_effective_batch_size(1);
  populate_trace_timing(*job);
  job->timing_info().submission_id = job->submission_id();
  job->timing_info().last_enqueued_time =
      starpu_server::MonotonicClock::time_point{};

  starpu_server::StarPUTaskRunnerTestAdapter::trace_batch_if_enabled(
      runner_.get(), job, /*warmup_job=*/false, job->submission_id());

  session.close();
  const auto trace_content = read_trace_file(session.path());
  ASSERT_FALSE(trace_content.empty());

  constexpr std::string_view kBatchEvent = "\"name\":\"batch\"";
  EXPECT_NE(trace_content.find(kBatchEvent), std::string::npos)
      << trace_content;
  const auto id_token = std::format("\"batch_id\":{}", job->submission_id());
  EXPECT_NE(trace_content.find(id_token), std::string::npos) << trace_content;
}

TEST_F(
    StarPUTaskRunnerFixture,
    TraceBatchIfEnabledClampsLastEnqueuedTimestampBeforeStart)
{
  using clock = starpu_server::MonotonicClock;
  using namespace std::chrono_literals;

  TraceLoggerSession session;

  auto job = make_job(778, {torch::ones({1})}, {at::kFloat});
  job->completion().set_model_name("demo_trace");
  job->set_submission_id(32);
  job->batch().set_effective_batch_size(1);
  populate_trace_timing(*job);
  job->timing_info().submission_id = job->submission_id();

  job->timing_info().last_enqueued_time =
      job->timing_info().enqueued_time - 5ms;
  ASSERT_TRUE(job->timing_info().enqueued_time != clock::time_point{});
  ASSERT_LT(
      job->timing_info().last_enqueued_time, job->timing_info().enqueued_time);

  starpu_server::StarPUTaskRunnerTestAdapter::trace_batch_if_enabled(
      runner_.get(), job, /*warmup_job=*/false, job->submission_id());

  session.close();
  const auto trace_content = read_trace_file(session.path());
  ASSERT_FALSE(trace_content.empty());

  constexpr std::string_view kBatchEvent = "\"name\":\"batch\"";
  EXPECT_NE(trace_content.find(kBatchEvent), std::string::npos)
      << trace_content;
  const auto id_token = std::format("\"batch_id\":{}", job->submission_id());
  EXPECT_NE(trace_content.find(id_token), std::string::npos) << trace_content;
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
    StarPUTaskRunnerFixture, SubmitInferenceTaskLogsBatchWhenTraceLoggerEnabled)
{
  TraceLoggerSession session;

  opts_.devices.use_cpu = true;
  opts_.devices.use_cuda = false;

  auto model_config = make_model_config(
      "trace_model", {make_tensor_config("input0", {1}, at::kFloat)},
      {make_tensor_config("output0", {1}, at::kFloat)});
  reset_runner_with_model(model_config, /*pool_size=*/1);

  model_cpu_ = torch::jit::script::Module("trace_model");
  model_cpu_.define(R"JIT(
        def forward(self, x):
            return x + 1
    )JIT");

  constexpr int kRequestId = 602;
  constexpr int kSubmissionId = 77;
  const auto tensor_opts = torch::TensorOptions().dtype(torch::kFloat);
  auto job =
      make_job(kRequestId, {torch::ones({1}, tensor_opts)}, {at::kFloat});
  job->completion().set_model_name("trace_model");
  job->set_output_tensors({torch::zeros({1}, tensor_opts)});
  job->set_submission_id(kSubmissionId);
  job->timing_info().submission_id = kSubmissionId;
  job->set_start_time(starpu_server::MonotonicClock::now());

  ASSERT_TRUE(starpu_setup_->has_input_pool());
  ASSERT_TRUE(starpu_setup_->has_output_pool());

  ASSERT_NO_THROW(runner_->submit_inference_task(job));
  starpu_task_wait_for_all();

  session.close();
  const auto trace_content = read_trace_file(session.path());
  ASSERT_FALSE(trace_content.empty());
  EXPECT_NE(trace_content.find("\"batch_submitted\""), std::string::npos)
      << trace_content;
  const auto expected_ids = std::format("\"request_ids\":[{}]", kRequestId);
  EXPECT_NE(trace_content.find(expected_ids), std::string::npos)
      << trace_content;
}

TEST_F(
    StarPUTaskRunnerFixture,
    SubmitInferenceTaskWithoutPoolsLogsBatchWhenTraceLoggerEnabled)
{
  TraceLoggerSession session;

  opts_.devices.use_cpu = true;
  opts_.devices.use_cuda = false;

  auto model_config = make_model_config("trace_no_pools", {}, {});
  reset_runner_with_model(model_config, /*pool_size=*/1);

  model_cpu_ = torch::jit::script::Module("trace_no_pools");
  model_cpu_.define(R"JIT(
        def forward(self, x):
            return x + 1
    )JIT");

  constexpr int kRequestId = 905;
  constexpr int kSubmissionId = 58;
  const auto tensor_opts = torch::TensorOptions().dtype(torch::kFloat);
  auto job =
      make_job(kRequestId, {torch::ones({1}, tensor_opts)}, {at::kFloat});
  job->completion().set_model_name("trace_no_pools");
  job->set_output_tensors({torch::zeros({1}, tensor_opts)});
  job->set_submission_id(kSubmissionId);
  job->timing_info().submission_id = kSubmissionId;
  job->set_start_time(starpu_server::MonotonicClock::now());
  populate_trace_timing(*job);

  ASSERT_FALSE(starpu_setup_->has_input_pool());
  ASSERT_FALSE(starpu_setup_->has_output_pool());

  ASSERT_NO_THROW(runner_->submit_inference_task(job));
  starpu_task_wait_for_all();

  session.close();
  const auto trace_content = read_trace_file(session.path());
  ASSERT_FALSE(trace_content.empty());
  EXPECT_NE(trace_content.find("\"batch_submitted\""), std::string::npos)
      << trace_content;
  const auto expected_ids = std::format("\"request_ids\":[{}]", kRequestId);
  EXPECT_NE(trace_content.find(expected_ids), std::string::npos)
      << trace_content;
}
