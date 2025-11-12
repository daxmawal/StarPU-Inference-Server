#include <starpu.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>

#include "core/inference_task.hpp"
#include "exceptions.hpp"
#include "starpu_task_worker/task_runner_internal.hpp"
#include "test_starpu_task_runner.hpp"
#include "utils/batching_trace_logger.hpp"
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

  static auto merge_input_tensors(
      const std::vector<std::shared_ptr<InferenceJob>>& jobs,
      int64_t total_samples) -> std::vector<torch::Tensor>
  {
    return StarPUTaskRunner::merge_input_tensors(jobs, total_samples);
  }

  static auto merge_input_memory_holders(
      const std::vector<std::shared_ptr<InferenceJob>>& jobs)
      -> std::vector<std::shared_ptr<const void>>
  {
    return StarPUTaskRunner::merge_input_memory_holders(jobs);
  }

  static void set_submit_hook(std::function<void()> hook)
  {
    task_runner_internal::set_submit_inference_task_hook(std::move(hook));
  }

  static void reset_submit_hook()
  {
    task_runner_internal::reset_submit_inference_task_hook();
  }

  static auto validate_batch_and_copy_inputs(
      StarPUTaskRunner* runner, const std::shared_ptr<InferenceJob>& job,
      InputSlotPool* input_pool, int input_slot) -> int64_t
  {
    StarPUTaskRunner::PoolResources pools{};
    pools.input_pool = input_pool;
    pools.input_slot = input_slot;
    return runner->validate_batch_and_copy_inputs(job, pools);
  }

  static auto configure_task_context(
      InferenceTask& task, InputSlotPool* input_pool, int input_slot,
      OutputSlotPool* output_pool, int output_slot,
      const std::vector<starpu_data_handle_t>& input_handles,
      const std::vector<starpu_data_handle_t>& output_handles,
      int64_t batch_size) -> std::shared_ptr<InferenceCallbackContext>
  {
    StarPUTaskRunner::PoolResources pools{};
    pools.input_pool = input_pool;
    pools.input_slot = input_slot;
    pools.output_pool = output_pool;
    pools.output_slot = output_slot;
    return StarPUTaskRunner::configure_task_context(
        task, pools, input_handles, output_handles, batch_size);
  }

  static void trace_batch_if_enabled(
      StarPUTaskRunner* runner, const std::shared_ptr<InferenceJob>& job,
      bool warmup_job, int submission_id)
  {
    runner->trace_batch_if_enabled(job, warmup_job, submission_id);
  }

  static void enqueue_prepared_job(
      StarPUTaskRunner* runner, const std::shared_ptr<InferenceJob>& job)
  {
    runner->enqueue_prepared_job(job);
  }

  static auto wait_for_prepared_job(StarPUTaskRunner* runner)
      -> std::shared_ptr<InferenceJob>
  {
    return runner->wait_for_prepared_job();
  }

  static void release_pending_jobs(
      const std::shared_ptr<InferenceJob>& job,
      std::vector<std::shared_ptr<InferenceJob>>& pending_jobs)
  {
    StarPUTaskRunner::release_pending_jobs(job, pending_jobs);
  }

  static void store_completed_job_result(
      StarPUTaskRunner* runner, const std::shared_ptr<InferenceJob>& job,
      const std::vector<torch::Tensor>& results, double latency_ms)
  {
    runner->store_completed_job_result(job, results, latency_ms);
  }

  static void ensure_callback_timing(detail::TimingInfo& timing)
  {
    StarPUTaskRunner::ensure_callback_timing(timing);
  }

  static void record_job_metrics(
      StarPUTaskRunner* runner, const std::shared_ptr<InferenceJob>& job,
      StarPUTaskRunner::DurationMs latency, std::size_t batch_size)
  {
    runner->record_job_metrics(job, latency, batch_size);
  }

  static void finalize_job_completion(
      StarPUTaskRunner* runner, const std::shared_ptr<InferenceJob>& job)
  {
    runner->finalize_job_completion(job);
  }
};
}  // namespace starpu_server

namespace {
class TraceLoggerSession {
 public:
  TraceLoggerSession()
      : path_(
            std::filesystem::temp_directory_path() /
            std::format(
                "trace_request_ids_{}.json",
                std::chrono::steady_clock::now().time_since_epoch().count()))
  {
    starpu_server::BatchingTraceLogger::instance().configure(
        true, path_.string());
  }

  TraceLoggerSession(const TraceLoggerSession&) = delete;
  auto operator=(const TraceLoggerSession&) -> TraceLoggerSession& = delete;

  ~TraceLoggerSession()
  {
    close();
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }

  void close()
  {
    if (closed_) {
      return;
    }
    starpu_server::BatchingTraceLogger::instance().configure(false, "");
    closed_ = true;
  }

  [[nodiscard]] auto path() const -> const std::filesystem::path&
  {
    return path_;
  }

 private:
  std::filesystem::path path_;
  bool closed_{false};
};

auto
read_trace_file(const std::filesystem::path& path) -> std::string
{
  std::ifstream stream(path);
  if (!stream.is_open()) {
    return {};
  }
  return std::string(
      (std::istreambuf_iterator<char>(stream)),
      std::istreambuf_iterator<char>());
}

void
populate_trace_timing(starpu_server::InferenceJob& job)
{
  using clock = std::chrono::high_resolution_clock;
  const auto now = clock::now();
  job.timing_info().enqueued_time = now - std::chrono::milliseconds(3);
  job.timing_info().last_enqueued_time = now - std::chrono::milliseconds(2);
  job.timing_info().batch_collect_start_time =
      now - std::chrono::milliseconds(1);
  job.timing_info().batch_collect_end_time = now;
}

auto
make_aggregated_sub_job(
    const std::shared_ptr<starpu_server::InferenceJob>& job,
    int request_id) -> starpu_server::InferenceJob::AggregatedSubJob
{
  starpu_server::InferenceJob::AggregatedSubJob aggregated{};
  aggregated.job = job;
  aggregated.request_id = request_id;
  aggregated.batch_size = 1;
  return aggregated;
}

struct VectorInterfaceSnapshot {
  starpu_vector_interface* iface = nullptr;
  std::size_t elemsize = 0;
  std::size_t allocsize = 0;
  std::size_t nx = 0;
};

auto
snapshot_vector_interfaces(starpu_data_handle_t handle)
    -> std::vector<VectorInterfaceSnapshot>
{
  std::vector<VectorInterfaceSnapshot> snapshots;
  const unsigned memory_nodes = starpu_memory_nodes_get_count();
  snapshots.reserve(memory_nodes);
  for (unsigned node = 0; node < memory_nodes; ++node) {
    auto* raw_interface = starpu_data_get_interface_on_node(handle, node);
    if (raw_interface == nullptr) {
      continue;
    }
    auto* vector_interface =
        static_cast<starpu_vector_interface*>(raw_interface);
    snapshots.push_back(VectorInterfaceSnapshot{
        vector_interface, static_cast<std::size_t>(vector_interface->elemsize),
        vector_interface->allocsize, vector_interface->nx});
  }
  return snapshots;
}

void
restore_vector_interfaces(const std::vector<VectorInterfaceSnapshot>& snapshots)
{
  for (const auto& snapshot : snapshots) {
    if (snapshot.iface == nullptr) {
      continue;
    }
    snapshot.iface->elemsize =
        static_cast<decltype(snapshot.iface->elemsize)>(snapshot.elemsize);
    snapshot.iface->allocsize = snapshot.allocsize;
    snapshot.iface->nx = static_cast<decltype(snapshot.iface->nx)>(snapshot.nx);
  }
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

TEST_F(StarPUTaskRunnerFixture, StoreCompletedJobResultTracksInputsAndOutputs)
{
  opts_.validation.validate_results = true;

  std::vector<torch::Tensor> inputs{torch::tensor({1.0, 2.0})};
  const std::vector<torch::Tensor> outputs{torch::tensor({5.0})};
  constexpr double kLatencyMs = 7.5;

  auto job = make_job(77, inputs);
  job->set_submission_id(11);
  job->timing_info().submission_id = job->submission_id();
  job->get_device_id() = 3;
  job->get_worker_id() = 9;
  job->get_executed_on() = starpu_server::DeviceType::CUDA;
  job->timing_info().callback_start_time =
      starpu_server::task_runner_internal::Clock::now();
  job->timing_info().callback_end_time =
      job->timing_info().callback_start_time + std::chrono::milliseconds(1);

  starpu_server::StarPUTaskRunnerTestAdapter::store_completed_job_result(
      runner_.get(), job, outputs, kLatencyMs);

  ASSERT_EQ(results_.size(), 1U);
  const auto& stored = results_.front();
  ASSERT_EQ(stored.inputs.size(), inputs.size());
  EXPECT_TRUE(torch::equal(stored.inputs[0], inputs[0]));
  ASSERT_EQ(stored.results.size(), outputs.size());
  EXPECT_TRUE(torch::equal(stored.results[0], outputs[0]));
  EXPECT_EQ(stored.latency_ms, kLatencyMs);
  EXPECT_EQ(stored.request_id, job->get_request_id());
  EXPECT_EQ(stored.submission_id, job->submission_id());
  EXPECT_EQ(stored.device_id, job->get_device_id());
  EXPECT_EQ(stored.worker_id, job->get_worker_id());
  EXPECT_EQ(stored.executed_on, job->get_executed_on());
  EXPECT_EQ(
      stored.timing_info.callback_start_time,
      job->timing_info().callback_start_time);
  EXPECT_TRUE(job->get_input_tensors().empty());
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
  job->set_logical_job_count(3);

  starpu_server::StarPUTaskRunnerTestAdapter::finalize_job_completion(
      runner_.get(), job);
  EXPECT_EQ(completed_jobs_.load(), 3);

  job->set_logical_job_count(0);
  starpu_server::StarPUTaskRunnerTestAdapter::finalize_job_completion(
      runner_.get(), job);
  EXPECT_EQ(completed_jobs_.load(), 4);
}

TEST_F(StarPUTaskRunnerFixture, TraceBatchIfEnabledLogsAggregatedRequestIds)
{
  TraceLoggerSession session;

  auto aggregated_job = std::make_shared<starpu_server::InferenceJob>();
  aggregated_job->set_model_name("demo_model");
  aggregated_job->set_submission_id(11);
  aggregated_job->set_effective_batch_size(2);
  populate_trace_timing(*aggregated_job);

  auto sub_job_a = std::make_shared<starpu_server::InferenceJob>();
  sub_job_a->set_request_id(41);
  auto sub_job_b = std::make_shared<starpu_server::InferenceJob>();
  sub_job_b->set_request_id(42);

  aggregated_job->set_aggregated_sub_jobs(
      {make_aggregated_sub_job(sub_job_a, 41),
       make_aggregated_sub_job(sub_job_b, 42)});
  aggregated_job->set_logical_job_count(2);

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
  aggregated_job->set_model_name("demo_model");
  aggregated_job->set_submission_id(15);
  aggregated_job->set_effective_batch_size(2);
  populate_trace_timing(*aggregated_job);

  auto explicit_job = std::make_shared<starpu_server::InferenceJob>();
  explicit_job->set_request_id(77);
  auto inferred_job = std::make_shared<starpu_server::InferenceJob>();
  inferred_job->set_request_id(88);

  aggregated_job->set_aggregated_sub_jobs(
      {make_aggregated_sub_job(explicit_job, 77),
       make_aggregated_sub_job(inferred_job, -1)});
  aggregated_job->set_logical_job_count(2);

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
  job->set_model_name("demo_model");
  job->set_submission_id(21);
  job->set_effective_batch_size(1);
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
    ValidateBatchAndCopyInputsThrowsWhenElementSizeZero)
{
  auto model_config = make_model_config(
      "input_only", {make_tensor_config("input0", {3}, at::kFloat)}, {});

  reset_runner_with_model(model_config, /*pool_size=*/1);
  ASSERT_TRUE(starpu_setup_->has_input_pool());

  auto job = make_job(
      21, {torch::ones({3}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});

  auto& input_pool = starpu_setup_->input_pool();
  const int slot = input_pool.acquire();
  const auto& handles = input_pool.handles(slot);
  ASSERT_EQ(handles.size(), 1U);
  const auto handle = handles[0];
  ASSERT_NE(handle, nullptr);

  const auto snapshots = snapshot_vector_interfaces(handle);
  ASSERT_FALSE(snapshots.empty());
  for (const auto& snapshot : snapshots) {
    if (snapshot.iface != nullptr) {
      snapshot.iface->elemsize = 0;
    }
  }

  EXPECT_THROW(
      starpu_server::StarPUTaskRunnerTestAdapter::
          validate_batch_and_copy_inputs(runner_.get(), job, &input_pool, slot),
      starpu_server::StarPUDataAcquireException);

  restore_vector_interfaces(snapshots);
  input_pool.release(slot);
}

TEST_F(
    StarPUTaskRunnerFixture,
    ValidateBatchAndCopyInputsThrowsWhenTensorBytesMisaligned)
{
  auto model_config = make_model_config(
      "input_only", {make_tensor_config("input0", {3}, at::kFloat)}, {});

  reset_runner_with_model(model_config, /*pool_size=*/1);
  ASSERT_TRUE(starpu_setup_->has_input_pool());

  auto job = make_job(
      22, {torch::ones({3}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});

  auto& input_pool = starpu_setup_->input_pool();
  const int slot = input_pool.acquire();
  const auto& handles = input_pool.handles(slot);
  ASSERT_EQ(handles.size(), 1U);
  const auto handle = handles[0];
  ASSERT_NE(handle, nullptr);

  const auto snapshots = snapshot_vector_interfaces(handle);
  ASSERT_FALSE(snapshots.empty());
  for (const auto& snapshot : snapshots) {
    if (snapshot.iface != nullptr) {
      snapshot.iface->elemsize = snapshot.elemsize + 1;
    }
  }

  EXPECT_THROW(
      starpu_server::StarPUTaskRunnerTestAdapter::
          validate_batch_and_copy_inputs(runner_.get(), job, &input_pool, slot),
      starpu_server::InvalidInputTensorException);

  restore_vector_interfaces(snapshots);
  input_pool.release(slot);
}

TEST_F(
    StarPUTaskRunnerFixture,
    ValidateBatchAndCopyInputsThrowsWhenSlotCapacityExceeded)
{
  auto model_config = make_model_config(
      "input_only", {make_tensor_config("input0", {3}, at::kFloat)}, {});

  reset_runner_with_model(model_config, /*pool_size=*/1);
  ASSERT_TRUE(starpu_setup_->has_input_pool());

  auto job = make_job(
      23, {torch::ones({3}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});

  auto& input_pool = starpu_setup_->input_pool();
  const int slot = input_pool.acquire();
  const auto& handles = input_pool.handles(slot);
  ASSERT_EQ(handles.size(), 1U);
  const auto handle = handles[0];
  ASSERT_NE(handle, nullptr);

  const auto snapshots = snapshot_vector_interfaces(handle);
  ASSERT_FALSE(snapshots.empty());
  const auto expected_bytes =
      static_cast<std::size_t>(job->get_input_tensors()[0].nbytes());
  for (const auto& snapshot : snapshots) {
    if (snapshot.iface != nullptr) {
      snapshot.iface->allocsize = expected_bytes - 1;
    }
  }

  EXPECT_THROW(
      starpu_server::StarPUTaskRunnerTestAdapter::
          validate_batch_and_copy_inputs(runner_.get(), job, &input_pool, slot),
      starpu_server::InputPoolCapacityException);

  restore_vector_interfaces(snapshots);
  input_pool.release(slot);
}

TEST_F(
    StarPUTaskRunnerFixture,
    ValidateBatchAndCopyInputsUpdatesVectorNumelWhenNeeded)
{
  auto model_config = make_model_config(
      "input_only", {make_tensor_config("input0", {3}, at::kFloat)}, {});

  reset_runner_with_model(model_config, /*pool_size=*/1);
  ASSERT_TRUE(starpu_setup_->has_input_pool());

  auto job = make_job(
      24, {torch::ones({3}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});

  auto& input_pool = starpu_setup_->input_pool();
  const int slot = input_pool.acquire();
  const auto& handles = input_pool.handles(slot);
  ASSERT_EQ(handles.size(), 1U);
  const auto handle = handles[0];
  ASSERT_NE(handle, nullptr);

  auto snapshots = snapshot_vector_interfaces(handle);
  ASSERT_FALSE(snapshots.empty());
  const auto tensor_bytes =
      static_cast<std::size_t>(job->get_input_tensors()[0].nbytes());
  const auto adjusted_elem_size = tensor_bytes / 2;
  ASSERT_GT(adjusted_elem_size, 0U);
  for (const auto& snapshot : snapshots) {
    if (snapshot.iface != nullptr) {
      snapshot.iface->elemsize = adjusted_elem_size;
    }
  }

  const auto expected_numel = tensor_bytes / adjusted_elem_size;
  const auto batch = starpu_server::StarPUTaskRunnerTestAdapter::
      validate_batch_and_copy_inputs(runner_.get(), job, &input_pool, slot);
  EXPECT_EQ(batch, 1);
  for (const auto& snapshot : snapshots) {
    if (snapshot.iface != nullptr) {
      EXPECT_EQ(
          snapshot.iface->nx,
          static_cast<decltype(snapshot.iface->nx)>(expected_numel));
    }
  }

  restore_vector_interfaces(snapshots);
  input_pool.release(slot);
}

TEST_F(
    StarPUTaskRunnerFixture,
    ValidateBatchAndCopyInputsCopiesPendingJobInputsSequentially)
{
  auto model_config = make_model_config(
      "input_only", {make_tensor_config("input0", {2}, at::kFloat)}, {});

  opts_.batching.max_batch_size = 2;
  reset_runner_with_model(model_config, /*pool_size=*/2);
  ASSERT_TRUE(starpu_setup_->has_input_pool());

  auto job = make_job(
      30,
      {torch::tensor(
          std::vector<float>{1.0F, 2.0F},
          torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  auto pending = make_job(
      31,
      {torch::tensor(
          std::vector<float>{3.0F, 4.0F},
          torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});

  job->set_pending_sub_jobs({pending});

  auto& input_pool = starpu_setup_->input_pool();
  const int slot = input_pool.acquire();
  const auto& handles = input_pool.handles(slot);
  ASSERT_EQ(handles.size(), 1U);
  const auto handle = handles[0];
  ASSERT_NE(handle, nullptr);

  auto snapshots = snapshot_vector_interfaces(handle);
  ASSERT_FALSE(snapshots.empty());

  const int64_t batch = starpu_server::StarPUTaskRunnerTestAdapter::
      validate_batch_and_copy_inputs(runner_.get(), job, &input_pool, slot);
  EXPECT_EQ(batch, 1);

  const std::vector<float> expected{1.0F, 2.0F, 3.0F, 4.0F};
  auto base_ptr = input_pool.base_ptrs(slot).at(0);
  ASSERT_NE(base_ptr, nullptr);
  std::vector<float> actual(expected.size());
  std::memcpy(actual.data(), base_ptr, expected.size() * sizeof(float));
  for (size_t idx = 0; idx < expected.size(); ++idx) {
    EXPECT_FLOAT_EQ(actual[idx], expected[idx]);
  }

  const auto total_numel = expected.size();
  for (const auto& snapshot : snapshots) {
    if (snapshot.iface != nullptr) {
      EXPECT_EQ(
          snapshot.iface->nx,
          static_cast<decltype(snapshot.iface->nx)>(total_numel));
    }
  }

  input_pool.release(slot);
}

TEST_F(
    StarPUTaskRunnerFixture,
    ValidateBatchAndCopyInputsThrowsWhenPendingJobInputCountMismatch)
{
  auto model_config = make_model_config(
      "input_only", {make_tensor_config("input0", {2}, at::kFloat)}, {});

  opts_.batching.max_batch_size = 2;
  reset_runner_with_model(model_config, /*pool_size=*/2);
  ASSERT_TRUE(starpu_setup_->has_input_pool());

  auto job = make_job(
      32,
      {torch::tensor(
          std::vector<float>{1.0F, 2.0F},
          torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  auto pending = make_job(33, {}, {});

  job->set_pending_sub_jobs({pending});

  auto& input_pool = starpu_setup_->input_pool();
  const int slot = input_pool.acquire();

  EXPECT_THROW(
      starpu_server::StarPUTaskRunnerTestAdapter::
          validate_batch_and_copy_inputs(runner_.get(), job, &input_pool, slot),
      starpu_server::InconsistentInputTensorCountException);

  input_pool.release(slot);
}

TEST_F(StarPUTaskRunnerFixture, MergeInputTensorsConcatenatesBatchedJobs)
{
  auto job_a = make_job(
      40,
      {torch::tensor(
          {{1.0F}, {2.0F}}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  auto job_b = make_job(
      41,
      {torch::tensor({{3.0F}}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});

  std::vector<std::shared_ptr<starpu_server::InferenceJob>> jobs{job_a, job_b};
  const auto merged =
      starpu_server::StarPUTaskRunnerTestAdapter::merge_input_tensors(
          jobs, /*total_samples=*/3);

  ASSERT_EQ(merged.size(), 1U);
  const auto expected = torch::tensor(
      {{1.0F}, {2.0F}, {3.0F}}, torch::TensorOptions().dtype(torch::kFloat));
  EXPECT_TRUE(torch::equal(merged[0], expected));
}

TEST_F(StarPUTaskRunnerFixture, MergeInputTensorsThrowsWhenTotalSamplesMismatch)
{
  auto job_a = make_job(
      42,
      {torch::tensor(
          {{1.0F}, {2.0F}}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  auto job_b = make_job(
      43,
      {torch::tensor({{3.0F}}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});

  std::vector<std::shared_ptr<starpu_server::InferenceJob>> jobs{job_a, job_b};

  EXPECT_THROW(
      starpu_server::StarPUTaskRunnerTestAdapter::merge_input_tensors(
          jobs, /*total_samples=*/5),
      starpu_server::InvalidInputTensorException);
}

TEST_F(
    StarPUTaskRunnerFixture, MergeInputMemoryHoldersPreservesOriginalOrdering)
{
  auto job_a = make_job(44, {});
  auto owner_a0 = std::make_shared<int>(7);
  auto owner_a1 = std::make_shared<int>(9);
  job_a->set_input_memory_holders(
      {std::shared_ptr<const void>(owner_a0, owner_a0.get()),
       std::shared_ptr<const void>(owner_a1, owner_a1.get())});

  auto job_b = make_job(45, {});
  auto owner_b0 = std::make_shared<int>(11);
  job_b->set_input_memory_holders(
      {std::shared_ptr<const void>(owner_b0, owner_b0.get())});

  std::vector<std::shared_ptr<starpu_server::InferenceJob>> jobs{job_a, job_b};
  const auto holders =
      starpu_server::StarPUTaskRunnerTestAdapter::merge_input_memory_holders(
          jobs);

  ASSERT_EQ(holders.size(), 3U);
  EXPECT_EQ(holders[0].get(), owner_a0.get());
  EXPECT_EQ(holders[1].get(), owner_a1.get());
  EXPECT_EQ(holders[2].get(), owner_b0.get());
}

TEST_F(StarPUTaskRunnerFixture, EnqueuePreparedJobDeliversJobToWaiter)
{
  auto job = make_job(46, {});

  starpu_server::StarPUTaskRunnerTestAdapter::enqueue_prepared_job(
      runner_.get(), job);

  auto dequeued =
      starpu_server::StarPUTaskRunnerTestAdapter::wait_for_prepared_job(
          runner_.get());
  ASSERT_TRUE(dequeued);
  EXPECT_EQ(dequeued, job);
}

TEST_F(
    StarPUTaskRunnerFixture,
    ReleasePendingJobsClearsInputsForAdditionalJobsOnly)
{
  auto master_job = make_job(
      47, {torch::tensor({5.0F}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  auto master_holder = std::make_shared<int>(21);
  master_job->set_input_memory_holders(
      {std::shared_ptr<const void>(master_holder, master_holder.get())});

  auto pending_job = make_job(
      48, {torch::tensor({9.0F}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  auto pending_holder = std::make_shared<int>(22);
  pending_job->set_input_memory_holders(
      {std::shared_ptr<const void>(pending_holder, pending_holder.get())});

  std::vector<std::shared_ptr<starpu_server::InferenceJob>> pending_jobs{
      pending_job};
  starpu_server::StarPUTaskRunnerTestAdapter::release_pending_jobs(
      master_job, pending_jobs);

  EXPECT_TRUE(pending_jobs.empty());
  EXPECT_TRUE(pending_job->get_input_tensors().empty());
  EXPECT_TRUE(pending_job->get_input_memory_holders().empty());
  EXPECT_FALSE(master_job->get_input_tensors().empty());
  EXPECT_FALSE(master_job->get_input_memory_holders().empty());
}

TEST_F(
    StarPUTaskRunnerFixture,
    ConfigureTaskContextThrowsWhenOutputBytesMisaligned)
{
  auto model_config = make_model_config(
      "output_only", {}, {make_tensor_config("output0", {3}, at::kFloat)});

  reset_runner_with_model(model_config, /*pool_size=*/1);
  ASSERT_TRUE(starpu_setup_->has_output_pool());

  auto job = make_job(25, {});
  job->set_output_tensors(
      {torch::zeros({3}, torch::TensorOptions().dtype(torch::kFloat))});

  starpu_server::InferenceTask task(
      starpu_setup_.get(), job, &model_cpu_, &models_gpu_, &opts_,
      dependencies_);

  auto& output_pool = starpu_setup_->output_pool();
  const int slot = output_pool.acquire();
  const auto& output_handles = output_pool.handles(slot);
  ASSERT_EQ(output_handles.size(), 1U);
  const auto handle = output_handles[0];
  ASSERT_NE(handle, nullptr);

  const auto snapshots = snapshot_vector_interfaces(handle);
  ASSERT_FALSE(snapshots.empty());
  for (const auto& snapshot : snapshots) {
    if (snapshot.iface != nullptr) {
      snapshot.iface->elemsize = snapshot.elemsize + 1;
    }
  }

  std::vector<starpu_data_handle_t> input_handles;
  EXPECT_THROW(
      starpu_server::StarPUTaskRunnerTestAdapter::configure_task_context(
          task, nullptr, -1, &output_pool, slot, input_handles, output_handles,
          /*batch_size=*/1),
      starpu_server::InvalidInferenceJobException);

  restore_vector_interfaces(snapshots);
  output_pool.release(slot);
}

TEST_F(
    StarPUTaskRunnerFixture,
    ConfigureTaskContextThrowsWhenOutputCapacityExceeded)
{
  auto model_config = make_model_config(
      "output_only", {}, {make_tensor_config("output0", {3}, at::kFloat)});

  reset_runner_with_model(model_config, /*pool_size=*/1);
  ASSERT_TRUE(starpu_setup_->has_output_pool());

  auto job = make_job(26, {});
  job->set_output_tensors(
      {torch::zeros({3}, torch::TensorOptions().dtype(torch::kFloat))});

  starpu_server::InferenceTask task(
      starpu_setup_.get(), job, &model_cpu_, &models_gpu_, &opts_,
      dependencies_);

  auto& output_pool = starpu_setup_->output_pool();
  const int slot = output_pool.acquire();
  const auto& output_handles = output_pool.handles(slot);
  ASSERT_EQ(output_handles.size(), 1U);
  const auto handle = output_handles[0];
  ASSERT_NE(handle, nullptr);

  const auto snapshots = snapshot_vector_interfaces(handle);
  ASSERT_FALSE(snapshots.empty());
  const auto tensor_bytes =
      static_cast<std::size_t>(job->get_output_tensors()[0].nbytes());
  for (const auto& snapshot : snapshots) {
    if (snapshot.iface != nullptr) {
      snapshot.iface->allocsize = tensor_bytes - 1;
    }
  }

  std::vector<starpu_data_handle_t> input_handles;
  EXPECT_THROW(
      starpu_server::StarPUTaskRunnerTestAdapter::configure_task_context(
          task, nullptr, -1, &output_pool, slot, input_handles, output_handles,
          /*batch_size=*/1),
      starpu_server::InvalidInferenceJobException);

  restore_vector_interfaces(snapshots);
  output_pool.release(slot);
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

TEST_F(
    StarPUTaskRunnerFixture,
    MaybeBuildBatchedJobPreservesEffectiveBatchSizeAfterMergingInputs)
{
  opts_.validation.validate_results = true;

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

TEST_F(StarPUTaskRunnerFixture, RunLogsDequeuedJobsAtTraceVerbosity)
{
  opts_.batching.dynamic_batching = false;
  opts_.verbosity = starpu_server::VerbosityLevel::Trace;

  constexpr int kRequestId = 123;
  auto probe = starpu_server::make_callback_probe();
  auto job = probe.job;
  job->set_request_id(kRequestId);
  job->set_input_tensors(
      {torch::ones({1}, torch::TensorOptions().dtype(torch::kFloat))});
  job->set_input_types({at::kFloat});

  ASSERT_TRUE(queue_.push(job));
  ASSERT_TRUE(queue_.push(starpu_server::InferenceJob::make_shutdown_job()));

  starpu_server::StarPUTaskRunnerTestAdapter::set_submit_hook([&]() {
    starpu_server::StarPUTaskRunnerTestAdapter::reset_submit_hook();
    throw starpu_server::InferenceEngineException("trace guard");
  });

  CaptureStream capture{std::cout};
  runner_->run();
  starpu_server::StarPUTaskRunnerTestAdapter::reset_submit_hook();

  const auto logs = capture.str();
  EXPECT_NE(logs.find("Dequeued job submission"), std::string::npos);
  EXPECT_NE(
      logs.find(std::format("(request {})", kRequestId)), std::string::npos);
  EXPECT_NE(logs.find("aggregated requests: 1"), std::string::npos);

  assert_failure_result(probe);
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
