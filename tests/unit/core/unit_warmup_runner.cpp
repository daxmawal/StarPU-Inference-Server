#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/inference_runner.hpp"
#include "core/warmup.hpp"
#include "monitoring/metrics.hpp"
#include "starpu_task_worker/inference_queue.hpp"
#include "test_helpers.hpp"
#include "test_inference_runner.hpp"
#include "test_warmup_runner.hpp"

namespace starpu_server::testing {
auto collect_device_workers_for_test(const RuntimeConfig& opts)
    -> std::map<int, std::vector<int>>;
using WarmupThreadHook = std::function<void()>;
auto set_warmup_server_thread_hook(WarmupThreadHook hook) -> WarmupThreadHook;
auto set_warmup_client_thread_hook(WarmupThreadHook hook) -> WarmupThreadHook;
auto set_warmup_drain_timeout_for_test(
    std::optional<std::chrono::milliseconds> timeout)
    -> std::optional<std::chrono::milliseconds>;
auto set_warmup_drain_wait_step_for_test(
    std::optional<std::chrono::milliseconds> wait_step)
    -> std::optional<std::chrono::milliseconds>;
}  // namespace starpu_server::testing

namespace {
auto
test_worker_stream_query(
    unsigned int device_id, int* worker_ids,
    enum starpu_worker_archtype type) -> int
{
  if (type != STARPU_CUDA_WORKER || device_id != 0U) {
    return 0;
  }
  if (worker_ids == nullptr) {
    return 0;
  }
  worker_ids[0] = 7;
  worker_ids[1] = 9;
  return 2;
}

class WorkerStreamQueryGuard {
 public:
  WorkerStreamQueryGuard()
  {
    starpu_server::StarPUSetup::set_worker_stream_query_fn(
        &test_worker_stream_query);
  }
  ~WorkerStreamQueryGuard()
  {
    starpu_server::StarPUSetup::reset_worker_stream_query_fn();
  }
  WorkerStreamQueryGuard(const WorkerStreamQueryGuard&) = delete;
  auto operator=(const WorkerStreamQueryGuard&) -> WorkerStreamQueryGuard& =
                                                       delete;
};

class WarmupThreadHookGuard {
 public:
  WarmupThreadHookGuard(
      starpu_server::testing::WarmupThreadHook server_hook,
      starpu_server::testing::WarmupThreadHook client_hook)
      : server_prev_(starpu_server::testing::set_warmup_server_thread_hook(
            std::move(server_hook))),
        client_prev_(starpu_server::testing::set_warmup_client_thread_hook(
            std::move(client_hook)))
  {
  }
  ~WarmupThreadHookGuard()
  {
    starpu_server::testing::set_warmup_server_thread_hook(
        std::move(server_prev_));
    starpu_server::testing::set_warmup_client_thread_hook(
        std::move(client_prev_));
  }
  WarmupThreadHookGuard(const WarmupThreadHookGuard&) = delete;
  auto operator=(const WarmupThreadHookGuard&) -> WarmupThreadHookGuard& =
                                                      delete;

 private:
  starpu_server::testing::WarmupThreadHook server_prev_;
  starpu_server::testing::WarmupThreadHook client_prev_;
};

class WarmupTimingOverrideGuard {
 public:
  WarmupTimingOverrideGuard(
      std::optional<std::chrono::milliseconds> timeout,
      std::optional<std::chrono::milliseconds> wait_step)
      : timeout_prev_(
            starpu_server::testing::set_warmup_drain_timeout_for_test(timeout)),
        wait_step_prev_(
            starpu_server::testing::set_warmup_drain_wait_step_for_test(
                wait_step))
  {
  }
  ~WarmupTimingOverrideGuard()
  {
    starpu_server::testing::set_warmup_drain_timeout_for_test(timeout_prev_);
    starpu_server::testing::set_warmup_drain_wait_step_for_test(
        wait_step_prev_);
  }
  WarmupTimingOverrideGuard(const WarmupTimingOverrideGuard&) = delete;
  auto operator=(const WarmupTimingOverrideGuard&)
      -> WarmupTimingOverrideGuard& = delete;

 private:
  std::optional<std::chrono::milliseconds> timeout_prev_;
  std::optional<std::chrono::milliseconds> wait_step_prev_;
};
}  // namespace

TEST_F(WarmupRunnerTest, ClientWorkerPositiveRequestNb_Unit)
{
  auto device_workers = make_device_workers();
  starpu_server::InferenceQueue queue;

  const auto enqueued_jobs =
      starpu_server::WarmupRunnerTestHelper::client_worker(
          *runner, device_workers, queue, 2);

  std::vector<int> request_ids;
  std::vector<int> worker_ids;
  for (;;) {
    std::shared_ptr<starpu_server::InferenceJob> job;
    if (!queue.wait_and_pop(job)) {
      break;
    }
    request_ids.push_back(job->get_request_id());
    const int worker = job->get_fixed_worker_id().value_or(-1);
    ASSERT_NE(worker, -1);
    worker_ids.push_back(worker);
  }

  EXPECT_EQ(enqueued_jobs, 4U);
  ASSERT_EQ(request_ids.size(), 4U);
  EXPECT_EQ(request_ids, (std::vector<int>{0, 1, 2, 3}));
  EXPECT_EQ(worker_ids, (std::vector<int>{1, 1, 2, 2}));
}

TEST_F(WarmupRunnerTest, WarmupPregenInputsRespected_Unit)
{
  auto device_workers = make_device_workers();

  opts.seed = 0;
  opts.batching.warmup_pregen_inputs = 1;
  starpu_server::InferenceQueue queue_single;
  starpu_server::WarmupRunnerTestHelper::client_worker(
      *runner, device_workers, queue_single, 1);

  std::unordered_set<const void*> unique_single;
  for (;;) {
    std::shared_ptr<starpu_server::InferenceJob> job;
    if (!queue_single.wait_and_pop(job)) {
      break;
    }
    unique_single.insert(job->get_input_tensors()[0].data_ptr());
  }
  EXPECT_EQ(unique_single.size(), 1U);

  opts.seed = 0;
  opts.batching.warmup_pregen_inputs = 2;
  starpu_server::InferenceQueue queue_double;
  constexpr int kDoublerequest_nb = 5;
  starpu_server::WarmupRunnerTestHelper::client_worker(
      *runner, device_workers, queue_double, kDoublerequest_nb);

  std::unordered_set<const void*> unique_double;
  for (;;) {
    std::shared_ptr<starpu_server::InferenceJob> job;
    if (!queue_double.wait_and_pop(job)) {
      break;
    }
    unique_double.insert(job->get_input_tensors()[0].data_ptr());
  }
  EXPECT_EQ(unique_double.size(), 2U);
}

TEST_F(WarmupRunnerTest, ClientWorkerStopsWhenQueuePushFails)
{
  auto device_workers = make_device_workers();
  starpu_server::InferenceQueue queue;
  const int request_nb = 1;

  testing::internal::CaptureStderr();
  queue.shutdown();
  const auto enqueued_jobs =
      starpu_server::WarmupRunnerTestHelper::client_worker(
          *runner, device_workers, queue, request_nb);
  const std::string captured = testing::internal::GetCapturedStderr();

  EXPECT_EQ(enqueued_jobs, 0U);
  EXPECT_NE(captured.find("Failed to enqueue job"), std::string::npos);

  std::shared_ptr<starpu_server::InferenceJob> job;
  EXPECT_FALSE(queue.wait_and_pop(job));
}

TEST_F(WarmupRunnerTest, ClientWorkerReportsPartialEnqueueWhenQueueIsFull)
{
  auto device_workers = make_device_workers();
  starpu_server::InferenceQueue queue(/*max_size=*/1);

  testing::internal::CaptureStderr();
  const auto enqueued_jobs =
      starpu_server::WarmupRunnerTestHelper::client_worker(
          *runner, device_workers, queue, /*request_nb_per_worker=*/2);
  const std::string captured = testing::internal::GetCapturedStderr();

  EXPECT_EQ(enqueued_jobs, 1U);
  EXPECT_NE(captured.find("queue is full"), std::string::npos);

  std::shared_ptr<starpu_server::InferenceJob> job;
  EXPECT_TRUE(queue.wait_and_pop(job));
  EXPECT_FALSE(queue.wait_and_pop(job));
}

TEST(WarmupRunnerEdgesTest, CollectDeviceWorkersAddsCudaWorkers)
{
  WorkerStreamQueryGuard guard;
  starpu_server::RuntimeConfig opts;
  opts.devices.use_cpu = false;
  opts.devices.use_cuda = true;
  opts.devices.ids = {0};

  const auto workers =
      starpu_server::testing::collect_device_workers_for_test(opts);

  const auto it = workers.find(0);
  ASSERT_NE(it, workers.end());
  EXPECT_EQ(it->second, (std::vector<int>{7, 9}));
}

TEST(WarmupRunnerEdgesTest, CollectDeviceWorkersSkipsCudaDevicesWithoutWorkers)
{
  WorkerStreamQueryGuard guard;
  starpu_server::RuntimeConfig opts;
  opts.devices.use_cpu = false;
  opts.devices.use_cuda = true;
  opts.devices.ids = {0, 1};

  const auto workers =
      starpu_server::testing::collect_device_workers_for_test(opts);

  const auto present_device = workers.find(0);
  ASSERT_NE(present_device, workers.end());
  EXPECT_EQ(present_device->second, (std::vector<int>{7, 9}));
  EXPECT_EQ(workers.find(1), workers.end());
}

TEST(WarmupRunnerEdgesTest, RunCapturesClientThreadException)
{
  WarmupRunnerTestFixture fixture;
  fixture.init();
  fixture.opts.batching.warmup_pregen_inputs = 1;
  auto runner = fixture.make_runner();

  std::atomic<bool> hook_called{false};
  WarmupThreadHookGuard guard(
      starpu_server::testing::WarmupThreadHook{}, [&hook_called]() {
        hook_called.store(true, std::memory_order_relaxed);
        throw std::runtime_error("client hook failure");
      });

  EXPECT_THROW(runner.run(1), std::runtime_error);
  EXPECT_TRUE(hook_called.load(std::memory_order_relaxed));
}

TEST(WarmupRunnerEdgesTest, RunCapturesServerThreadException)
{
  WarmupRunnerTestFixture fixture;
  fixture.init();
  fixture.opts.batching.warmup_pregen_inputs = 1;
  auto runner = fixture.make_runner();

  std::atomic<bool> hook_called{false};
  WarmupThreadHookGuard guard(
      [&hook_called]() {
        hook_called.store(true, std::memory_order_relaxed);
        throw std::runtime_error("server hook failure");
      },
      starpu_server::testing::WarmupThreadHook{});

  starpu_server::CaptureStream capture{std::cerr};
  EXPECT_THROW(runner.run(1), std::runtime_error);
  EXPECT_TRUE(hook_called.load(std::memory_order_relaxed));
  EXPECT_NE(
      capture.str().find("[Warmup] Failed to enqueue job 0"),
      std::string::npos);
}

TEST(WarmupRunnerEdgesTest, RunAppliesConfiguredInflightLimitBelowQueueCap)
{
  WarmupRunnerTestFixture fixture;
  fixture.init();
  fixture.opts.batching.max_queue_size = 11;
  fixture.opts.batching.max_inflight_tasks = 4;
  fixture.opts.batching.warmup_pregen_inputs = 1;
  auto runner = fixture.make_runner();

  starpu_server::shutdown_metrics();
  ASSERT_TRUE(starpu_server::init_metrics(0));
  struct MetricsGuard {
    ~MetricsGuard() { starpu_server::shutdown_metrics(); }
  } guard;

  EXPECT_NO_THROW(runner.run(0));

  const auto metrics = starpu_server::get_metrics();
  ASSERT_NE(metrics, nullptr);
  ASSERT_NE(metrics->gauges().max_inflight_tasks, nullptr);
  EXPECT_DOUBLE_EQ(
      metrics->gauges().max_inflight_tasks->Value(),
      static_cast<double>(fixture.opts.batching.max_inflight_tasks));
}

TEST(WarmupRunnerEdgesTest, RunTimesOutWhenClientStaysPendingPastDeadline)
{
  using namespace std::chrono_literals;

  WarmupRunnerTestFixture fixture;
  fixture.init();
  fixture.opts.batching.warmup_pregen_inputs = 1;
  auto runner = fixture.make_runner();

  WarmupTimingOverrideGuard timing_guard(0ms, 1ms);
  WarmupThreadHookGuard thread_guard(
      starpu_server::testing::WarmupThreadHook{},
      [] { std::this_thread::sleep_for(10ms); });

  starpu_server::CaptureStream capture{std::cerr};
  try {
    runner.run(1);
    FAIL() << "Expected timeout while waiting for client completion.";
  }
  catch (const std::runtime_error& e) {
    const std::string message = e.what();
    EXPECT_NE(message.find("Warmup drain timeout"), std::string::npos);
    EXPECT_NE(message.find("client_done=false"), std::string::npos);
  }
  EXPECT_NE(capture.str().find("Warmup drain timeout"), std::string::npos);
}

TEST(WarmupRunnerEdgesTest, RunTimesOutWhenCompletionLagsPastDeadline)
{
  using namespace std::chrono_literals;

  WarmupRunnerTestFixture fixture;
  fixture.init();
  fixture.opts.batching.warmup_pregen_inputs = 1;
  auto runner = fixture.make_runner();

  WarmupTimingOverrideGuard timing_guard(300ms, 1ms);
  WarmupThreadHookGuard thread_guard(
      [] { std::this_thread::sleep_for(600ms); },
      starpu_server::testing::WarmupThreadHook{});

  starpu_server::CaptureStream capture{std::cerr};
  try {
    runner.run(1);
    FAIL() << "Expected timeout while draining completed warmup jobs.";
  }
  catch (const std::runtime_error& e) {
    const std::string message = e.what();
    EXPECT_NE(message.find("Warmup drain timeout"), std::string::npos);
    EXPECT_NE(message.find("client_done=true"), std::string::npos);
  }
  EXPECT_NE(capture.str().find("Warmup drain timeout"), std::string::npos);
}

TEST(WarmupRunnerEdgesTest, RunPropagatesCompletionObserverException)
{
  WarmupRunnerTestFixture fixture;
  fixture.init();
  fixture.opts.batching.warmup_pregen_inputs = 1;
  auto runner = fixture.make_runner([](std::atomic<std::size_t>&) {
    throw std::runtime_error("completion observer failure");
  });

  starpu_server::CaptureStream capture{std::cerr};
  try {
    runner.run(1);
    FAIL() << "Expected completion observer exception.";
  }
  catch (const std::runtime_error& e) {
    EXPECT_EQ(std::string(e.what()), "completion observer failure");
  }
  EXPECT_NE(
      capture.str().find("[Warmup] Failed to enqueue job 0"),
      std::string::npos);
}

TEST(WarmupRunnerEdgesTest, RunWarmupSkipsWhenNoDevicesConfigured)
{
  WarmupRunnerTestFixture fixture;
  fixture.init();
  fixture.opts.verbosity = starpu_server::VerbosityLevel::Info;
  fixture.opts.devices.use_cpu = false;
  fixture.opts.devices.use_cuda = false;
  fixture.starpu = std::make_unique<starpu_server::StarPUSetup>(fixture.opts);
  fixture.model_cpu = starpu_server::make_identity_model();
  fixture.models_gpu.clear();
  fixture.outputs_ref = {torch::zeros({1})};

  starpu_server::CaptureStream capture{std::cout};
  starpu_server::run_warmup(
      fixture.opts, *fixture.starpu, fixture.model_cpu, fixture.models_gpu,
      fixture.outputs_ref);
  EXPECT_TRUE(capture.str().empty());
}

TEST(WarmupRunnerEdgesTest, RunWarmupSkipsWhenNoWarmupRequestsConfigured)
{
  WarmupRunnerTestFixture fixture;
  fixture.init();
  fixture.opts.verbosity = starpu_server::VerbosityLevel::Info;
  fixture.opts.batching.warmup_request_nb = 0;
  fixture.opts.batching.warmup_batches_per_worker = 0;
  fixture.starpu = std::make_unique<starpu_server::StarPUSetup>(fixture.opts);
  fixture.model_cpu = starpu_server::make_identity_model();
  fixture.models_gpu.clear();
  fixture.outputs_ref = {torch::zeros({1})};

  starpu_server::CaptureStream capture{std::cout};
  starpu_server::run_warmup(
      fixture.opts, *fixture.starpu, fixture.model_cpu, fixture.models_gpu,
      fixture.outputs_ref);
  EXPECT_TRUE(capture.str().empty());
}

TEST(WarmupRunnerEdgesTest, RunWarmupSkipsWhenComputedRequestsNonPositive)
{
  WarmupRunnerTestFixture fixture;
  fixture.init();
  fixture.opts.verbosity = starpu_server::VerbosityLevel::Info;
  fixture.opts.batching.warmup_request_nb = -1;
  fixture.opts.batching.warmup_batches_per_worker =
      std::numeric_limits<int>::max();
  fixture.opts.batching.max_batch_size = std::numeric_limits<int>::max() - 1;

  const int configured_batches =
      std::max(0, fixture.opts.batching.warmup_batches_per_worker);
  ASSERT_GT(configured_batches, 0);
  const int max_batch_size = std::max(1, fixture.opts.batching.max_batch_size);
  const auto product =
      static_cast<long long>(configured_batches) * max_batch_size;
  ASSERT_GT(product, static_cast<long long>(std::numeric_limits<int>::max()));

  fixture.starpu = std::make_unique<starpu_server::StarPUSetup>(fixture.opts);
  fixture.model_cpu = starpu_server::make_identity_model();
  fixture.models_gpu.clear();
  fixture.outputs_ref = {torch::zeros({1})};

  starpu_server::CaptureStream capture{std::cout};
  starpu_server::run_warmup(
      fixture.opts, *fixture.starpu, fixture.model_cpu, fixture.models_gpu,
      fixture.outputs_ref);
  EXPECT_TRUE(capture.str().empty());
}

TEST(WarmupRunnerEdgesTest, RunWarmupLogsCpuAndCudaTargetDescription)
{
  skip_if_no_cuda();
  WarmupRunnerTestFixture fixture;
  fixture.init();
  fixture.opts.verbosity = starpu_server::VerbosityLevel::Info;
  fixture.opts.devices.use_cpu = true;
  fixture.opts.devices.use_cuda = true;
  fixture.opts.devices.ids = {0};
  fixture.opts.batching.warmup_request_nb = 1;
  fixture.opts.batching.warmup_batches_per_worker = 0;
  fixture.opts.batching.warmup_pregen_inputs = 0;
  fixture.starpu = std::make_unique<starpu_server::StarPUSetup>(fixture.opts);
  fixture.model_cpu = starpu_server::make_identity_model();
  fixture.models_gpu.clear();
  fixture.outputs_ref = {torch::zeros({1})};

  starpu_server::CaptureStream capture{std::cout};
  starpu_server::run_warmup(
      fixture.opts, *fixture.starpu, fixture.model_cpu, fixture.models_gpu,
      fixture.outputs_ref);
  const std::string log = capture.str();
  EXPECT_NE(log.find("CPU and CUDA workers"), std::string::npos);
}

TEST(WarmupRunnerEdgesTest, RunWarmupLogsCudaTargetDescription)
{
  skip_if_no_cuda();
  WarmupRunnerTestFixture fixture;
  fixture.init();
  fixture.opts.verbosity = starpu_server::VerbosityLevel::Info;
  fixture.opts.devices.use_cpu = false;
  fixture.opts.devices.use_cuda = true;
  fixture.opts.devices.ids = {0};
  fixture.opts.batching.warmup_request_nb = 1;
  fixture.opts.batching.warmup_batches_per_worker = 0;
  fixture.opts.batching.warmup_pregen_inputs = 0;
  fixture.starpu = std::make_unique<starpu_server::StarPUSetup>(fixture.opts);
  fixture.model_cpu = starpu_server::make_identity_model();
  fixture.models_gpu.clear();
  fixture.outputs_ref = {torch::zeros({1})};

  starpu_server::CaptureStream capture{std::cout};
  starpu_server::run_warmup(
      fixture.opts, *fixture.starpu, fixture.model_cpu, fixture.models_gpu,
      fixture.outputs_ref);
  const std::string log = capture.str();
  EXPECT_NE(log.find("per CUDA workers"), std::string::npos);
}

TEST(WarmupRunnerEdgesTest, RunWarmupLogsCpuTargetDescription)
{
  WarmupRunnerTestFixture fixture;
  fixture.init();
  fixture.opts.verbosity = starpu_server::VerbosityLevel::Info;
  fixture.opts.devices.use_cpu = true;
  fixture.opts.devices.use_cuda = false;
  fixture.opts.batching.warmup_request_nb = 1;
  fixture.opts.batching.warmup_batches_per_worker = 0;
  fixture.opts.batching.warmup_pregen_inputs = 0;
  fixture.starpu = std::make_unique<starpu_server::StarPUSetup>(fixture.opts);
  fixture.model_cpu = starpu_server::make_identity_model();
  fixture.models_gpu.clear();
  fixture.outputs_ref = {torch::zeros({1})};

  starpu_server::CaptureStream capture{std::cout};
  starpu_server::run_warmup(
      fixture.opts, *fixture.starpu, fixture.model_cpu, fixture.models_gpu,
      fixture.outputs_ref);
  const std::string log = capture.str();
  EXPECT_NE(log.find("per CPU workers"), std::string::npos);
}
