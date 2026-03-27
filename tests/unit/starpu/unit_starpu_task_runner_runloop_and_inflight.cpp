#include "monitoring/congestion_monitor.hpp"
#include "unit_starpu_task_runner_support.hpp"

namespace {

auto
wait_until(
    const std::function<bool()>& predicate, std::chrono::milliseconds timeout,
    std::chrono::milliseconds poll_interval = std::chrono::milliseconds(10))
    -> bool
{
  const auto sleep_interval = poll_interval > std::chrono::milliseconds::zero()
                                  ? poll_interval
                                  : std::chrono::milliseconds(1);
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(sleep_interval);
  }
  return predicate();
}

struct ScopedCongestionMonitor {
  explicit ScopedCongestionMonitor(
      starpu_server::InferenceQueue* queue,
      starpu_server::congestion::Config config)
  {
    started = starpu_server::congestion::start(queue, std::move(config));
  }

  ~ScopedCongestionMonitor() { starpu_server::congestion::shutdown(); }

  bool started = false;
};

struct BatchCollectorAfterBuildJobHookGuard {
  explicit BatchCollectorAfterBuildJobHookGuard(
      std::function<void(std::shared_ptr<starpu_server::InferenceJob>&)> hook)
  {
    test_api::batch_collector_set_after_build_job_hook(std::move(hook));
  }

  ~BatchCollectorAfterBuildJobHookGuard()
  {
    test_api::batch_collector_reset_after_build_job_hook();
  }
};

struct DuplicateBatchingThreadExceptionCaptureGuard {
  explicit DuplicateBatchingThreadExceptionCaptureGuard(bool enable = true)
  {
    starpu_server::StarPUTaskRunnerTestAdapter::
        set_duplicate_batching_thread_exception_capture_for_test(enable);
  }

  ~DuplicateBatchingThreadExceptionCaptureGuard()
  {
    starpu_server::StarPUTaskRunnerTestAdapter::
        reset_duplicate_batching_thread_exception_capture_for_test();
  }
};

struct RunAfterBatchingThreadStartHookGuard {
  explicit RunAfterBatchingThreadStartHookGuard(std::function<void()> hook)
  {
    starpu_server::StarPUTaskRunnerTestAdapter::
        set_run_after_batching_thread_start_hook(std::move(hook));
  }

  ~RunAfterBatchingThreadStartHookGuard()
  {
    starpu_server::StarPUTaskRunnerTestAdapter::
        reset_run_after_batching_thread_start_hook();
  }
};

struct RunBeforeSubmitHookGuard {
  explicit RunBeforeSubmitHookGuard(std::function<void()> hook)
  {
    starpu_server::StarPUTaskRunnerTestAdapter::set_run_before_submit_hook(
        std::move(hook));
  }

  ~RunBeforeSubmitHookGuard()
  {
    starpu_server::StarPUTaskRunnerTestAdapter::reset_run_before_submit_hook();
  }
};

}  // namespace

TEST(
    TaskRunnerInternal,
    BuildRequestArrivalUsForTraceReturnsZeroWhenSubJobExpiredAndArrivalIsDefault)
{
  namespace internal = starpu_server::task_runner_internal;
  using Clock = starpu_server::MonotonicClock;

  auto aggregated = std::make_shared<starpu_server::InferenceJob>();

  std::vector<starpu_server::InferenceJob::AggregatedSubJob> sub_jobs;
  {
    auto temp_sub_job = std::make_shared<starpu_server::InferenceJob>();
    sub_jobs.push_back(
        {temp_sub_job,
         std::function<void(const std::vector<torch::Tensor>&, double)>{}, 1});
    sub_jobs.back().arrival_time = Clock::time_point{};
  }

  aggregated->batch().set_aggregated_sub_jobs(std::move(sub_jobs));

  const auto result = internal::build_request_arrival_us_for_trace(aggregated);

  ASSERT_EQ(result.size(), 1U);
  EXPECT_EQ(result[0], 0);
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

TEST_F(StarPUTaskRunnerFixture, CollectBatchReturnsFirstJobOnlyWhenQueueIsNull)
{
  opts_.batching.dynamic_batching = true;
  opts_.batching.max_batch_size = 3;
  opts_.batching.batch_coalesce_timeout_ms = 0;

  auto first = make_job(
      501, {torch::ones({1, 2}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});

  auto* original_queue =
      starpu_server::StarPUTaskRunnerTestAdapter::get_batch_collector_queue(
          runner_.get());
  ASSERT_NE(original_queue, nullptr);

  starpu_server::StarPUTaskRunnerTestAdapter::set_batch_collector_queue_to_null(
      runner_.get());

  auto collected = starpu_server::StarPUTaskRunnerTestAdapter::collect_batch(
      runner_.get(), first);

  ASSERT_EQ(collected.size(), 1U);
  EXPECT_EQ(collected[0], first);
}

TEST_F(StarPUTaskRunnerFixture, CollectBatchStopsWhenCoalesceDeadlineExpires)
{
  opts_.batching.dynamic_batching = true;
  opts_.batching.max_batch_size = 5;
  opts_.batching.batch_coalesce_timeout_ms = 5;

  auto first = make_job(
      502, {torch::ones({1, 2}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});

  auto collected = starpu_server::StarPUTaskRunnerTestAdapter::collect_batch(
      runner_.get(), first);

  ASSERT_EQ(collected.size(), 1U);
  EXPECT_EQ(collected[0], first);
}

TEST_F(
    StarPUTaskRunnerFixture,
    CollectBatchAdaptsTargetWithQueuePressureByShrinkingThenGrowing)
{
  opts_.batching.dynamic_batching = true;
  opts_.batching.max_batch_size = 4;
  opts_.batching.batch_coalesce_timeout_ms = 0;
  opts_.congestion.enabled = true;

  auto make_compatible_job = [this](int request_id) {
    return make_job(
        request_id,
        {torch::ones({1, 2}, torch::TensorOptions().dtype(torch::kFloat))},
        {at::kFloat});
  };

  int request_id = 700;
  for (int i = 0; i < 8; ++i) {
    auto first = make_compatible_job(request_id++);
    auto collected = starpu_server::StarPUTaskRunnerTestAdapter::collect_batch(
        runner_.get(), first);
    ASSERT_EQ(collected.size(), 1U);
  }

  auto low_pressure_first = make_compatible_job(request_id++);
  auto low_pressure_queued = make_compatible_job(request_id++);
  ASSERT_TRUE(queue_.push(low_pressure_queued));

  auto low_pressure_collected =
      starpu_server::StarPUTaskRunnerTestAdapter::collect_batch(
          runner_.get(), low_pressure_first);
  ASSERT_GE(low_pressure_collected.size(), 1U);
  if (low_pressure_collected.size() == 1U) {
    auto low_pressure_remaining = runner_->wait_for_next_job();
    ASSERT_EQ(low_pressure_remaining, low_pressure_queued);
  }

  constexpr std::size_t kHighPressureQueueDepth = 90;
  for (std::size_t i = 0; i < kHighPressureQueueDepth; ++i) {
    ASSERT_TRUE(queue_.push(make_compatible_job(request_id++)));
  }

  auto high_pressure_first = make_compatible_job(request_id++);
  auto high_pressure_collected =
      starpu_server::StarPUTaskRunnerTestAdapter::collect_batch(
          runner_.get(), high_pressure_first);

  EXPECT_GE(high_pressure_collected.size(), 2U);
}

TEST_F(
    StarPUTaskRunnerFixture,
    CollectBatchShrinksAtMostOncePerCongestionTickUnderLowPressure)
{
  opts_.batching.dynamic_batching = true;
  opts_.batching.max_batch_size = 4;
  opts_.batching.batch_coalesce_timeout_ms = 0;
  opts_.congestion.enabled = true;
  opts_.congestion.tick_interval_ms = 1000;
  opts_.congestion.entry_horizon_ms = 1000;
  opts_.congestion.exit_horizon_ms = 1000;

  auto make_compatible_job = [this](int request_id) {
    return make_job(
        request_id,
        {torch::ones({1, 2}, torch::TensorOptions().dtype(torch::kFloat))},
        {at::kFloat});
  };

  int request_id = 750;
  for (int i = 0; i < 5; ++i) {
    auto first = make_compatible_job(request_id++);
    auto second = make_compatible_job(request_id++);
    ASSERT_TRUE(queue_.push(second));

    auto collected = starpu_server::StarPUTaskRunnerTestAdapter::collect_batch(
        runner_.get(), first);
    EXPECT_EQ(collected.size(), 2U);
  }
}

TEST_F(StarPUTaskRunnerFixture, CollectBatchTreatsInflightBacklogAsHighPressure)
{
  opts_.batching.dynamic_batching = true;
  opts_.batching.max_batch_size = 4;
  opts_.batching.batch_coalesce_timeout_ms = 0;
  opts_.batching.max_inflight_tasks = 4;
  opts_.congestion.enabled = true;
  opts_.congestion.tick_interval_ms = 2;
  opts_.congestion.entry_horizon_ms = 2;
  opts_.congestion.exit_horizon_ms = 2;

  runner_.reset();
  starpu_setup_.reset();
  starpu_setup_ = std::make_unique<starpu_server::StarPUSetup>(opts_);
  config_.starpu = starpu_setup_.get();
  runner_ = std::make_unique<starpu_server::StarPUTaskRunner>(config_);

  auto make_compatible_job = [this](int request_id) {
    return make_job(
        request_id,
        {torch::ones({1, 2}, torch::TensorOptions().dtype(torch::kFloat))},
        {at::kFloat});
  };

  int request_id = 820;
  std::size_t last_low_pressure_batch_size = 0;
  for (int i = 0; i < 3; ++i) {
    auto first = make_compatible_job(request_id++);
    auto second = make_compatible_job(request_id++);
    ASSERT_TRUE(queue_.push(second));
    auto collected = starpu_server::StarPUTaskRunnerTestAdapter::collect_batch(
        runner_.get(), first);
    last_low_pressure_batch_size = collected.size();
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
  }
  ASSERT_EQ(last_low_pressure_batch_size, 1U);

  auto pending_low_pressure_job = runner_->wait_for_next_job();
  ASSERT_NE(pending_low_pressure_job, nullptr);

  starpu_server::StarPUTaskRunnerTestAdapter::reserve_inflight_slot(
      runner_.get());
  starpu_server::StarPUTaskRunnerTestAdapter::reserve_inflight_slot(
      runner_.get());
  starpu_server::StarPUTaskRunnerTestAdapter::reserve_inflight_slot(
      runner_.get());

  std::this_thread::sleep_for(std::chrono::milliseconds(3));
  auto high_pressure_first = make_compatible_job(request_id++);
  auto high_pressure_second = make_compatible_job(request_id++);
  ASSERT_TRUE(queue_.push(high_pressure_second));
  auto high_pressure_collected =
      starpu_server::StarPUTaskRunnerTestAdapter::collect_batch(
          runner_.get(), high_pressure_first);
  EXPECT_GE(high_pressure_collected.size(), 2U);

  starpu_server::StarPUTaskRunnerTestAdapter::release_inflight_slot(
      runner_.get());
  starpu_server::StarPUTaskRunnerTestAdapter::release_inflight_slot(
      runner_.get());
  starpu_server::StarPUTaskRunnerTestAdapter::release_inflight_slot(
      runner_.get());
}

TEST_F(
    StarPUTaskRunnerFixture,
    CollectBatchUsesConfiguredMaximumWhenCongestionDetected)
{
  opts_.batching.dynamic_batching = true;
  opts_.batching.max_batch_size = 4;
  opts_.batching.batch_coalesce_timeout_ms = 0;
  opts_.congestion.enabled = true;
  opts_.congestion.tick_interval_ms = 20;
  opts_.congestion.entry_horizon_ms = 40;
  opts_.congestion.exit_horizon_ms = 40;

  starpu_server::congestion::shutdown();
  starpu_server::congestion::Config monitor_cfg;
  monitor_cfg.enabled = true;
  monitor_cfg.tick_interval = std::chrono::milliseconds(20);
  monitor_cfg.entry_horizon = std::chrono::milliseconds(40);
  monitor_cfg.exit_horizon = std::chrono::milliseconds(60);
  ScopedCongestionMonitor monitor(&queue_, monitor_cfg);
  ASSERT_TRUE(monitor.started);

  auto make_compatible_job = [this](int request_id) {
    return make_job(
        request_id,
        {torch::ones({1, 2}, torch::TensorOptions().dtype(torch::kFloat))},
        {at::kFloat});
  };

  int request_id = 800;
  for (int i = 0; i < 8; ++i) {
    auto first = make_compatible_job(request_id++);
    auto collected = starpu_server::StarPUTaskRunnerTestAdapter::collect_batch(
        runner_.get(), first);
    ASSERT_EQ(collected.size(), 1U);
  }

  starpu_server::congestion::record_arrival(1);
  starpu_server::congestion::record_rejection(1);
  ASSERT_TRUE(wait_until(
      [] { return starpu_server::congestion::is_congested(); },
      std::chrono::milliseconds(1000), std::chrono::milliseconds(5)));

  for (int i = 0; i < opts_.batching.max_batch_size - 1; ++i) {
    ASSERT_TRUE(queue_.push(make_compatible_job(request_id++)));
  }

  auto first = make_compatible_job(request_id++);
  auto collected = starpu_server::StarPUTaskRunnerTestAdapter::collect_batch(
      runner_.get(), first);

  EXPECT_EQ(
      collected.size(),
      static_cast<std::size_t>(opts_.batching.max_batch_size));
}

TEST_F(
    StarPUTaskRunnerFixture,
    CollectBatchWaitsToFillWhenCongestedAndCoalesceTimeoutIsZero)
{
  opts_.batching.dynamic_batching = true;
  opts_.batching.max_batch_size = 4;
  opts_.batching.batch_coalesce_timeout_ms = 0;
  opts_.congestion.enabled = true;
  opts_.congestion.tick_interval_ms = 80;
  opts_.congestion.entry_horizon_ms = 40;
  opts_.congestion.exit_horizon_ms = 40;

  starpu_server::congestion::shutdown();
  starpu_server::congestion::Config monitor_cfg;
  monitor_cfg.enabled = true;
  monitor_cfg.tick_interval = std::chrono::milliseconds(10);
  monitor_cfg.entry_horizon = std::chrono::milliseconds(20);
  monitor_cfg.exit_horizon = std::chrono::milliseconds(60);
  ScopedCongestionMonitor monitor(&queue_, monitor_cfg);
  ASSERT_TRUE(monitor.started);

  auto make_compatible_job = [this](int request_id) {
    return make_job(
        request_id,
        {torch::ones({1, 2}, torch::TensorOptions().dtype(torch::kFloat))},
        {at::kFloat});
  };

  starpu_server::congestion::record_arrival(1);
  starpu_server::congestion::record_rejection(1);
  ASSERT_TRUE(wait_until(
      [] { return starpu_server::congestion::is_congested(); },
      std::chrono::milliseconds(1000), std::chrono::milliseconds(5)));

  auto first = make_compatible_job(900);
  auto second = make_compatible_job(901);
  auto third = make_compatible_job(902);
  auto fourth = make_compatible_job(903);

  std::atomic<bool> pushed_all{true};
  std::jthread async_pusher([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    const bool all_ok =
        queue_.push(second) && queue_.push(third) && queue_.push(fourth);
    pushed_all.store(all_ok, std::memory_order_release);
  });

  auto collected = starpu_server::StarPUTaskRunnerTestAdapter::collect_batch(
      runner_.get(), first);

  EXPECT_TRUE(pushed_all.load(std::memory_order_acquire));
  EXPECT_EQ(
      collected.size(),
      static_cast<std::size_t>(opts_.batching.max_batch_size));
}

TEST_F(
    StarPUTaskRunnerFixture, CollectBatchWaitsForJobWhenQueueEmptyAndTimeoutSet)
{
  opts_.batching.dynamic_batching = true;
  opts_.batching.max_batch_size = 3;
  opts_.batching.batch_coalesce_timeout_ms = 100;

  auto first = make_job(
      503, {torch::ones({1, 2}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  auto second = make_job(
      504, {torch::ones({1, 2}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});

  std::atomic<bool> thread_started{false};
  std::jthread async_pusher([&] {
    thread_started.store(true, std::memory_order_release);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    [[maybe_unused]] bool pushed = queue_.push(second);
  });

  while (!thread_started.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  auto collected = starpu_server::StarPUTaskRunnerTestAdapter::collect_batch(
      runner_.get(), first);

  ASSERT_EQ(collected.size(), 2U);
  EXPECT_EQ(collected[0], first);
  EXPECT_EQ(collected[1], second);
}

TEST_F(StarPUTaskRunnerFixture, WaitForNextJobDoesNotBlockWhenNoInflightLimit)
{
  ASSERT_EQ(opts_.batching.max_inflight_tasks, 0U);

  auto job = make_job(
      601, {torch::ones({1, 2}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  ASSERT_TRUE(queue_.push(job));

  auto retrieved = runner_->wait_for_next_job();

  ASSERT_NE(retrieved, nullptr);
  EXPECT_EQ(retrieved, job);
}

TEST_F(StarPUTaskRunnerFixture, WaitForNextJobBlocksWhenInflightLimitReached)
{
  opts_.batching.max_inflight_tasks = 1;
  runner_.reset();
  starpu_setup_.reset();
  starpu_setup_ = std::make_unique<starpu_server::StarPUSetup>(opts_);
  config_.starpu = starpu_setup_.get();
  config_.opts = &opts_;
  runner_ = std::make_unique<starpu_server::StarPUTaskRunner>(config_);

  auto job1 = make_job(
      602, {torch::ones({1, 2}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  auto job2 = make_job(
      603, {torch::ones({1, 2}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});

  ASSERT_TRUE(queue_.push(job1));
  ASSERT_TRUE(queue_.push(job2));

  starpu_server::StarPUTaskRunnerTestAdapter::reserve_inflight_slot(
      runner_.get());
  ASSERT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      1U);

  std::atomic<bool> thread_started{false};
  std::atomic<bool> thread_completed{false};
  std::shared_ptr<starpu_server::InferenceJob> retrieved_job;

  std::jthread waiter([&] {
    thread_started.store(true, std::memory_order_release);
    retrieved_job = runner_->wait_for_next_job();
    thread_completed.store(true, std::memory_order_release);
  });

  while (!thread_started.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_FALSE(thread_completed.load(std::memory_order_acquire));

  starpu_server::StarPUTaskRunnerTestAdapter::release_inflight_slot(
      runner_.get());

  waiter.join();
  EXPECT_TRUE(thread_completed.load(std::memory_order_acquire));
  ASSERT_NE(retrieved_job, nullptr);
  EXPECT_EQ(retrieved_job, job1);
}

TEST_F(
    StarPUTaskRunnerFixture,
    WaitForNextJobUnblocksOnQueueShutdownWhenInflightLimitReached)
{
  opts_.batching.max_inflight_tasks = 1;
  runner_.reset();
  starpu_setup_.reset();
  starpu_setup_ = std::make_unique<starpu_server::StarPUSetup>(opts_);
  config_.starpu = starpu_setup_.get();
  config_.opts = &opts_;
  runner_ = std::make_unique<starpu_server::StarPUTaskRunner>(config_);

  starpu_server::StarPUTaskRunnerTestAdapter::reserve_inflight_slot(
      runner_.get());
  ASSERT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      1U);

  std::atomic<bool> thread_started{false};
  std::atomic<bool> thread_completed{false};
  std::shared_ptr<starpu_server::InferenceJob> retrieved_job;

  std::jthread waiter([&] {
    thread_started.store(true, std::memory_order_release);
    retrieved_job = runner_->wait_for_next_job();
    thread_completed.store(true, std::memory_order_release);
  });

  while (!thread_started.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  EXPECT_FALSE(thread_completed.load(std::memory_order_acquire));

  queue_.shutdown();

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(300);
  bool forced_release = false;
  while (!thread_completed.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  if (!thread_completed.load(std::memory_order_acquire)) {
    forced_release = true;
    starpu_server::StarPUTaskRunnerTestAdapter::release_inflight_slot(
        runner_.get());
  }

  waiter.join();
  EXPECT_FALSE(forced_release);
  EXPECT_TRUE(thread_completed.load(std::memory_order_acquire));
  EXPECT_EQ(retrieved_job, nullptr);
}

TEST_F(
    StarPUTaskRunnerFixture, WaitForNextJobReturnsImmediatelyWhenSlotsAvailable)
{
  opts_.batching.max_inflight_tasks = 5;
  runner_.reset();
  starpu_setup_.reset();
  starpu_setup_ = std::make_unique<starpu_server::StarPUSetup>(opts_);
  config_.starpu = starpu_setup_.get();
  config_.opts = &opts_;
  runner_ = std::make_unique<starpu_server::StarPUTaskRunner>(config_);

  auto job = make_job(
      604, {torch::ones({1, 2}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  ASSERT_TRUE(queue_.push(job));

  starpu_server::StarPUTaskRunnerTestAdapter::reserve_inflight_slot(
      runner_.get());
  starpu_server::StarPUTaskRunnerTestAdapter::reserve_inflight_slot(
      runner_.get());
  ASSERT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      2U);

  auto retrieved = runner_->wait_for_next_job();

  ASSERT_NE(retrieved, nullptr);
  EXPECT_EQ(retrieved, job);
}

TEST_F(StarPUTaskRunnerFixture, WaitForNextJobReturnsPendingJobFirst)
{
  opts_.batching.max_inflight_tasks = 5;
  runner_.reset();
  starpu_setup_.reset();
  starpu_setup_ = std::make_unique<starpu_server::StarPUSetup>(opts_);
  config_.starpu = starpu_setup_.get();
  config_.opts = &opts_;
  runner_ = std::make_unique<starpu_server::StarPUTaskRunner>(config_);

  auto pending_job = make_job(
      605, {torch::ones({1, 2}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  auto queued_job = make_job(
      606, {torch::ones({1, 2}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});

  starpu_server::StarPUTaskRunnerTestAdapter::set_batch_collector_pending_job(
      runner_.get(), pending_job);
  ASSERT_TRUE(queue_.push(queued_job));

  auto retrieved = runner_->wait_for_next_job();

  ASSERT_NE(retrieved, nullptr);
  EXPECT_EQ(retrieved, pending_job);
  EXPECT_EQ(queue_.size(), 1U);
}

TEST_F(
    StarPUTaskRunnerFixture,
    WaitForPreparedJobReturnsNullWhenSyncPrimitivesMissing)
{
  starpu_server::StarPUTaskRunnerTestAdapter::disable_prepared_job_sync(
      runner_.get());

  auto job = starpu_server::StarPUTaskRunnerTestAdapter::wait_for_prepared_job(
      runner_.get());
  EXPECT_EQ(job, nullptr);
}

TEST_F(
    StarPUTaskRunnerFixture, EnqueuePreparedJobNoopsWhenSyncPrimitivesMissing)
{
  starpu_server::StarPUTaskRunnerTestAdapter::disable_prepared_job_sync(
      runner_.get());

  auto job = make_job(901, {});
  starpu_server::StarPUTaskRunnerTestAdapter::enqueue_prepared_job(
      runner_.get(), job);

  auto dequeued =
      starpu_server::StarPUTaskRunnerTestAdapter::wait_for_prepared_job(
          runner_.get());
  EXPECT_EQ(dequeued, nullptr);
}

TEST_F(
    StarPUTaskRunnerFixture, ProcessPreparedJobReturnsImmediatelyWhenJobIsNull)
{
  std::atomic<int> submit_hook_calls{0};
  starpu_server::StarPUTaskRunnerTestAdapter::set_submit_hook(
      [&]() { submit_hook_calls.fetch_add(1, std::memory_order_acq_rel); });

  std::shared_ptr<starpu_server::InferenceJob> missing_job;
  starpu_server::StarPUTaskRunnerTestAdapter::process_prepared_job(
      runner_.get(), missing_job);

  starpu_server::StarPUTaskRunnerTestAdapter::reset_submit_hook();

  EXPECT_EQ(submit_hook_calls.load(std::memory_order_acquire), 0);
  EXPECT_EQ(completed_jobs_.load(std::memory_order_acquire), 0U);
}

TEST_F(StarPUTaskRunnerFixture, BatchingLoopStopsWhenNoJobAvailable)
{
  starpu_server::StarPUTaskRunnerTestAdapter::set_batch_collector_queue_to_null(
      runner_.get());

  starpu_server::StarPUTaskRunnerTestAdapter::run_batching_loop(runner_.get());

  EXPECT_TRUE(
      starpu_server::StarPUTaskRunnerTestAdapter::batching_done(runner_.get()));
  auto dequeued =
      starpu_server::StarPUTaskRunnerTestAdapter::wait_for_prepared_job(
          runner_.get());
  EXPECT_EQ(dequeued, nullptr);
}

TEST_F(
    StarPUTaskRunnerFixture,
    BatchCollectorIsBatchingDoneReturnsFalseWhenFlagPointerMissing)
{
  starpu_server::StarPUTaskRunnerTestAdapter::
      set_batch_collector_batching_done_ptr(runner_.get(), nullptr);

  EXPECT_FALSE(starpu_server::StarPUTaskRunnerTestAdapter::
                   is_batch_collector_batching_done(runner_.get()));
}

TEST_F(
    StarPUTaskRunnerFixture,
    BatchCollectorIsBatchingDoneUsesFlagWhenPreparedMutexMissing)
{
  starpu_server::StarPUTaskRunnerTestAdapter::disable_prepared_job_sync(
      runner_.get());
  starpu_server::StarPUTaskRunnerTestAdapter::
      set_batch_collector_batching_done_value(runner_.get(), true);

  EXPECT_TRUE(starpu_server::StarPUTaskRunnerTestAdapter::
                  is_batch_collector_batching_done(runner_.get()));

  starpu_server::StarPUTaskRunnerTestAdapter::
      set_batch_collector_batching_done_value(runner_.get(), false);

  EXPECT_FALSE(starpu_server::StarPUTaskRunnerTestAdapter::
                   is_batch_collector_batching_done(runner_.get()));
}

TEST_F(
    StarPUTaskRunnerFixture, ShouldAbortInflightWaitReturnsTrueWhenBatchingDone)
{
  starpu_server::StarPUTaskRunnerTestAdapter::
      set_batch_collector_batching_done_value(runner_.get(), true);

  EXPECT_TRUE(starpu_server::StarPUTaskRunnerTestAdapter::
                  should_abort_batch_collector_inflight_wait(runner_.get()));
}

TEST_F(
    StarPUTaskRunnerFixture, ShouldAbortInflightWaitReturnsTrueWhenQueueMissing)
{
  starpu_server::StarPUTaskRunnerTestAdapter::
      set_batch_collector_batching_done_value(runner_.get(), false);
  starpu_server::StarPUTaskRunnerTestAdapter::set_batch_collector_queue_to_null(
      runner_.get());

  EXPECT_TRUE(starpu_server::StarPUTaskRunnerTestAdapter::
                  should_abort_batch_collector_inflight_wait(runner_.get()));
}

TEST_F(
    StarPUTaskRunnerFixture,
    ShouldAbortInflightWaitReturnsFalseWhenPendingJobIsAvailable)
{
  starpu_server::StarPUTaskRunnerTestAdapter::
      set_batch_collector_batching_done_value(runner_.get(), false);
  auto pending_job = make_job(
      903, {torch::ones({1, 2}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  starpu_server::StarPUTaskRunnerTestAdapter::set_batch_collector_pending_job(
      runner_.get(), pending_job);

  EXPECT_FALSE(starpu_server::StarPUTaskRunnerTestAdapter::
                   should_abort_batch_collector_inflight_wait(runner_.get()));
}

TEST_F(StarPUTaskRunnerFixture, BatchingLoopContinuesWhenBuiltJobBecomesNull)
{
  auto job = make_job(
      902, {torch::ones({1, 2}, torch::TensorOptions().dtype(torch::kFloat))},
      {at::kFloat});
  ASSERT_TRUE(queue_.push(job));
  queue_.shutdown();

  std::atomic<int> hook_calls{0};
  BatchCollectorAfterBuildJobHookGuard hook_guard{
      [&hook_calls](std::shared_ptr<starpu_server::InferenceJob>& built_job) {
        hook_calls.fetch_add(1, std::memory_order_acq_rel);
        built_job.reset();
      }};

  starpu_server::StarPUTaskRunnerTestAdapter::run_batching_loop(runner_.get());

  EXPECT_EQ(hook_calls.load(std::memory_order_acquire), 1);
  EXPECT_TRUE(
      starpu_server::StarPUTaskRunnerTestAdapter::batching_done(runner_.get()));
  auto dequeued =
      starpu_server::StarPUTaskRunnerTestAdapter::wait_for_prepared_job(
          runner_.get());
  EXPECT_EQ(dequeued, nullptr);
}

TEST_F(StarPUTaskRunnerFixture, TryAcquireNextJobReturnsNullWhenDeadlinePassed)
{
  const auto deadline = starpu_server::task_runner_internal::Clock::now() -
                        std::chrono::milliseconds(1);

  auto job = starpu_server::StarPUTaskRunnerTestAdapter::try_acquire_next_job(
      runner_.get(), true, deadline);

  EXPECT_EQ(job, nullptr);
}

TEST_F(
    StarPUTaskRunnerFixture, RunRethrowsStdExceptionEscapedFromBatchingThread)
{
  auto probe = starpu_server::make_callback_probe();
  auto job = probe.job;
  job->set_input_tensors(
      {torch::ones({1}, torch::TensorOptions().dtype(torch::kFloat))});
  job->set_input_types({at::kFloat});

  ASSERT_TRUE(queue_.push(job));
  queue_.shutdown();

  BatchCollectorAfterBuildJobHookGuard hook_guard{
      [](std::shared_ptr<starpu_server::InferenceJob>&) {
        throw std::runtime_error("batching thread failure");
      }};

  try {
    runner_->run();
    FAIL() << "Expected run() to rethrow batching thread failure.";
  }
  catch (const std::runtime_error& error) {
    EXPECT_NE(
        std::string(error.what())
            .find("Unhandled exception escaped 'starpu-batching' thread: "
                  "batching thread failure"),
        std::string::npos);
  }
  catch (...) {
    FAIL() << "Expected std::runtime_error.";
  }

  EXPECT_EQ(completed_jobs_.load(std::memory_order_acquire), 0U);
}

TEST_F(
    StarPUTaskRunnerFixture,
    RunKeepsFirstThreadExceptionWhenDuplicateCaptureIsRequestedForTest)
{
  auto probe = starpu_server::make_callback_probe();
  auto job = probe.job;
  job->set_input_tensors(
      {torch::ones({1}, torch::TensorOptions().dtype(torch::kFloat))});
  job->set_input_types({at::kFloat});

  ASSERT_TRUE(queue_.push(job));
  queue_.shutdown();

  DuplicateBatchingThreadExceptionCaptureGuard duplicate_capture_guard;
  BatchCollectorAfterBuildJobHookGuard hook_guard{
      [](std::shared_ptr<starpu_server::InferenceJob>&) {
        throw std::runtime_error("primary batching thread failure");
      }};

  try {
    runner_->run();
    FAIL() << "Expected run() to rethrow batching thread failure.";
  }
  catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(
        message.find("primary batching thread failure"), std::string::npos);
    EXPECT_EQ(
        message.find("secondary batching thread failure"), std::string::npos);
  }
  catch (...) {
    FAIL() << "Expected std::runtime_error.";
  }

  EXPECT_EQ(completed_jobs_.load(std::memory_order_acquire), 0U);
}

TEST_F(
    StarPUTaskRunnerFixture,
    RunJoinsBatchingThreadWhenExceptionRaisedAfterThreadStart)
{
  RunAfterBatchingThreadStartHookGuard hook_guard{
      [] { throw std::runtime_error("post start failure"); }};

  try {
    runner_->run();
    FAIL() << "Expected run() to rethrow post-start failure.";
  }
  catch (const std::runtime_error& error) {
    EXPECT_EQ(std::string(error.what()), "post start failure");
  }
  catch (...) {
    FAIL() << "Expected std::runtime_error.";
  }

  EXPECT_EQ(completed_jobs_.load(std::memory_order_acquire), 0U);
}

TEST_F(StarPUTaskRunnerFixture, RunFinalizesJobAfterExceptionRaisedBeforeSubmit)
{
  auto probe = starpu_server::make_callback_probe();
  auto job = probe.job;
  job->set_input_tensors(
      {torch::ones({1}, torch::TensorOptions().dtype(torch::kFloat))});
  job->set_input_types({at::kFloat});

  ASSERT_TRUE(queue_.push(job));
  queue_.shutdown();

  RunBeforeSubmitHookGuard hook_guard{
      [] { throw std::runtime_error("run loop injected failure"); }};

  runner_->run();

  EXPECT_TRUE(probe.called);
  EXPECT_TRUE(probe.results.empty());
  EXPECT_EQ(probe.latency, -1);
  EXPECT_EQ(completed_jobs_.load(std::memory_order_acquire), 1U);

  const auto failure = job->completion().failure_info();
  ASSERT_TRUE(failure.has_value());
  EXPECT_EQ(failure->stage, "execution");
  EXPECT_EQ(failure->reason, "runtime_error");
  EXPECT_TRUE(failure->metrics_reported);
  EXPECT_EQ(
      failure->message,
      "Unexpected exception while processing dequeued job: run loop injected "
      "failure");
}

TEST_F(
    StarPUTaskRunnerFixture,
    RunFinalizesJobAfterUnknownExceptionRaisedBeforeSubmit)
{
  auto probe = starpu_server::make_callback_probe();
  auto job = probe.job;
  job->set_input_tensors(
      {torch::ones({1}, torch::TensorOptions().dtype(torch::kFloat))});
  job->set_input_types({at::kFloat});

  ASSERT_TRUE(queue_.push(job));
  queue_.shutdown();

  RunBeforeSubmitHookGuard hook_guard{[] { throw 123; }};

  runner_->run();

  EXPECT_TRUE(probe.called);
  EXPECT_TRUE(probe.results.empty());
  EXPECT_EQ(probe.latency, -1);
  EXPECT_EQ(completed_jobs_.load(std::memory_order_acquire), 1U);

  const auto failure = job->completion().failure_info();
  ASSERT_TRUE(failure.has_value());
  EXPECT_EQ(failure->stage, "execution");
  EXPECT_EQ(failure->reason, "exception");
  EXPECT_TRUE(failure->metrics_reported);
  EXPECT_EQ(
      failure->message,
      "Unexpected non-standard exception while processing dequeued job: "
      "Unknown non-standard exception");
}

TEST_F(
    StarPUTaskRunnerFixture,
    RunRethrowsNonStandardExceptionEscapedFromBatchingThread)
{
  auto probe = starpu_server::make_callback_probe();
  auto job = probe.job;
  job->set_input_tensors(
      {torch::ones({1}, torch::TensorOptions().dtype(torch::kFloat))});
  job->set_input_types({at::kFloat});

  ASSERT_TRUE(queue_.push(job));
  queue_.shutdown();

  BatchCollectorAfterBuildJobHookGuard hook_guard{
      [](std::shared_ptr<starpu_server::InferenceJob>&) { throw 7; }};

  try {
    runner_->run();
    FAIL() << "Expected run() to rethrow batching thread failure.";
  }
  catch (const std::runtime_error& error) {
    EXPECT_NE(
        std::string(error.what())
            .find("Unhandled non-standard exception escaped "
                  "'starpu-batching' thread."),
        std::string::npos);
  }
  catch (...) {
    FAIL() << "Expected std::runtime_error.";
  }

  EXPECT_EQ(completed_jobs_.load(std::memory_order_acquire), 0U);
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
  queue_.shutdown();

  starpu_server::StarPUTaskRunnerTestAdapter::set_submit_hook([&]() {
    starpu_server::StarPUTaskRunnerTestAdapter::reset_submit_hook();
    throw starpu_server::InferenceEngineException("test inference failure");
  });

  runner_->run();
  starpu_server::StarPUTaskRunnerTestAdapter::reset_submit_hook();

  EXPECT_TRUE(probe.called);
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
  queue_.shutdown();

  starpu_server::StarPUTaskRunnerTestAdapter::set_submit_hook([&]() {
    starpu_server::StarPUTaskRunnerTestAdapter::reset_submit_hook();
    throw std::runtime_error("runtime failure");
  });

  runner_->run();
  starpu_server::StarPUTaskRunnerTestAdapter::reset_submit_hook();

  EXPECT_TRUE(probe.called);
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
  queue_.shutdown();

  starpu_server::StarPUTaskRunnerTestAdapter::set_submit_hook([&]() {
    starpu_server::StarPUTaskRunnerTestAdapter::reset_submit_hook();
    throw std::logic_error("logic failure");
  });

  runner_->run();
  starpu_server::StarPUTaskRunnerTestAdapter::reset_submit_hook();

  EXPECT_TRUE(probe.called);
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
  queue_.shutdown();

  starpu_server::StarPUTaskRunnerTestAdapter::set_submit_hook([&]() {
    starpu_server::StarPUTaskRunnerTestAdapter::reset_submit_hook();
    throw std::bad_alloc();
  });

  runner_->run();
  starpu_server::StarPUTaskRunnerTestAdapter::reset_submit_hook();

  EXPECT_TRUE(probe.called);
  EXPECT_EQ(completed_jobs_.load(), 1);
  EXPECT_EQ(queue_.size(), 0U);
}

TEST_F(StarPUTaskRunnerFixture, RunCatchesGenericStdException)
{
  opts_.batching.dynamic_batching = false;

  auto probe = starpu_server::make_callback_probe();
  auto job = probe.job;
  job->set_input_tensors(
      {torch::ones({1}, torch::TensorOptions().dtype(torch::kFloat))});
  job->set_input_types({at::kFloat});

  ASSERT_TRUE(queue_.push(job));
  queue_.shutdown();

  struct CustomStdException final : std::exception {
    [[nodiscard]] auto what() const noexcept -> const char* override
    {
      return "custom std exception";
    }
  };

  starpu_server::StarPUTaskRunnerTestAdapter::set_submit_hook([&]() {
    starpu_server::StarPUTaskRunnerTestAdapter::reset_submit_hook();
    throw CustomStdException{};
  });

  runner_->run();
  starpu_server::StarPUTaskRunnerTestAdapter::reset_submit_hook();

  EXPECT_TRUE(probe.called);
  EXPECT_EQ(completed_jobs_.load(), 1);
  EXPECT_EQ(queue_.size(), 0U);

  const auto failure = job->completion().failure_info();
  ASSERT_TRUE(failure.has_value());
  EXPECT_EQ(
      failure->message, "Unexpected std::exception: custom std exception");
}

TEST_F(StarPUTaskRunnerFixture, RunCatchesNonStandardException)
{
  opts_.batching.dynamic_batching = false;

  auto probe = starpu_server::make_callback_probe();
  auto job = probe.job;
  job->set_input_tensors(
      {torch::ones({1}, torch::TensorOptions().dtype(torch::kFloat))});
  job->set_input_types({at::kFloat});

  ASSERT_TRUE(queue_.push(job));
  queue_.shutdown();

  starpu_server::StarPUTaskRunnerTestAdapter::set_submit_hook([&]() {
    starpu_server::StarPUTaskRunnerTestAdapter::reset_submit_hook();
    throw 42;
  });

  runner_->run();
  starpu_server::StarPUTaskRunnerTestAdapter::reset_submit_hook();

  EXPECT_TRUE(probe.called);
  EXPECT_EQ(completed_jobs_.load(), 1);
  EXPECT_EQ(queue_.size(), 0U);

  const auto failure = job->completion().failure_info();
  ASSERT_TRUE(failure.has_value());
  EXPECT_EQ(
      failure->message,
      "Unexpected non-standard exception: Unknown non-standard exception");
}

TEST_F(
    StarPUTaskRunnerFixture,
    SubmitJobOrHandleFailureHandlesUnknownExceptionCategory)
{
  auto probe = starpu_server::make_callback_probe();
  auto job = probe.job;
  job->set_request_id(4242);

  starpu_server::StarPUTaskRunnerTestAdapter::set_submit_hook([&]() {
    starpu_server::StarPUTaskRunnerTestAdapter::reset_submit_hook();
    throw 42;
  });

  starpu_server::StarPUTaskRunnerTestAdapter::submit_job_or_handle_failure(
      runner_.get(), job, /*submission_id=*/4242, /*job_id=*/4242);
  starpu_server::StarPUTaskRunnerTestAdapter::reset_submit_hook();

  assert_failure_result(probe);
  const auto failure = job->completion().failure_info();
  ASSERT_TRUE(failure.has_value());
  EXPECT_EQ(failure->stage, "execution");
  EXPECT_EQ(failure->reason, "exception");
  EXPECT_TRUE(failure->metrics_reported);
  EXPECT_EQ(
      failure->message,
      "Unexpected non-standard exception: Unknown non-standard exception");
}

TEST_F(StarPUTaskRunnerFixture, RunDrainsQueueWhenStarpuSubmitAlwaysFails)
{
  opts_.batching.dynamic_batching = false;

  auto model_config = make_model_config(
      "submit_fail_model", {make_tensor_config("input0", {1}, at::kFloat)},
      {make_tensor_config("output0", {1}, at::kFloat)});
  reset_runner_with_model(model_config, /*pool_size=*/1);

  starpu_test::ScopedStarpuDataAcquireOverride acquire_override(
      &NoOpStarpuDataAcquire);
  starpu_test::ScopedStarpuDataReleaseOverride release_override(
      &NoOpStarpuDataRelease);
  submit_override_calls.store(0, std::memory_order_relaxed);
  starpu_test::ScopedStarpuTaskSubmitOverride submit_override(
      &AlwaysFailStarpuSubmit);

  auto configure_job = [](starpu_server::CallbackProbe& probe, int request_id) {
    probe.job->set_request_id(request_id);
    probe.job->completion().set_model_name("submit_fail_model");
    probe.job->set_input_tensors(
        {torch::ones({1}, torch::TensorOptions().dtype(torch::kFloat))});
    probe.job->set_input_types({at::kFloat});
    probe.job->set_output_tensors(
        {torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat))});
  };

  auto first_probe = starpu_server::make_callback_probe();
  auto second_probe = starpu_server::make_callback_probe();
  configure_job(first_probe, 1001);
  configure_job(second_probe, 1002);

  ASSERT_TRUE(queue_.push(first_probe.job));
  ASSERT_TRUE(queue_.push(second_probe.job));
  queue_.shutdown();

  runner_->run();

  EXPECT_TRUE(first_probe.called);
  EXPECT_TRUE(second_probe.called);
  EXPECT_TRUE(first_probe.results.empty());
  EXPECT_TRUE(second_probe.results.empty());
  EXPECT_EQ(first_probe.latency, -1);
  EXPECT_EQ(second_probe.latency, -1);
  EXPECT_EQ(completed_jobs_.load(), 2U);
  EXPECT_EQ(queue_.size(), 0U);
  EXPECT_EQ(submit_override_calls.load(std::memory_order_relaxed), 2);
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
  queue_.shutdown();

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

TEST_F(StarPUTaskRunnerFixture, RunClearsOnCompleteWhenJobCancelled)
{
  opts_.batching.dynamic_batching = false;

  bool callback_invoked = false;
  auto job = make_job(1, {torch::tensor({1.0F})});
  job->completion().set_on_complete(
      [&callback_invoked](const std::vector<torch::Tensor>&, double) {
        callback_invoked = true;
      });

  auto cancel_flag = std::make_shared<std::atomic<bool>>(true);
  job->set_cancelled_flag(cancel_flag);

  auto holder = std::make_shared<int>(5);
  job->set_input_memory_holders(
      {std::shared_ptr<const void>(holder, holder.get())});
  job->set_output_tensors({torch::tensor({2.0F})});

  ASSERT_TRUE(queue_.push(job));
  queue_.shutdown();

  runner_->run();

  EXPECT_FALSE(job->completion().has_on_complete());
  EXPECT_FALSE(callback_invoked);
  EXPECT_TRUE(job->get_input_tensors().empty());
  EXPECT_TRUE(job->get_input_memory_holders().empty());
  EXPECT_TRUE(job->get_output_tensors().empty());
  EXPECT_EQ(completed_jobs_.load(), 1);
}

TEST_F(StarPUTaskRunnerFixture, RunReleasesPendingSubJobsWhenJobCancelled)
{
  opts_.batching.dynamic_batching = false;

  auto job = make_job(1, {torch::tensor({1.0F})});
  auto cancel_flag = std::make_shared<std::atomic<bool>>(true);
  job->set_cancelled_flag(cancel_flag);

  auto pending_a = make_job(2, {torch::tensor({2.0F})});
  auto pending_b = make_job(3, {torch::tensor({3.0F})});

  auto holder_a = std::make_shared<int>(7);
  pending_a->set_input_memory_holders(
      {std::shared_ptr<const void>(holder_a, holder_a.get())});
  auto holder_b = std::make_shared<int>(9);
  pending_b->set_input_memory_holders(
      {std::shared_ptr<const void>(holder_b, holder_b.get())});

  std::vector<std::shared_ptr<starpu_server::InferenceJob>> pending_jobs;
  pending_jobs.push_back(pending_a);
  pending_jobs.push_back(pending_b);
  job->batch().set_pending_sub_jobs(std::move(pending_jobs));

  ASSERT_TRUE(job->batch().has_pending_sub_jobs());
  ASSERT_TRUE(queue_.push(job));
  queue_.shutdown();

  runner_->run();

  EXPECT_FALSE(job->batch().has_pending_sub_jobs());
  EXPECT_TRUE(pending_a->get_input_tensors().empty());
  EXPECT_TRUE(pending_a->get_input_memory_holders().empty());
  EXPECT_TRUE(pending_b->get_input_tensors().empty());
  EXPECT_TRUE(pending_b->get_input_memory_holders().empty());
  EXPECT_EQ(completed_jobs_.load(), 1);
}

TEST_F(
    StarPUTaskRunnerFixture, HandleCancelledJobReturnsImmediatelyWhenJobMissing)
{
  opts_.batching.max_inflight_tasks = 1;
  reset_runner_with_model(
      make_model_config(
          "cancelled_missing_job",
          {make_tensor_config("input0", {1}, at::kFloat)},
          {make_tensor_config("output0", {1}, at::kFloat)}),
      /*pool_size=*/1);

  completed_jobs_.store(0, std::memory_order_release);
  starpu_server::StarPUTaskRunnerTestAdapter::reserve_inflight_slot(
      runner_.get());
  ASSERT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      1U);

  std::shared_ptr<starpu_server::InferenceJob> missing_job;
  starpu_server::StarPUTaskRunnerTestAdapter::handle_cancelled_job(
      runner_.get(), missing_job);

  EXPECT_EQ(completed_jobs_.load(std::memory_order_acquire), 0U);
  EXPECT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      1U);
}

TEST_F(
    StarPUTaskRunnerFixture,
    HandleCancelledJobReturnsWhenTerminalAlreadyHandled)
{
  opts_.batching.max_inflight_tasks = 1;
  reset_runner_with_model(
      make_model_config(
          "cancelled_terminal_handled",
          {make_tensor_config("input0", {1}, at::kFloat)},
          {make_tensor_config("output0", {1}, at::kFloat)}),
      /*pool_size=*/1);

  auto job = make_job(901, {torch::tensor({1.0F})});
  ASSERT_TRUE(job->completion().try_mark_terminal_handled());

  completed_jobs_.store(0, std::memory_order_release);
  starpu_server::StarPUTaskRunnerTestAdapter::reserve_inflight_slot(
      runner_.get());
  ASSERT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      1U);

  starpu_server::StarPUTaskRunnerTestAdapter::handle_cancelled_job(
      runner_.get(), job);

  EXPECT_EQ(completed_jobs_.load(std::memory_order_acquire), 0U);
  EXPECT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      1U);
}

TEST_F(
    StarPUTaskRunnerFixture,
    HandleCancelledJobReleasesInflightSlotWhenLimitIsConfigured)
{
  opts_.batching.max_inflight_tasks = 1;
  reset_runner_with_model(
      make_model_config(
          "cancelled_release_inflight",
          {make_tensor_config("input0", {1}, at::kFloat)},
          {make_tensor_config("output0", {1}, at::kFloat)}),
      /*pool_size=*/1);

  auto job = make_job(902, {torch::tensor({1.0F})});
  completed_jobs_.store(0, std::memory_order_release);

  starpu_server::StarPUTaskRunnerTestAdapter::reserve_inflight_slot(
      runner_.get());
  ASSERT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      1U);

  starpu_server::StarPUTaskRunnerTestAdapter::handle_cancelled_job(
      runner_.get(), job);

  EXPECT_EQ(completed_jobs_.load(std::memory_order_acquire), 1U);
  EXPECT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      0U);
}

TEST(StarPUTaskRunnerTestAdapter, ShouldHoldJobReturnsFalseWhenCandidateMissing)
{
  auto reference = std::make_shared<starpu_server::InferenceJob>();
  auto empty_candidate = std::shared_ptr<starpu_server::InferenceJob>();

  EXPECT_FALSE(starpu_server::StarPUTaskRunnerTestAdapter::should_hold_job(
      empty_candidate, reference, std::nullopt));
}

TEST(StarPUTaskRunnerTestAdapter, ShouldHoldJobReturnsTrueForAggregatedSubJobs)
{
  auto reference = std::make_shared<starpu_server::InferenceJob>();
  auto candidate = std::make_shared<starpu_server::InferenceJob>();
  auto aggregated = std::make_shared<starpu_server::InferenceJob>();
  starpu_server::InferenceJob::AggregatedSubJob sub{};
  sub.job = aggregated;
  sub.batch_size = 1;
  sub.request_id = 42;
  candidate->batch().set_aggregated_sub_jobs({sub});

  EXPECT_TRUE(starpu_server::StarPUTaskRunnerTestAdapter::should_hold_job(
      candidate, reference, std::nullopt));
}

TEST(StarPUTaskRunnerTestAdapter, ShouldHoldJobReturnsTrueForWorkerMismatch)
{
  auto reference = std::make_shared<starpu_server::InferenceJob>();
  auto candidate = std::make_shared<starpu_server::InferenceJob>();
  candidate->set_fixed_worker_id(7);

  const std::optional<int> target_worker = 3;
  EXPECT_TRUE(starpu_server::StarPUTaskRunnerTestAdapter::should_hold_job(
      candidate, reference, target_worker));
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

TEST(StarPUTaskRunnerTestAdapter, CanMergeJobsRejectsNullJobs)
{
  auto job = std::make_shared<starpu_server::InferenceJob>();
  job->set_input_tensors(
      {torch::ones({1, 1}, torch::TensorOptions().dtype(torch::kFloat))});
  job->set_input_types({at::kFloat});

  EXPECT_FALSE(
      starpu_server::StarPUTaskRunnerTestAdapter::can_merge_jobs(nullptr, job));
  EXPECT_FALSE(
      starpu_server::StarPUTaskRunnerTestAdapter::can_merge_jobs(job, nullptr));
}

TEST(StarPUTaskRunnerTestAdapter, CanMergeJobsRejectsDifferentInputCounts)
{
  auto lhs = std::make_shared<starpu_server::InferenceJob>();
  lhs->set_input_tensors(
      {torch::ones({1, 1}, torch::TensorOptions().dtype(torch::kFloat)),
       torch::ones({1, 1}, torch::TensorOptions().dtype(torch::kFloat))});
  lhs->set_input_types({at::kFloat, at::kFloat});

  auto rhs = std::make_shared<starpu_server::InferenceJob>();
  rhs->set_input_tensors(
      {torch::ones({1, 1}, torch::TensorOptions().dtype(torch::kFloat))});
  rhs->set_input_types({at::kFloat});

  EXPECT_FALSE(
      starpu_server::StarPUTaskRunnerTestAdapter::can_merge_jobs(lhs, rhs));
}

TEST(StarPUTaskRunnerTestAdapter, CanMergeJobsRejectsDifferentTypeCounts)
{
  auto lhs = std::make_shared<starpu_server::InferenceJob>();
  lhs->set_input_tensors(
      {torch::ones({1, 1}, torch::TensorOptions().dtype(torch::kFloat))});
  lhs->set_input_types({at::kFloat});

  auto rhs = std::make_shared<starpu_server::InferenceJob>();
  rhs->set_input_tensors(
      {torch::ones({1, 1}, torch::TensorOptions().dtype(torch::kFloat))});
  rhs->set_input_types({});

  EXPECT_FALSE(
      starpu_server::StarPUTaskRunnerTestAdapter::can_merge_jobs(lhs, rhs));
}

TEST(StarPUTaskRunnerTestAdapter, CanMergeJobsRejectsUndefinedTensors)
{
  auto make_job = [] {
    auto job = std::make_shared<starpu_server::InferenceJob>();
    job->set_input_tensors(
        {torch::ones({1, 1}, torch::TensorOptions().dtype(torch::kFloat))});
    job->set_input_types({at::kFloat});
    return job;
  };

  auto lhs = make_job();
  auto rhs = make_job();

  auto& lhs_inputs =
      const_cast<std::vector<torch::Tensor>&>(lhs->get_input_tensors());
  lhs_inputs[0] = torch::Tensor();

  EXPECT_FALSE(
      starpu_server::StarPUTaskRunnerTestAdapter::can_merge_jobs(lhs, rhs));
}

TEST(StarPUTaskRunnerTestAdapter, CanMergeJobsRejectsRankMismatch)
{
  auto lhs = std::make_shared<starpu_server::InferenceJob>();
  lhs->set_input_tensors(
      {torch::ones({2, 3}, torch::TensorOptions().dtype(torch::kFloat))});
  lhs->set_input_types({at::kFloat});

  auto rhs = std::make_shared<starpu_server::InferenceJob>();
  rhs->set_input_tensors(
      {torch::ones({2}, torch::TensorOptions().dtype(torch::kFloat))});
  rhs->set_input_types({at::kFloat});

  EXPECT_FALSE(
      starpu_server::StarPUTaskRunnerTestAdapter::can_merge_jobs(lhs, rhs));
}

TEST(StarPUTaskRunnerTestAdapter, CanMergeJobsRejectsNonPositiveRankTensors)
{
  auto make_job = [] {
    auto job = std::make_shared<starpu_server::InferenceJob>();
    job->set_input_tensors(
        {torch::tensor(1.0F, torch::TensorOptions().dtype(torch::kFloat))});
    job->set_input_types({at::kFloat});
    return job;
  };

  auto lhs = make_job();
  auto rhs = make_job();

  EXPECT_FALSE(
      starpu_server::StarPUTaskRunnerTestAdapter::can_merge_jobs(lhs, rhs));
}

TEST_F(
    StarPUTaskRunnerFixture, ReserveInflightSlotReturnsImmediatelyWhenNoLimit)
{
  ASSERT_FALSE(starpu_server::StarPUTaskRunnerTestAdapter::has_inflight_limit(
      runner_.get()));
  ASSERT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_max_inflight_tasks(
          runner_.get()),
      0U);
  ASSERT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      0U);

  EXPECT_NO_THROW(
      starpu_server::StarPUTaskRunnerTestAdapter::reserve_inflight_slot(
          runner_.get()));

  EXPECT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      0U);
}

TEST_F(
    StarPUTaskRunnerFixture,
    ReserveInflightSlotReturnsImmediatelyWhenInflightStateMissing)
{
  starpu_server::StarPUTaskRunnerTestAdapter::set_inflight_state_to_null(
      runner_.get());

  EXPECT_NO_THROW(
      starpu_server::StarPUTaskRunnerTestAdapter::reserve_inflight_slot(
          runner_.get()));
  EXPECT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      0U);
}

TEST_F(
    StarPUTaskRunnerFixture, ReleaseInflightSlotReturnsImmediatelyWhenNoLimit)
{
  ASSERT_FALSE(starpu_server::StarPUTaskRunnerTestAdapter::has_inflight_limit(
      runner_.get()));

  starpu_server::StarPUTaskRunnerTestAdapter::release_inflight_slot(
      runner_.get());

  EXPECT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      0U);
}

TEST_F(StarPUTaskRunnerFixture, ReleaseInflightSlotReturnsWhenCountIsZero)
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

  starpu_server::StarPUTaskRunnerTestAdapter::release_inflight_slot(
      runner_.get());

  EXPECT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      0U);
}

TEST_F(StarPUTaskRunnerFixture, ReleaseInflightSlotDecrementsCountWhenNonZero)
{
  opts_.batching.max_inflight_tasks = 10;
  runner_.reset();
  starpu_setup_.reset();
  starpu_setup_ = std::make_unique<starpu_server::StarPUSetup>(opts_);
  config_.starpu = starpu_setup_.get();
  config_.opts = &opts_;
  runner_ = std::make_unique<starpu_server::StarPUTaskRunner>(config_);

  starpu_server::StarPUTaskRunnerTestAdapter::reserve_inflight_slot(
      runner_.get());
  ASSERT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      1U);

  starpu_server::StarPUTaskRunnerTestAdapter::release_inflight_slot(
      runner_.get());

  EXPECT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      0U);
}

TEST_F(
    StarPUTaskRunnerFixture,
    ReleaseInflightSlotNotifiesWaitingThreadsWhenDecrementing)
{
  opts_.batching.max_inflight_tasks = 1;
  runner_.reset();
  starpu_setup_.reset();
  starpu_setup_ = std::make_unique<starpu_server::StarPUSetup>(opts_);
  config_.starpu = starpu_setup_.get();
  config_.opts = &opts_;
  runner_ = std::make_unique<starpu_server::StarPUTaskRunner>(config_);

  starpu_server::StarPUTaskRunnerTestAdapter::reserve_inflight_slot(
      runner_.get());
  ASSERT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      1U);

  std::atomic<bool> waiting_thread_started{false};
  std::atomic<bool> waiting_thread_completed{false};

  std::thread waiting_thread([&] {
    waiting_thread_started.store(true, std::memory_order_release);
    starpu_server::StarPUTaskRunnerTestAdapter::reserve_inflight_slot(
        runner_.get());
    waiting_thread_completed.store(true, std::memory_order_release);
  });

  while (!waiting_thread_started.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_FALSE(waiting_thread_completed.load(std::memory_order_acquire));

  starpu_server::StarPUTaskRunnerTestAdapter::release_inflight_slot(
      runner_.get());

  waiting_thread.join();
  EXPECT_TRUE(waiting_thread_completed.load(std::memory_order_acquire));
  EXPECT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      1U);
}

TEST_F(
    StarPUTaskRunnerFixture, ReserveAndReleaseInflightSlotWorkTogetherWithLimit)
{
  opts_.batching.max_inflight_tasks = 5;
  runner_.reset();
  starpu_setup_.reset();
  starpu_setup_ = std::make_unique<starpu_server::StarPUSetup>(opts_);
  config_.starpu = starpu_setup_.get();
  config_.opts = &opts_;
  runner_ = std::make_unique<starpu_server::StarPUTaskRunner>(config_);

  EXPECT_EQ(
      starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
          runner_.get()),
      0U);

  for (int i = 1; i <= 5; ++i) {
    starpu_server::StarPUTaskRunnerTestAdapter::reserve_inflight_slot(
        runner_.get());
    EXPECT_EQ(
        starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
            runner_.get()),
        static_cast<std::size_t>(i));
  }

  for (int i = 4; i >= 0; --i) {
    starpu_server::StarPUTaskRunnerTestAdapter::release_inflight_slot(
        runner_.get());
    EXPECT_EQ(
        starpu_server::StarPUTaskRunnerTestAdapter::get_inflight_tasks(
            runner_.get()),
        static_cast<std::size_t>(i));
  }
}

TEST_F(
    StarPUTaskRunnerFixture,
    ValidateBatchAndCopyInputsWithActiveCudaCopyBatchCopiesInputsSequentially)
{
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA runtime unavailable, skipping pinned buffer test";
  }

  opts_.devices.use_cuda = true;
  opts_.devices.ids = {0};
  auto model_config = make_model_config(
      "multi_input",
      {make_tensor_config("input0", {3}, at::kFloat),
       make_tensor_config("input1", {2}, at::kFloat)},
      {});
  reset_runner_with_model(model_config, /*pool_size=*/1);
  ASSERT_TRUE(starpu_setup_->has_input_pool());

  const auto tensor_opts = torch::TensorOptions().dtype(torch::kFloat);
  auto input0 =
      torch::tensor(std::vector<float>{1.0F, 2.0F, 3.0F}, tensor_opts);
  auto input1 = torch::tensor(std::vector<float>{4.0F, 5.0F}, tensor_opts);

  auto job = make_job(800, {input0, input1}, {at::kFloat, at::kFloat});

  auto& input_pool = starpu_setup_->input_pool();
  const int slot = input_pool.acquire();
  const auto& buffer_infos = input_pool.host_buffer_infos(slot);

  const bool has_pinned = std::ranges::any_of(
      buffer_infos,
      [](const starpu_server::InputSlotPool::HostBufferInfo& info) {
        return info.cuda_pinned || info.starpu_pinned;
      });
  if (!has_pinned) {
    input_pool.release(slot);
    GTEST_SKIP() << "No pinned buffers allocated, cannot test CUDA copy batch";
  }

  const int64_t batch = starpu_server::StarPUTaskRunnerTestAdapter::
      validate_batch_and_copy_inputs(runner_.get(), job, &input_pool, slot);
  EXPECT_EQ(batch, 1);

  const auto& base_ptrs = input_pool.base_ptrs(slot);
  ASSERT_GE(base_ptrs.size(), 2U);
  const std::vector<float> expected_input0{1.0F, 2.0F, 3.0F};
  std::vector<float> actual_input0(expected_input0.size());
  std::memcpy(
      actual_input0.data(), base_ptrs[0],
      expected_input0.size() * sizeof(float));
  for (size_t idx = 0; idx < expected_input0.size(); ++idx) {
    EXPECT_FLOAT_EQ(actual_input0[idx], expected_input0[idx]);
  }

  const std::vector<float> expected_input1{4.0F, 5.0F};
  std::vector<float> actual_input1(expected_input1.size());
  std::memcpy(
      actual_input1.data(), base_ptrs[1],
      expected_input1.size() * sizeof(float));
  for (size_t idx = 0; idx < expected_input1.size(); ++idx) {
    EXPECT_FLOAT_EQ(actual_input1[idx], expected_input1[idx]);
  }

  input_pool.release(slot);
}

TEST_F(
    StarPUTaskRunnerFixture,
    ValidateBatchAndCopyInputsWithActiveCudaCopyBatchAbortsOnError)
{
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA runtime unavailable, skipping pinned buffer test";
  }

  opts_.devices.use_cuda = true;
  opts_.devices.ids = {0};
  auto model_config = make_model_config(
      "multi_input",
      {make_tensor_config("input0", {3}, at::kFloat),
       make_tensor_config("input1", {3}, at::kFloat),
       make_tensor_config("input2", {3}, at::kFloat)},
      {});
  reset_runner_with_model(model_config, /*pool_size=*/1);
  ASSERT_TRUE(starpu_setup_->has_input_pool());

  const auto tensor_opts = torch::TensorOptions().dtype(torch::kFloat);
  auto valid_a =
      torch::tensor(std::vector<float>{1.0F, 2.0F, 3.0F}, tensor_opts);
  torch::Tensor undefined_tensor;
  ASSERT_FALSE(undefined_tensor.defined());
  auto valid_b =
      torch::tensor(std::vector<float>{4.0F, 5.0F, 6.0F}, tensor_opts);

  auto job = make_job(
      801, {valid_a, undefined_tensor, valid_b},
      {at::kFloat, at::kFloat, at::kFloat});

  auto& input_pool = starpu_setup_->input_pool();
  const int slot = input_pool.acquire();
  const auto& buffer_infos = input_pool.host_buffer_infos(slot);

  const bool has_pinned = std::ranges::any_of(
      buffer_infos,
      [](const starpu_server::InputSlotPool::HostBufferInfo& info) {
        return info.cuda_pinned || info.starpu_pinned;
      });
  if (!has_pinned) {
    input_pool.release(slot);
    GTEST_SKIP() << "No pinned buffers allocated, cannot test CUDA copy batch";
  }

  const auto& base_ptrs = input_pool.base_ptrs(slot);
  constexpr unsigned char kSentinel = 0x7F;
  std::memset(base_ptrs.back(), kSentinel, buffer_infos.back().bytes);

  const auto last_tensor_bytes = static_cast<std::size_t>(valid_b.nbytes());
  std::vector<std::byte> sentinel_pattern(
      last_tensor_bytes, static_cast<std::byte>(kSentinel));

  EXPECT_THROW(
      starpu_server::StarPUTaskRunnerTestAdapter::
          validate_batch_and_copy_inputs(runner_.get(), job, &input_pool, slot),
      starpu_server::InvalidInputTensorException);

  auto* last_destination = base_ptrs.back();
  ASSERT_NE(last_destination, nullptr);
  EXPECT_TRUE(std::equal(
      last_destination, last_destination + last_tensor_bytes,
      sentinel_pattern.begin(), sentinel_pattern.end()));

  input_pool.release(slot);
}

TEST_F(
    StarPUTaskRunnerFixture,
    ValidateBatchAndCopyInputsWithActiveCudaCopyBatchHandlesMultipleInputs)
{
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA runtime unavailable, skipping pinned buffer test";
  }

  opts_.devices.use_cuda = true;
  opts_.devices.ids = {0};
  auto model_config = make_model_config(
      "multi_input",
      {make_tensor_config("input0", {4}, at::kFloat),
       make_tensor_config("input1", {3}, at::kFloat),
       make_tensor_config("input2", {2}, at::kFloat),
       make_tensor_config("input3", {5}, at::kFloat)},
      {});
  reset_runner_with_model(model_config, /*pool_size=*/1);
  ASSERT_TRUE(starpu_setup_->has_input_pool());

  const auto tensor_opts = torch::TensorOptions().dtype(torch::kFloat);
  auto input0 =
      torch::tensor(std::vector<float>{1.0F, 2.0F, 3.0F, 4.0F}, tensor_opts);
  auto input1 =
      torch::tensor(std::vector<float>{5.0F, 6.0F, 7.0F}, tensor_opts);
  auto input2 = torch::tensor(std::vector<float>{8.0F, 9.0F}, tensor_opts);
  auto input3 = torch::tensor(
      std::vector<float>{10.0F, 11.0F, 12.0F, 13.0F, 14.0F}, tensor_opts);

  auto job = make_job(
      802, {input0, input1, input2, input3},
      {at::kFloat, at::kFloat, at::kFloat, at::kFloat});

  auto& input_pool = starpu_setup_->input_pool();
  const int slot = input_pool.acquire();
  const auto& buffer_infos = input_pool.host_buffer_infos(slot);

  const bool has_pinned = std::ranges::any_of(
      buffer_infos,
      [](const starpu_server::InputSlotPool::HostBufferInfo& info) {
        return info.cuda_pinned || info.starpu_pinned;
      });
  if (!has_pinned) {
    input_pool.release(slot);
    GTEST_SKIP() << "No pinned buffers allocated, cannot test CUDA copy batch";
  }

  const int64_t batch = starpu_server::StarPUTaskRunnerTestAdapter::
      validate_batch_and_copy_inputs(runner_.get(), job, &input_pool, slot);
  EXPECT_EQ(batch, 1);

  const auto& base_ptrs = input_pool.base_ptrs(slot);
  ASSERT_GE(base_ptrs.size(), 4U);

  const std::vector<std::vector<float>> expected{
      {1.0F, 2.0F, 3.0F, 4.0F},
      {5.0F, 6.0F, 7.0F},
      {8.0F, 9.0F},
      {10.0F, 11.0F, 12.0F, 13.0F, 14.0F}};

  for (size_t input_idx = 0; input_idx < expected.size(); ++input_idx) {
    const auto& expected_data = expected[input_idx];
    std::vector<float> actual_data(expected_data.size());
    std::memcpy(
        actual_data.data(), base_ptrs[input_idx],
        expected_data.size() * sizeof(float));
    for (size_t elem_idx = 0; elem_idx < expected_data.size(); ++elem_idx) {
      EXPECT_FLOAT_EQ(actual_data[elem_idx], expected_data[elem_idx])
          << "Input " << input_idx << ", element " << elem_idx;
    }
  }

  input_pool.release(slot);
}
