void
signal_handler(int /*signal*/)
{
  const int saved_errno = errno;
  signal_stop_requested_flag() = 1;
  const auto notify_fd = signal_stop_notify_fd();
  if (notify_fd >= 0) {
    const std::uint8_t byte = 1;
    const ssize_t write_result =
        ::write(static_cast<int>(notify_fd), &byte, sizeof(byte));
    (void)write_result;
  }
  errno = saved_errno;
}

constexpr auto kShutdownDrainTimeout = std::chrono::seconds(30);
constexpr auto kShutdownDrainWaitStep = std::chrono::milliseconds(250);

enum class ShutdownDrainStageForTest : std::uint8_t {
  Entered,
  CompletedReachedTotal,
  DeadlineReached,
  BeforeWait,
};

struct ShutdownDrainProgressForTest {
  std::size_t total_jobs = 0;
  std::size_t completed_jobs = 0;
};

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
STARPU_SERVER_DECLARE_TEST_OVERRIDE_SLOT(
    ShutdownDrainTimeoutOverrideForTestFn,
    shutdown_drain_timeout_override_for_test,
    std::chrono::steady_clock::duration (*)())
STARPU_SERVER_DECLARE_TEST_OVERRIDE_SLOT(
    ShutdownDrainWaitStepOverrideForTestFn,
    shutdown_drain_wait_step_override_for_test,
    std::chrono::steady_clock::duration (*)())
using ShutdownDrainObserverForTestFn = void (*)(
    ShutdownDrainStageForTest, ShutdownDrainProgressForTest,
    std::chrono::steady_clock::duration);
STARPU_SERVER_DECLARE_TEST_OVERRIDE_SLOT(
    ShutdownDrainObserverForTestFn, shutdown_drain_observer_for_test,
    void (*)(
        ShutdownDrainStageForTest, ShutdownDrainProgressForTest,
        std::chrono::steady_clock::duration))
#endif  // SONAR_IGNORE_STOP

auto
resolve_shutdown_drain_timeout() -> std::chrono::steady_clock::duration
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  return ::starpu_server::testing::server_main::detail::call_override_or(
      shutdown_drain_timeout_override_for_test,
      []() -> std::chrono::steady_clock::duration {
        return kShutdownDrainTimeout;
      });
#else
  return kShutdownDrainTimeout;
#endif  // SONAR_IGNORE_STOP
}

auto
resolve_shutdown_drain_wait_step() -> std::chrono::steady_clock::duration
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  return ::starpu_server::testing::server_main::detail::call_override_or(
      shutdown_drain_wait_step_override_for_test,
      []() -> std::chrono::steady_clock::duration {
        return kShutdownDrainWaitStep;
      });
#else
  return kShutdownDrainWaitStep;
#endif  // SONAR_IGNORE_STOP
}

void
notify_shutdown_drain_stage_for_test(
    ShutdownDrainStageForTest stage, ShutdownDrainProgressForTest progress,
    std::chrono::steady_clock::duration wait_budget =
        std::chrono::steady_clock::duration::zero())
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  ::starpu_server::testing::server_main::detail::call_override_or(
      shutdown_drain_observer_for_test,
      [](ShutdownDrainStageForTest, ShutdownDrainProgressForTest,
         std::chrono::steady_clock::duration) {},
      stage, progress, wait_budget);
#else
  (void)stage;
  (void)progress;
  (void)wait_budget;
#endif  // SONAR_IGNORE_STOP
}

auto
make_congestion_config(const starpu_server::RuntimeConfig& cfg)
    -> starpu_server::congestion::Config
{
  using namespace std::chrono;

  starpu_server::congestion::Config out{};
  out.enabled = cfg.congestion.enabled;
  out.latency_slo_ms = cfg.congestion.latency_slo_ms;
  out.queue_latency_budget_ms = cfg.congestion.queue_latency_budget_ms;
  out.queue_latency_budget_ratio = cfg.congestion.queue_latency_budget_ratio;
  out.e2e_warn_ratio = cfg.congestion.e2e_warn_ratio;
  out.e2e_ok_ratio = cfg.congestion.e2e_ok_ratio;
  out.fill_high = cfg.congestion.fill_high;
  out.fill_low = cfg.congestion.fill_low;
  out.rho_high = cfg.congestion.rho_high;
  out.rho_low = cfg.congestion.rho_low;
  out.alpha = cfg.congestion.alpha;
  out.entry_horizon =
      milliseconds(std::max(1, cfg.congestion.entry_horizon_ms));
  out.exit_horizon = milliseconds(std::max(1, cfg.congestion.exit_horizon_ms));
  out.tick_interval =
      milliseconds(std::max(1, cfg.congestion.tick_interval_ms));
  return out;
}

struct ShutdownRuntimeContext {
  ThreadExceptionState& thread_exception_state;
  std::atomic<std::size_t>& completed_jobs;
  std::condition_variable& all_done_cv;
  std::mutex& all_done_mutex;
};

void
wait_for_stop_request(ServerContext& server_ctx)
{
  std::unique_lock lock(server_ctx.stop_mutex);
  server_ctx.stop_cv.wait(lock, [&server_ctx] {
    return server_ctx.stop_requested.load(std::memory_order_relaxed);
  });
}

void
log_shutdown_drain_start(
    starpu_server::VerbosityLevel verbosity, std::size_t total_jobs,
    std::size_t completed_before_drain)
{
  if (total_jobs <= completed_before_drain) {
    return;
  }
  const auto remaining_before_drain = total_jobs - completed_before_drain;
  starpu_server::log_info(
      verbosity,
      std::format(
          "Shutdown drain started: completed={} total={} remaining={}.",
          completed_before_drain, total_jobs, remaining_before_drain));
}

void
drain_shutdown_jobs(
    const starpu_server::InferenceQueue& queue, std::size_t total_jobs,
    std::atomic<std::size_t>& completed_jobs,
    std::size_t completed_before_drain, std::condition_variable& all_done_cv,
    std::mutex& all_done_mutex)
{
  if (total_jobs == 0) {
    return;
  }

  notify_shutdown_drain_stage_for_test(
      ShutdownDrainStageForTest::Entered,
      ShutdownDrainProgressForTest{
          .total_jobs = total_jobs,
          .completed_jobs = completed_before_drain,
      });
  const auto shutdown_drain_timeout = resolve_shutdown_drain_timeout();
  const auto shutdown_drain_wait_step = resolve_shutdown_drain_wait_step();
  const auto deadline =
      std::chrono::steady_clock::now() + shutdown_drain_timeout;

  std::unique_lock lock(all_done_mutex);
  while (true) {
    const auto completed = completed_jobs.load(std::memory_order_acquire);
    if (completed >= total_jobs) {
      notify_shutdown_drain_stage_for_test(
          ShutdownDrainStageForTest::CompletedReachedTotal,
          ShutdownDrainProgressForTest{
              .total_jobs = total_jobs,
              .completed_jobs = completed,
          });
      break;
    }

    const auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      notify_shutdown_drain_stage_for_test(
          ShutdownDrainStageForTest::DeadlineReached,
          ShutdownDrainProgressForTest{
              .total_jobs = total_jobs,
              .completed_jobs = completed,
          });
      const auto remaining = total_jobs - completed;
      const auto timeout_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              shutdown_drain_timeout)
              .count();
      starpu_server::log_error(std::format(
          "Shutdown drain timeout after {} ms: completed={} total={} "
          "remaining={} queue_size={}.",
          timeout_ms, completed, total_jobs, remaining, queue.size()));
      break;
    }

    const auto until_deadline = deadline - now;
    const auto wait_budget = until_deadline < shutdown_drain_wait_step
                                 ? until_deadline
                                 : shutdown_drain_wait_step;
    notify_shutdown_drain_stage_for_test(
        ShutdownDrainStageForTest::BeforeWait,
        ShutdownDrainProgressForTest{
            .total_jobs = total_jobs,
            .completed_jobs = completed,
        },
        wait_budget);
    static_cast<void>(all_done_cv.wait_for(lock, wait_budget));
  }
}

void
setup_runtime_state(
    const starpu_server::RuntimeConfig& opts, ServerContext& server_ctx,
    starpu_server::InferenceQueue& queue)
{
  queue.reset_counters();
  server_ctx.stop_requested.store(false, std::memory_order_relaxed);
  signal_stop_requested_flag() = 0;
  reset_server_state(server_ctx);
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);
  starpu_server::congestion::start(&queue, make_congestion_config(opts));
}

void
run_shutdown_sequence(
    const starpu_server::RuntimeConfig& opts, ServerContext& server_ctx,
    starpu_server::InferenceQueue& queue,
    const ShutdownRuntimeContext& runtime_context)
{
  wait_for_stop_request(server_ctx);
  stop_server_when_available(server_ctx);
  queue.shutdown();
  const auto total_jobs = queue.total_pushed();
  const auto completed_before_drain =
      runtime_context.completed_jobs.load(std::memory_order_acquire);
  log_shutdown_drain_start(opts.verbosity, total_jobs, completed_before_drain);
  drain_shutdown_jobs(
      queue, total_jobs, runtime_context.completed_jobs, completed_before_drain,
      runtime_context.all_done_cv, runtime_context.all_done_mutex);
  starpu_server::congestion::shutdown();
  server_ctx.stop_cv.notify_one();
  rethrow_thread_exception_if_any(runtime_context.thread_exception_state);
}
