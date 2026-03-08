#include <fcntl.h>
#include <hwloc.h>
#include <starpu.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "core/inference_runner.hpp"
#include "core/starpu_setup.hpp"
#include "inference_service.hpp"
#include "monitoring/congestion_monitor.hpp"
#include "monitoring/metrics.hpp"
#include "starpu_task_worker/inference_queue.hpp"
#include "starpu_task_worker/starpu_task_worker.hpp"
#include "utils/batching_trace_logger.hpp"
#include "utils/config_loader.hpp"
#include "utils/exceptions.hpp"
#include "utils/logger.hpp"
#include "utils/runtime_config.hpp"

namespace {
struct ServerContext {
  grpc::Server* server = nullptr;
  std::mutex server_mutex;
  std::condition_variable server_cv;
  bool server_startup_observed = false;
  std::mutex stop_mutex;
  std::condition_variable stop_cv;
  std::atomic<bool> stop_requested{false};
};

struct ThreadExceptionState {
  std::mutex mutex;
  std::exception_ptr exception;
  std::string thread_name;

  void capture(std::string_view name, std::exception_ptr caught_exception)
  {
    std::lock_guard lock(mutex);
    if (exception != nullptr) {
      return;
    }
    exception = std::move(caught_exception);
    thread_name = name;
  }

  auto take() -> std::pair<std::exception_ptr, std::string>
  {
    std::lock_guard lock(mutex);
    return {std::move(exception), std::move(thread_name)};
  }
};

class RuntimeCleanupGuard {
 public:
  RuntimeCleanupGuard() = default;
  RuntimeCleanupGuard(const RuntimeCleanupGuard&) = delete;
  auto operator=(const RuntimeCleanupGuard&) -> RuntimeCleanupGuard& = delete;
  RuntimeCleanupGuard(RuntimeCleanupGuard&&) = delete;
  auto operator=(RuntimeCleanupGuard&&) -> RuntimeCleanupGuard& = delete;
  ~RuntimeCleanupGuard() noexcept
  {
    if (!active_) {
      return;
    }
    reset_trace_logger_noexcept();
    shutdown_metrics_noexcept();
  }

  void Dismiss() noexcept { active_ = false; }

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  static void ResetTraceLoggerNoexceptForTest() noexcept
  {
    reset_trace_logger_noexcept();
  }

  static void SetResetTraceLoggerNoexceptForceThrowForTest(
      bool enabled) noexcept
  {
    reset_trace_logger_noexcept_force_throw_for_test() = enabled;
  }

  static void ShutdownMetricsNoexceptForTest() noexcept
  {
    shutdown_metrics_noexcept();
  }

  static void SetShutdownMetricsNoexceptForceThrowForTest(bool enabled) noexcept
  {
    shutdown_metrics_noexcept_force_throw_for_test() = enabled;
  }
#endif  // SONAR_IGNORE_STOP

 private:
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  static auto reset_trace_logger_noexcept_force_throw_for_test() noexcept
      -> bool&
  {
    static bool enabled = false;
    return enabled;
  }

  static auto shutdown_metrics_noexcept_force_throw_for_test() noexcept -> bool&
  {
    static bool enabled = false;
    return enabled;
  }
#endif  // SONAR_IGNORE_STOP

  static void reset_trace_logger_noexcept() noexcept
  {
    try {
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
      if (reset_trace_logger_noexcept_force_throw_for_test()) {
        throw std::runtime_error("forced reset_trace_logger_noexcept failure");
      }
#endif  // SONAR_IGNORE_STOP
      auto& tracer = starpu_server::BatchingTraceLogger::instance();
      tracer.configure(false, "");
    }
    catch (...) {
      // Best-effort cleanup in noexcept context.
      return;
    }
  }

  static void shutdown_metrics_noexcept() noexcept
  {
    try {
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
      if (shutdown_metrics_noexcept_force_throw_for_test()) {
        throw std::runtime_error("forced shutdown_metrics_noexcept failure");
      }
#endif  // SONAR_IGNORE_STOP
      starpu_server::shutdown_metrics();
    }
    catch (...) {
      // Best-effort cleanup in noexcept context.
      return;
    }
  }

  bool active_ = true;
};

auto
signal_stop_requested_flag() -> volatile std::sig_atomic_t&
{
  static volatile std::sig_atomic_t value = 0;
  return value;
}

auto
signal_stop_notify_fd() -> volatile std::sig_atomic_t&
{
  static volatile std::sig_atomic_t value = -1;
  return value;
}

class SignalNotificationPipe {
 public:
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  static void SetPipeFailureForTest(bool enabled) noexcept
  {
    pipe_failure_for_test() = enabled;
  }

  static void SetSetNonBlockingFailureForTest(bool enabled) noexcept
  {
    set_non_blocking_failure_for_test() = enabled;
  }

  static auto SetNonBlockingForTest(int file_descriptor) -> bool
  {
    return set_non_blocking(file_descriptor);
  }
#endif  // SONAR_IGNORE_STOP

  SignalNotificationPipe()
  {
    std::array<int, 2> file_descriptors{-1, -1};
    if (create_pipe(file_descriptors) != 0) {
      starpu_server::log_warning(std::format(
          "Failed to create stop-notification pipe; falling back to polling "
          "signal flag: {}",
          std::strerror(errno)));
      return;
    }
    read_fd_ = file_descriptors[0];
    write_fd_ = file_descriptors[1];
    if (!set_non_blocking(write_fd_)) {
      starpu_server::log_warning(std::format(
          "Failed to configure stop-notification pipe write end as "
          "non-blocking; falling back to polling signal flag: {}",
          std::strerror(errno)));
      close_fd_noexcept(read_fd_);
      close_fd_noexcept(write_fd_);
      read_fd_ = -1;
      write_fd_ = -1;
      return;
    }
    signal_stop_notify_fd() = static_cast<std::sig_atomic_t>(write_fd_);
  }

  SignalNotificationPipe(const SignalNotificationPipe&) = delete;
  auto operator=(const SignalNotificationPipe&) -> SignalNotificationPipe& =
                                                       delete;
  SignalNotificationPipe(SignalNotificationPipe&&) = delete;
  auto operator=(SignalNotificationPipe&&) -> SignalNotificationPipe& = delete;

  ~SignalNotificationPipe() noexcept
  {
    shutdown();
    close_fd_noexcept(read_fd_);
  }

  void shutdown() noexcept
  {
    if (write_fd_ >= 0) {
      signal_stop_notify_fd() = -1;
      close_fd_noexcept(write_fd_);
      write_fd_ = -1;
    } else {
      signal_stop_notify_fd() = -1;
    }
  }

  [[nodiscard]] auto read_fd() const noexcept -> int { return read_fd_; }

  [[nodiscard]] auto active() const noexcept -> bool
  {
    return read_fd_ >= 0 && write_fd_ >= 0;
  }

 private:
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  static auto pipe_failure_for_test() noexcept -> bool&
  {
    static bool enabled = false;
    return enabled;
  }

  static auto set_non_blocking_failure_for_test() noexcept -> bool&
  {
    static bool enabled = false;
    return enabled;
  }
#endif  // SONAR_IGNORE_STOP

  static auto create_pipe(std::array<int, 2>& file_descriptors) -> int
  {
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
    if (pipe_failure_for_test()) {
      errno = EMFILE;
      return -1;
    }
#endif  // SONAR_IGNORE_STOP
    return ::pipe(file_descriptors.data());
  }

  static auto set_non_blocking(int file_descriptor) -> bool
  {
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
    if (set_non_blocking_failure_for_test()) {
      errno = EINVAL;
      return false;
    }
#endif  // SONAR_IGNORE_STOP
    if (file_descriptor < 0) {
      return false;
    }
    // POSIX `fcntl` is variadic by API contract.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
    const int flags = ::fcntl(file_descriptor, F_GETFL);
    if (flags < 0) {
      return false;
    }
    // POSIX `fcntl` is variadic by API contract.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg)
    return ::fcntl(file_descriptor, F_SETFL, flags | O_NONBLOCK) == 0;
  }

  static void close_fd_noexcept(int file_descriptor) noexcept
  {
    if (file_descriptor >= 0) {
      (void)::close(file_descriptor);
    }
  }

  int read_fd_ = -1;
  int write_fd_ = -1;
};

void
request_server_stop(ServerContext& ctx)
{
  ctx.stop_requested.store(true, std::memory_order_relaxed);
  ctx.stop_cv.notify_one();
}

template <typename Fn>
void
run_thread_entry_with_exception_capture(
    std::string_view thread_name, ThreadExceptionState& state,
    ServerContext& server_ctx, starpu_server::InferenceQueue* queue,
    Fn&& thread_entry) noexcept
{
  try {
    std::forward<Fn>(thread_entry)();
  }
  catch (const std::exception& e) {
    starpu_server::log_error(std::format(
        "Unhandled exception escaped '{}' thread: {}", thread_name, e.what()));
    state.capture(thread_name, std::current_exception());
    if (queue != nullptr) {
      queue->shutdown();
    }
    request_server_stop(server_ctx);
  }
  catch (...) {
    starpu_server::log_error(std::format(
        "Unhandled non-standard exception escaped '{}' thread.", thread_name));
    state.capture(thread_name, std::current_exception());
    if (queue != nullptr) {
      queue->shutdown();
    }
    request_server_stop(server_ctx);
  }
}

void
rethrow_thread_exception_if_any(ThreadExceptionState& state)
{
  auto [thread_exception, thread_name] = state.take();
  if (thread_exception == nullptr) {
    return;
  }

  try {
    std::rethrow_exception(thread_exception);
  }
  catch (const std::exception& e) {
    throw std::runtime_error(std::format(
        "Thread '{}' terminated with exception: {}", thread_name, e.what()));
  }
  catch (...) {
    throw std::runtime_error(std::format(
        "Thread '{}' terminated with unknown exception.", thread_name));
  }
}

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
using WaitForSignalNotificationReadOverrideForTestFn =
    ssize_t (*)(int, void*, std::size_t);

auto
wait_for_signal_notification_read_override_for_test() noexcept
    -> WaitForSignalNotificationReadOverrideForTestFn&
{
  static WaitForSignalNotificationReadOverrideForTestFn override_fn = nullptr;
  return override_fn;
}
#endif  // SONAR_IGNORE_STOP

auto
read_signal_notification(int read_fd, void* buffer, std::size_t buffer_size)
    -> ssize_t
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto override_fn =
          wait_for_signal_notification_read_override_for_test();
      override_fn != nullptr) {
    return override_fn(read_fd, buffer, buffer_size);
  }
#endif  // SONAR_IGNORE_STOP
  return ::read(read_fd, buffer, buffer_size);
}

void
wait_for_signal_notification(int read_fd)
{
  if (read_fd < 0) {
    return;
  }
  constexpr std::size_t kSignalNotificationBufferSize = 16;
  std::array<char, kSignalNotificationBufferSize> buffer{};
  while (true) {
    const ssize_t bytes_read =
        read_signal_notification(read_fd, buffer.data(), buffer.size());
    if (bytes_read > 0 || bytes_read == 0) {
      return;
    }
    if (errno == EINTR) {
      continue;
    }
    starpu_server::log_warning(std::format(
        "Failed while waiting for stop signal notification: {}",
        std::strerror(errno)));
    return;
  }
}

auto
server_context() -> ServerContext&
{
  static ServerContext ctx;
  return ctx;
}

void
reset_server_state(ServerContext& ctx)
{
  std::lock_guard lock(ctx.server_mutex);
  ctx.server = nullptr;
  ctx.server_startup_observed = false;
}

void
mark_server_started(ServerContext& ctx, grpc::Server* server)
{
  {
    std::lock_guard lock(ctx.server_mutex);
    ctx.server = server;
    ctx.server_startup_observed = true;
  }
  ctx.server_cv.notify_all();
}

void
mark_server_stopped(ServerContext& ctx)
{
  {
    std::lock_guard lock(ctx.server_mutex);
    ctx.server = nullptr;
    ctx.server_startup_observed = true;
  }
  ctx.server_cv.notify_all();
}

void
stop_server_when_available(ServerContext& ctx)
{
  std::unique_lock lock(ctx.server_mutex);
  ctx.server_cv.wait(lock, [&ctx]() { return ctx.server_startup_observed; });
  starpu_server::StopServer(ctx.server);
}

constexpr auto kPlotScriptTimeout = std::chrono::steady_clock::duration::zero();
constexpr auto kPlotScriptPollInterval = std::chrono::milliseconds(50);
constexpr auto kPlotScriptTerminateTimeout = std::chrono::seconds(1);
constexpr auto kShutdownDrainTimeout = std::chrono::seconds(30);
constexpr auto kShutdownDrainWaitStep = std::chrono::milliseconds(250);
constexpr int kSignalExitCodeOffset = 128;
constexpr int kExecFailedExitCode = 127;
constexpr int kPlotScriptSearchDepth = 6;

enum class ShutdownDrainStageForTest : std::uint8_t {
  Entered,
  CompletedReachedTotal,
  DeadlineReached,
  BeforeWait,
};

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START

using ShutdownDrainTimeoutOverrideForTestFn =
    std::chrono::steady_clock::duration (*)();
using ShutdownDrainWaitStepOverrideForTestFn =
    std::chrono::steady_clock::duration (*)();
using ShutdownDrainObserverForTestFn = void (*)(
    ShutdownDrainStageForTest, std::size_t, std::size_t,
    std::chrono::steady_clock::duration);

auto
shutdown_drain_timeout_override_for_test() noexcept
    -> ShutdownDrainTimeoutOverrideForTestFn&
{
  static ShutdownDrainTimeoutOverrideForTestFn override_fn = nullptr;
  return override_fn;
}

auto
shutdown_drain_wait_step_override_for_test() noexcept
    -> ShutdownDrainWaitStepOverrideForTestFn&
{
  static ShutdownDrainWaitStepOverrideForTestFn override_fn = nullptr;
  return override_fn;
}

auto
shutdown_drain_observer_for_test() noexcept -> ShutdownDrainObserverForTestFn&
{
  static ShutdownDrainObserverForTestFn observer_fn = nullptr;
  return observer_fn;
}
#endif  // SONAR_IGNORE_STOP

auto
resolve_shutdown_drain_timeout() -> std::chrono::steady_clock::duration
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto override_fn = shutdown_drain_timeout_override_for_test();
      override_fn != nullptr) {
    return override_fn();
  }
#endif  // SONAR_IGNORE_STOP
  return kShutdownDrainTimeout;
}

auto
resolve_shutdown_drain_wait_step() -> std::chrono::steady_clock::duration
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto override_fn = shutdown_drain_wait_step_override_for_test();
      override_fn != nullptr) {
    return override_fn();
  }
#endif  // SONAR_IGNORE_STOP
  return kShutdownDrainWaitStep;
}

void
notify_shutdown_drain_stage_for_test(
    ShutdownDrainStageForTest stage, std::size_t total_jobs,
    std::size_t completed_jobs,
    std::chrono::steady_clock::duration wait_budget =
        std::chrono::steady_clock::duration::zero())
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto observer_fn = shutdown_drain_observer_for_test();
      observer_fn != nullptr) {
    observer_fn(stage, total_jobs, completed_jobs, wait_budget);
  }
#else
  (void)stage;
  (void)total_jobs;
  (void)completed_jobs;
  (void)wait_budget;
#endif  // SONAR_IGNORE_STOP
}

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
using ResolvePythonCandidatesOverrideForTestFn =
    std::vector<std::filesystem::path> (*)();

using ResolvePythonIsRegularFileOverrideForTestFn =
    bool (*)(const std::filesystem::path&, std::error_code&);

auto
resolve_python_candidates_override_for_test() noexcept
    -> ResolvePythonCandidatesOverrideForTestFn&
{
  static ResolvePythonCandidatesOverrideForTestFn override_fn = nullptr;
  return override_fn;
}

auto
resolve_python_is_regular_file_override_for_test() noexcept
    -> ResolvePythonIsRegularFileOverrideForTestFn&
{
  static ResolvePythonIsRegularFileOverrideForTestFn override_fn = nullptr;
  return override_fn;
}
#endif  // SONAR_IGNORE_STOP

auto
resolve_python_executable() -> std::optional<std::filesystem::path>
{
  static const std::array<std::filesystem::path, 3> kDefaultCandidates = {
      "/usr/bin/python3",
      "/usr/local/bin/python3",
      "/bin/python3",
  };

  std::span<const std::filesystem::path> candidates = kDefaultCandidates;
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  std::vector<std::filesystem::path> override_candidates_storage;
  if (const auto override_fn = resolve_python_candidates_override_for_test();
      override_fn != nullptr) {
    override_candidates_storage = override_fn();
    candidates = override_candidates_storage;
  }
#endif  // SONAR_IGNORE_STOP

  for (const auto& candidate : candidates) {
    std::error_code status_ec;
    const bool is_regular = [&]() {
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
      if (const auto override_fn =
              resolve_python_is_regular_file_override_for_test();
          override_fn != nullptr) {
        return override_fn(candidate, status_ec);
      }
#endif  // SONAR_IGNORE_STOP
      return std::filesystem::is_regular_file(candidate, status_ec);
    }();
    if (!is_regular || status_ec) {
      continue;
    }
    if (::access(candidate.c_str(), X_OK) == 0) {
      return candidate;
    }
  }
  return std::nullopt;
}

auto
wait_status_to_exit_code(int status) -> std::optional<int>
{
  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  if (WIFSIGNALED(status)) {
    return kSignalExitCodeOffset + WTERMSIG(status);
  }
  return std::nullopt;
}

void
log_waitpid_error()
{
  starpu_server::log_warning(std::format(
      "Failed to wait for plot generation process: {}", std::strerror(errno)));
}

enum class WaitPidState : std::uint8_t { Exited, StillRunning, Error };

struct WaitPidResult {
  WaitPidState state = WaitPidState::Error;
  std::optional<int> exit_code;
};

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
using WaitPidNoHangOverrideForTestFn = pid_t (*)(pid_t, int*, int);

auto
waitpid_nohang_override_for_test() noexcept -> WaitPidNoHangOverrideForTestFn&
{
  static WaitPidNoHangOverrideForTestFn override_fn = nullptr;
  return override_fn;
}
#endif  // SONAR_IGNORE_STOP

auto
read_waitpid_nohang(pid_t pid, int* status, int options) -> pid_t
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto override_fn = waitpid_nohang_override_for_test();
      override_fn != nullptr) {
    return override_fn(pid, status, options);
  }
#endif  // SONAR_IGNORE_STOP
  return ::waitpid(pid, status, options);
}

auto
waitpid_nohang(pid_t pid, int& status) -> WaitPidResult
{
  using enum WaitPidState;
  while (true) {
    const pid_t result = read_waitpid_nohang(pid, &status, WNOHANG);
    if (result == pid) {
      return {Exited, wait_status_to_exit_code(status)};
    }
    if (result == 0) {
      return {StillRunning, std::nullopt};
    }
    if (errno == EINTR) {
      continue;
    }
    log_waitpid_error();
    return {Error, std::nullopt};
  }
}

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
using WaitPidBlockingOverrideForTestFn = pid_t (*)(pid_t, int*, int);

auto
waitpid_blocking_override_for_test() noexcept
    -> WaitPidBlockingOverrideForTestFn&
{
  static WaitPidBlockingOverrideForTestFn override_fn = nullptr;
  return override_fn;
}
#endif  // SONAR_IGNORE_STOP

auto
read_waitpid_blocking(pid_t pid, int* status) -> pid_t
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto override_fn = waitpid_blocking_override_for_test();
      override_fn != nullptr) {
    return override_fn(pid, status, 0);
  }
#endif  // SONAR_IGNORE_STOP
  return ::waitpid(pid, status, 0);
}

enum class WaitOutcome : std::uint8_t { Exited, TimedOut, Error };

struct WaitOutcomeResult {
  WaitOutcome outcome = WaitOutcome::Error;
  std::optional<int> exit_code;
};

auto
wait_for_exit_with_timeout(
    pid_t pid, std::chrono::steady_clock::duration timeout) -> WaitOutcomeResult
{
  const auto deadline = (timeout == std::chrono::steady_clock::duration::zero())
                            ? std::chrono::steady_clock::time_point::max()
                            : std::chrono::steady_clock::now() + timeout;
  int status = 0;
  while (true) {
    const auto wait_result = waitpid_nohang(pid, status);
    if (wait_result.state == WaitPidState::Exited) {
      return {WaitOutcome::Exited, wait_result.exit_code};
    }
    if (wait_result.state == WaitPidState::Error) {
      return {WaitOutcome::Error, std::nullopt};
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return {WaitOutcome::TimedOut, std::nullopt};
    }
    std::this_thread::sleep_for(kPlotScriptPollInterval);
  }
}

auto
wait_for_exit_blocking(pid_t pid) -> std::optional<int>
{
  int status = 0;
  while (true) {
    const pid_t result = read_waitpid_blocking(pid, &status);
    if (result == pid) {
      return wait_status_to_exit_code(status);
    }
    if (result < 0 && errno == EINTR) {
      continue;
    }
    if (result < 0) {
      log_waitpid_error();
      return std::nullopt;
    }
  }
}

auto
terminate_and_wait(pid_t pid) -> std::optional<int>
{
  starpu_server::log_warning("Plot generation timed out; terminating python3.");
  (void)::kill(pid, SIGTERM);
  const auto term_result =
      wait_for_exit_with_timeout(pid, kPlotScriptTerminateTimeout);
  if (term_result.outcome == WaitOutcome::Exited) {
    return term_result.exit_code;
  }
  if (term_result.outcome == WaitOutcome::Error) {
    return std::nullopt;
  }
  (void)::kill(pid, SIGKILL);
  return wait_for_exit_blocking(pid);
}

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
using WaitForPlotProcessWaitOverrideForTestFn =
    WaitOutcomeResult (*)(pid_t, std::chrono::steady_clock::duration);
using TerminateAndWaitOverrideForTestFn = std::optional<int> (*)(pid_t);

auto
wait_for_plot_process_wait_override_for_test() noexcept
    -> WaitForPlotProcessWaitOverrideForTestFn&
{
  static WaitForPlotProcessWaitOverrideForTestFn override_fn = nullptr;
  return override_fn;
}

auto
terminate_and_wait_override_for_test() noexcept
    -> TerminateAndWaitOverrideForTestFn&
{
  static TerminateAndWaitOverrideForTestFn override_fn = nullptr;
  return override_fn;
}
#endif  // SONAR_IGNORE_STOP

auto
wait_for_plot_process(pid_t pid) -> std::optional<int>
{
  const auto result = [&]() {
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
    if (const auto override_fn = wait_for_plot_process_wait_override_for_test();
        override_fn != nullptr) {
      return override_fn(pid, kPlotScriptTimeout);
    }
#endif  // SONAR_IGNORE_STOP
    return wait_for_exit_with_timeout(pid, kPlotScriptTimeout);
  }();
  if (result.outcome == WaitOutcome::Exited) {
    return result.exit_code;
  }
  if (result.outcome == WaitOutcome::Error) {
    return std::nullopt;
  }
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto override_fn = terminate_and_wait_override_for_test();
      override_fn != nullptr) {
    return override_fn(pid);
  }
#endif  // SONAR_IGNORE_STOP
  return terminate_and_wait(pid);
}

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
using RunPlotScriptOverrideForTestFn = std::optional<int> (*)(
    const std::filesystem::path&, const std::filesystem::path&,
    const std::filesystem::path&);
using RunPlotScriptForkOverrideForTestFn = pid_t (*)();

using LocatePlotScriptOverrideForTestFn =
    std::optional<std::filesystem::path> (*)(
        const starpu_server::RuntimeConfig&);

using TraceSummaryFilePathOverrideForTestFn =
    std::optional<std::filesystem::path> (*)();

auto
run_plot_script_override_for_test() noexcept -> RunPlotScriptOverrideForTestFn&
{
  static RunPlotScriptOverrideForTestFn override_fn = nullptr;
  return override_fn;
}

auto
run_plot_script_fork_override_for_test() noexcept
    -> RunPlotScriptForkOverrideForTestFn&
{
  static RunPlotScriptForkOverrideForTestFn override_fn = nullptr;
  return override_fn;
}

auto
locate_plot_script_override_for_test() noexcept
    -> LocatePlotScriptOverrideForTestFn&
{
  static LocatePlotScriptOverrideForTestFn override_fn = nullptr;
  return override_fn;
}

auto
trace_summary_file_path_override_for_test() noexcept
    -> TraceSummaryFilePathOverrideForTestFn&
{
  static TraceSummaryFilePathOverrideForTestFn override_fn = nullptr;
  return override_fn;
}
#endif  // SONAR_IGNORE_STOP

auto
run_plot_script(
    const std::filesystem::path& script_path,
    const std::filesystem::path& summary_path,
    const std::filesystem::path& output_path) -> std::optional<int>
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto override_fn = run_plot_script_override_for_test();
      override_fn != nullptr) {
    return override_fn(script_path, summary_path, output_path);
  }
#endif  // SONAR_IGNORE_STOP

  const auto python_path = resolve_python_executable();
  if (!python_path) {
    starpu_server::log_warning(
        "python3 was not found in the allowlist; skipping plot generation.");
    return std::nullopt;
  }

  std::vector<std::string> args{
      python_path->string(), script_path.string(),
      summary_path.string(), "--output",
      output_path.string(),
  };
  std::vector<char*> argv;
  argv.reserve(args.size() + 1);
  for (auto& arg : args) {
    argv.push_back(arg.data());
  }
  argv.push_back(nullptr);

  const pid_t pid = [&]() {
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
    if (const auto override_fn = run_plot_script_fork_override_for_test();
        override_fn != nullptr) {
      return override_fn();
    }
#endif  // SONAR_IGNORE_STOP
    return fork();
  }();
  if (pid < 0) {
    starpu_server::log_warning(std::format(
        "Failed to launch python3 for plot generation: {}",
        std::strerror(errno)));
    return std::nullopt;
  }
  if (pid == 0) {
    execv(argv[0], argv.data());
    _exit(kExecFailedExitCode);
  }

  return wait_for_plot_process(pid);
}

auto
resolve_starpu_scheduler(const starpu_server::RuntimeConfig& opts)
    -> std::string
{
  if (const auto scheduler_env_it =
          opts.starpu_env.find(starpu_server::kStarpuSchedulerEnvVar);
      scheduler_env_it != opts.starpu_env.end()) {
    return std::format("{} (from starpu_env)", scheduler_env_it->second);
  }

  if (const char* env_value =
          std::getenv(starpu_server::kStarpuSchedulerEnvVar.data())) {
    return std::format("{} (from environment)", env_value);
  }

  return std::format("{} (default)", starpu_server::kDefaultStarpuScheduler);
}

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
using CandidatePlotScriptsReadSymlinkOverrideForTestFn =
    std::filesystem::path (*)(const std::filesystem::path&, std::error_code&);
using LocatePlotScriptCandidatesOverrideForTestFn =
    std::vector<std::filesystem::path> (*)();

auto
candidate_plot_scripts_read_symlink_override_for_test() noexcept
    -> CandidatePlotScriptsReadSymlinkOverrideForTestFn&
{
  static CandidatePlotScriptsReadSymlinkOverrideForTestFn override_fn = nullptr;
  return override_fn;
}

auto
locate_plot_script_candidates_override_for_test() noexcept
    -> LocatePlotScriptCandidatesOverrideForTestFn&
{
  static LocatePlotScriptCandidatesOverrideForTestFn override_fn = nullptr;
  return override_fn;
}
#endif  // SONAR_IGNORE_STOP

auto
read_symlink_for_candidate_plot_scripts(
    const std::filesystem::path& path,
    std::error_code& ec) -> std::filesystem::path
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto override_fn =
          candidate_plot_scripts_read_symlink_override_for_test();
      override_fn != nullptr) {
    return override_fn(path, ec);
  }
#endif  // SONAR_IGNORE_STOP
  return std::filesystem::read_symlink(path, ec);
}

auto
candidate_plot_scripts() -> std::vector<std::filesystem::path>
{
  std::vector<std::filesystem::path> candidates;
  std::error_code exe_ec;
  const auto exe_path =
      read_symlink_for_candidate_plot_scripts("/proc/self/exe", exe_ec);
  if (exe_ec) {
    return candidates;
  }
  auto base_dir = std::filesystem::path(exe_path).parent_path();
  for (int depth = 0; depth < kPlotScriptSearchDepth; ++depth) {
    candidates.emplace_back(base_dir / "scripts/plot_batch_summary.py");
    if (!base_dir.has_parent_path()) {
      break;
    }
    base_dir = base_dir.parent_path();
  }
  return candidates;
}

auto
locate_plot_script(const starpu_server::RuntimeConfig& opts)
    -> std::optional<std::filesystem::path>
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto override_fn = locate_plot_script_override_for_test();
      override_fn != nullptr) {
    return override_fn(opts);
  }
#endif  // SONAR_IGNORE_STOP

  const auto candidates = [&]() {
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
    if (const auto override_fn =
            locate_plot_script_candidates_override_for_test();
        override_fn != nullptr) {
      return override_fn();
    }
#endif  // SONAR_IGNORE_STOP
    return candidate_plot_scripts();
  }();

  for (const auto& candidate : candidates) {
    if (candidate.empty()) {
      continue;
    }
    auto resolved = candidate;
    if (!resolved.is_absolute()) {
      std::error_code abs_ec;
      const auto absolute = std::filesystem::absolute(resolved, abs_ec);
      if (!abs_ec) {
        resolved = absolute;
      }
    }
    std::error_code exists_ec;
    if (std::filesystem::exists(resolved, exists_ec) && !exists_ec) {
      if (std::error_code type_ec;
          !std::filesystem::is_regular_file(resolved, type_ec) || type_ec) {
        continue;
      }
      return resolved;
    }
  }
  return std::nullopt;
}

auto
plots_output_path(const std::filesystem::path& summary_path)
    -> std::filesystem::path
{
  auto filename = summary_path.stem().string();
  if (const auto pos = filename.rfind("_summary"); pos != std::string::npos) {
    filename.erase(pos);
  }
  filename += "_plots.png";
  auto output = summary_path;
  output.replace_filename(filename);
  return output;
}

void
run_trace_plots_if_enabled(const starpu_server::RuntimeConfig& opts)
{
  if (!opts.batching.trace_enabled) {
    return;
  }

  std::optional<std::filesystem::path> summary_path_opt;
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto override_fn = trace_summary_file_path_override_for_test();
      override_fn != nullptr) {
    summary_path_opt = override_fn();
  } else {
    const auto& tracer = starpu_server::BatchingTraceLogger::instance();
    summary_path_opt = tracer.summary_file_path();
  }
#else
  const auto& tracer = starpu_server::BatchingTraceLogger::instance();
  summary_path_opt = tracer.summary_file_path();
#endif  // SONAR_IGNORE_STOP
  if (!summary_path_opt) {
    starpu_server::log_warning(
        "Tracing was enabled but no trace.csv was produced; "
        "skipping plot generation.");
    return;
  }

  const auto& summary_path = *summary_path_opt;
  if (std::error_code err_code;
      !std::filesystem::exists(summary_path, err_code) || err_code) {
    starpu_server::log_warning(std::format(
        "Tracing summary file '{}' not found; skipping plot generation.",
        summary_path.string()));
    return;
  }

  const auto script_path = locate_plot_script(opts);
  if (!script_path) {
    starpu_server::log_warning(
        "Unable to locate scripts/plot_batch_summary.py; skipping plot "
        "generation.");
    return;
  }

  const auto output_path = plots_output_path(summary_path);
  const auto exit_code =
      run_plot_script(*script_path, summary_path, output_path);
  if (!exit_code.has_value()) {
    starpu_server::log_warning(
        "Failed to generate batching latency plots; plot script did not "
        "complete.");
  } else if (*exit_code != 0) {
    starpu_server::log_warning(std::format(
        "Failed to generate batching latency plots; python3 {} {} --output {} "
        "exited with code {}.",
        script_path->string(), summary_path.string(), output_path.string(),
        *exit_code));
  } else {
    starpu_server::log_info(
        opts.verbosity,
        std::format(
            "Batching latency plots written to '{}'.", output_path.string()));
  }
}
}  // namespace

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

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
using HandleProgramArgumentsFatalOverrideForTestFn = void (*)(std::string_view);

auto
handle_program_arguments_fatal_override_for_test() noexcept
    -> HandleProgramArgumentsFatalOverrideForTestFn&
{
  static HandleProgramArgumentsFatalOverrideForTestFn override_fn = nullptr;
  return override_fn;
}
#endif  // SONAR_IGNORE_STOP

[[noreturn]] void
handle_program_arguments_fatal(std::string_view message)
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto override_fn =
          handle_program_arguments_fatal_override_for_test();
      override_fn != nullptr) {
    override_fn(message);
    std::terminate();
  }
#endif  // SONAR_IGNORE_STOP
  starpu_server::log_fatal(std::string(message));
}

auto
handle_program_arguments(std::span<char const* const> args)
    -> starpu_server::RuntimeConfig
{
  const char* config_path = nullptr;

  auto remaining = args.subspan(1);
  auto require_value = [&](std::string_view flag) {
    if (remaining.empty() || remaining.front() == nullptr) {
      handle_program_arguments_fatal(
          std::format("Missing value for {} argument.\n", flag));
    }
    const char* value = remaining.front();
    remaining = remaining.subspan(1);
    return value;
  };

  while (!remaining.empty()) {
    const char* raw_arg = remaining.front();
    remaining = remaining.subspan(1);

    if (raw_arg == nullptr) {
      handle_program_arguments_fatal("Unexpected null program argument.\n");
    }

    std::string_view arg{raw_arg};
    if (arg == "--config" || arg == "-c") {
      config_path = require_value(arg);
      continue;
    }
    handle_program_arguments_fatal(std::format(
        "Unknown argument '{}'. Only --config/-c is supported; all other "
        "settings must live in the YAML file.\n",
        arg));
  }

  if (config_path == nullptr) {
    handle_program_arguments_fatal("Missing required --config argument.\n");
  }

  starpu_server::RuntimeConfig cfg = starpu_server::load_config(config_path);

  if (!cfg.valid) {
    handle_program_arguments_fatal("Invalid configuration file.\n");
  }

  log_info(cfg.verbosity, std::format("__cplusplus = {}", __cplusplus));
  log_info(cfg.verbosity, std::format("LibTorch version: {}", TORCH_VERSION));
  log_info(
      cfg.verbosity,
      std::format("StarPU scheduler: {}", resolve_starpu_scheduler(cfg)));
  if (!cfg.name.empty()) {
    log_info(cfg.verbosity, std::format("Configuration   : {}", cfg.name));
  }

  return cfg;
}

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
auto
load_model_and_reference_output_override_for_test() noexcept
    -> std::optional<std::tuple<
        torch::jit::script::Module, std::vector<torch::jit::script::Module>,
        std::vector<torch::Tensor>>> (*&)(const starpu_server::RuntimeConfig&)
{
  static std::optional<std::tuple<
      torch::jit::script::Module, std::vector<torch::jit::script::Module>,
      std::vector<torch::Tensor>>> (*override_fn)(
      const starpu_server::RuntimeConfig&) = nullptr;
  return override_fn;
}

auto
run_warmup_override_for_test() noexcept
    -> void (*&)(
        const starpu_server::RuntimeConfig&, starpu_server::StarPUSetup&,
        torch::jit::script::Module&, std::vector<torch::jit::script::Module>&,
        const std::vector<torch::Tensor>&)
{
  static void (*override_fn)(
      const starpu_server::RuntimeConfig&, starpu_server::StarPUSetup&,
      torch::jit::script::Module&, std::vector<torch::jit::script::Module>&,
      const std::vector<torch::Tensor>&) = nullptr;
  return override_fn;
}
#endif  // SONAR_IGNORE_STOP

auto
prepare_models_and_warmup(
    const starpu_server::RuntimeConfig& opts,
    starpu_server::StarPUSetup& starpu)
    -> std::tuple<
        torch::jit::script::Module, std::vector<torch::jit::script::Module>,
        std::vector<torch::Tensor>>
{
  auto models = [&]() {
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
    if (const auto override_fn =
            load_model_and_reference_output_override_for_test();
        override_fn != nullptr) {
      return override_fn(opts);
    }
#endif  // SONAR_IGNORE_STOP
    return starpu_server::load_model_and_reference_output(opts);
  }();
  if (!models) {
    throw starpu_server::ModelLoadingException(
        "Failed to load model or reference outputs");
  }
  auto [model_cpu, models_gpu, reference_outputs] = std::move(*models);
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto override_fn = run_warmup_override_for_test();
      override_fn != nullptr) {
    override_fn(opts, starpu, model_cpu, models_gpu, reference_outputs);
  } else {
    starpu_server::run_warmup(
        opts, starpu, model_cpu, models_gpu, reference_outputs);
  }
#else
  starpu_server::run_warmup(
      opts, starpu, model_cpu, models_gpu, reference_outputs);
#endif  // SONAR_IGNORE_STOP
  return {model_cpu, models_gpu, reference_outputs};
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

auto
resolve_default_model_name(const starpu_server::RuntimeConfig& opts)
    -> std::string
{
  if (opts.model.has_value() && !opts.model->name.empty()) {
    return opts.model->name;
  }
  return opts.name;
}

auto
make_grpc_server_options(const starpu_server::RuntimeConfig& opts)
    -> starpu_server::GrpcServerOptions
{
  return {opts.server_address, opts.batching.max_message_bytes,
          opts.verbosity,      resolve_default_model_name(opts),
          opts.name,           ""};
}

auto
make_grpc_model_spec(
    const starpu_server::RuntimeConfig& opts,
    std::span<const at::ScalarType> expected_input_types,
    std::span<const std::vector<int64_t>> expected_input_dims,
    std::span<const std::string> expected_input_names,
    std::span<const std::string> expected_output_names)
    -> starpu_server::GrpcModelSpec
{
  return {
      .expected_input_types = expected_input_types,
      .expected_input_dims = expected_input_dims,
      .expected_input_names = expected_input_names,
      .expected_output_names = expected_output_names,
      .max_batch_size = opts.batching.max_batch_size};
}

void
launch_threads(  // NOLINT(readability-function-cognitive-complexity)
    const starpu_server::RuntimeConfig& opts,
    starpu_server::StarPUSetup& starpu, torch::jit::script::Module& model_cpu,
    std::vector<torch::jit::script::Module>& models_gpu,
    std::vector<torch::Tensor>& reference_outputs,
    starpu_server::InferenceQueue& queue)
{
  queue.reset_counters();
  auto& server_ctx = server_context();
  ThreadExceptionState thread_exception_state;
  server_ctx.stop_requested.store(false, std::memory_order_relaxed);
  signal_stop_requested_flag() = 0;
  reset_server_state(server_ctx);

  SignalNotificationPipe signal_pipe;
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  starpu_server::congestion::start(&queue, make_congestion_config(opts));

  const bool use_blocking_signal_wait = signal_pipe.active();
  std::jthread notifier_thread([&server_ctx, &queue, &thread_exception_state,
                                read_fd = signal_pipe.read_fd(),
                                use_blocking_signal_wait]() {
    run_thread_entry_with_exception_capture(
        "signal-notifier", thread_exception_state, server_ctx, &queue,
        [read_fd, use_blocking_signal_wait, &server_ctx]() {
          if (use_blocking_signal_wait) {
            if (signal_stop_requested_flag() == 0) {
              wait_for_signal_notification(read_fd);
            }
          } else {
            constexpr auto kNotifierSleep = std::chrono::milliseconds(10);
            while (signal_stop_requested_flag() == 0) {
              std::this_thread::sleep_for(kNotifierSleep);
            }
          }
          if (signal_stop_requested_flag() != 0) {
            request_server_stop(server_ctx);
          }
        });
  });
  struct SignalPipeShutdownGuard {
    SignalNotificationPipe* pipe = nullptr;
    SignalPipeShutdownGuard() = default;
    explicit SignalPipeShutdownGuard(
        SignalNotificationPipe* signal_pipe) noexcept
        : pipe(signal_pipe)
    {
    }
    SignalPipeShutdownGuard(const SignalPipeShutdownGuard&) = delete;
    auto operator=(const SignalPipeShutdownGuard&) -> SignalPipeShutdownGuard& =
                                                          delete;
    SignalPipeShutdownGuard(SignalPipeShutdownGuard&&) = delete;
    auto operator=(SignalPipeShutdownGuard&&) -> SignalPipeShutdownGuard& =
                                                     delete;
    ~SignalPipeShutdownGuard()
    {
      if (pipe != nullptr) {
        pipe->shutdown();
      }
    }
  };
  const SignalPipeShutdownGuard signal_pipe_shutdown_guard{&signal_pipe};

  std::atomic<std::size_t> completed_jobs{0};
  std::condition_variable all_done_cv;
  std::mutex all_done_mutex;

  starpu_server::StarPUTaskRunnerConfig config{};
  config.queue = &queue;
  config.model_cpu = &model_cpu;
  config.models_gpu = &models_gpu;
  config.starpu = &starpu;
  config.opts = &opts;
  config.completed_jobs = &completed_jobs;
  config.all_done_cv = &all_done_cv;
  starpu_server::StarPUTaskRunner worker(config);

  std::jthread worker_thread(
      [&server_ctx, &queue, &thread_exception_state, &worker]() {
        run_thread_entry_with_exception_capture(
            "starpu-worker", thread_exception_state, server_ctx, &queue,
            [&worker]() { worker.run(); });
      });
  std::vector<at::ScalarType> expected_input_types;
  if (opts.model.has_value()) {
    expected_input_types.reserve(opts.model->inputs.size());
    for (const auto& input : opts.model->inputs) {
      expected_input_types.push_back(input.type);
    }
  }
  std::vector<std::string> expected_input_names;
  if (opts.model.has_value()) {
    expected_input_names.reserve(opts.model->inputs.size());
    for (const auto& input : opts.model->inputs) {
      expected_input_names.push_back(input.name);
    }
  }
  std::vector<std::vector<int64_t>> expected_input_dims;
  if (opts.model.has_value()) {
    expected_input_dims.reserve(opts.model->inputs.size());
    for (const auto& input : opts.model->inputs) {
      expected_input_dims.push_back(input.dims);
    }
  }
  std::vector<std::string> expected_output_names;
  if (opts.model.has_value()) {
    expected_output_names.reserve(opts.model->outputs.size());
    for (const auto& output : opts.model->outputs) {
      expected_output_names.push_back(output.name);
    }
  }

  std::jthread grpc_thread([&server_ctx, &queue, &thread_exception_state, &opts,
                            &reference_outputs, &expected_input_types,
                            &expected_input_dims, &expected_input_names,
                            &expected_output_names]() {
    run_thread_entry_with_exception_capture(
        "grpc-server", thread_exception_state, server_ctx, &queue, [&]() {
          std::unique_ptr<grpc::Server> grpc_server;
          const auto server_hooks = starpu_server::GrpcServerLifecycleHooks{
              .on_started =
                  [&server_ctx](grpc::Server* server) {
                    mark_server_started(server_ctx, server);
                  },
              .on_stopped =
                  [&server_ctx]() {
                    mark_server_stopped(server_ctx);
                    request_server_stop(server_ctx);
                  }};
          const auto server_options = make_grpc_server_options(opts);
          const auto model_spec = make_grpc_model_spec(
              opts, expected_input_types, expected_input_dims,
              expected_input_names, expected_output_names);
          starpu_server::RunGrpcServer(
              queue, reference_outputs, model_spec, server_options, grpc_server,
              server_hooks);
        });
  });

  {
    std::unique_lock lock(server_ctx.stop_mutex);
    server_ctx.stop_cv.wait(lock, [&server_ctx] {
      return server_ctx.stop_requested.load(std::memory_order_relaxed);
    });
  }
  stop_server_when_available(server_ctx);
  queue.shutdown();
  const auto total_jobs = queue.total_pushed();
  const auto completed_before_drain =
      completed_jobs.load(std::memory_order_acquire);
  if (total_jobs > completed_before_drain) {
    const auto remaining_before_drain = total_jobs - completed_before_drain;
    starpu_server::log_info(
        opts.verbosity,
        std::format(
            "Shutdown drain started: completed={} total={} remaining={}.",
            completed_before_drain, total_jobs, remaining_before_drain));
  }
  if (total_jobs > 0) {
    notify_shutdown_drain_stage_for_test(
        ShutdownDrainStageForTest::Entered, total_jobs, completed_before_drain);
    const auto shutdown_drain_timeout = resolve_shutdown_drain_timeout();
    const auto shutdown_drain_wait_step = resolve_shutdown_drain_wait_step();
    const auto deadline =
        std::chrono::steady_clock::now() + shutdown_drain_timeout;
    std::unique_lock lock(all_done_mutex);
    while (true) {
      const auto completed = completed_jobs.load(std::memory_order_acquire);
      if (completed >= total_jobs) {
        notify_shutdown_drain_stage_for_test(
            ShutdownDrainStageForTest::CompletedReachedTotal, total_jobs,
            completed);
        break;
      }

      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline) {
        notify_shutdown_drain_stage_for_test(
            ShutdownDrainStageForTest::DeadlineReached, total_jobs, completed);
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
          ShutdownDrainStageForTest::BeforeWait, total_jobs, completed,
          wait_budget);
      static_cast<void>(all_done_cv.wait_for(lock, wait_budget));
    }
  }
  starpu_server::congestion::shutdown();
  server_ctx.stop_cv.notify_one();
  rethrow_thread_exception_if_any(thread_exception_state);
}

auto
worker_type_label(const enum starpu_worker_archtype type) -> std::string
{
  switch (type) {
    case STARPU_CPU_WORKER:
      return "CPU";
    case STARPU_CUDA_WORKER:
      return "CUDA";
    default:
      return std::format("Other({})", static_cast<int>(type));
  }
}

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
using WorkerCpusetProviderOverrideForTestFn =
    decltype(&starpu_worker_get_hwloc_cpuset);
using HwlocBitmapFirstOverrideForTestFn = decltype(&hwloc_bitmap_first);
using HwlocBitmapNextOverrideForTestFn = decltype(&hwloc_bitmap_next);
using HwlocBitmapFreeOverrideForTestFn = decltype(&hwloc_bitmap_free);

auto
worker_cpuset_provider_override_for_test() noexcept
    -> WorkerCpusetProviderOverrideForTestFn&
{
  static WorkerCpusetProviderOverrideForTestFn override_fn = nullptr;
  return override_fn;
}

auto
hwloc_bitmap_first_override_for_test() noexcept
    -> HwlocBitmapFirstOverrideForTestFn&
{
  static HwlocBitmapFirstOverrideForTestFn override_fn = nullptr;
  return override_fn;
}

auto
hwloc_bitmap_next_override_for_test() noexcept
    -> HwlocBitmapNextOverrideForTestFn&
{
  static HwlocBitmapNextOverrideForTestFn override_fn = nullptr;
  return override_fn;
}

auto
hwloc_bitmap_free_override_for_test() noexcept
    -> HwlocBitmapFreeOverrideForTestFn&
{
  static HwlocBitmapFreeOverrideForTestFn override_fn = nullptr;
  return override_fn;
}
#endif  // SONAR_IGNORE_STOP

auto
get_worker_cpuset_for_affinity(int worker_id) -> hwloc_cpuset_t
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto override_fn = worker_cpuset_provider_override_for_test();
      override_fn != nullptr) {
    return override_fn(worker_id);
  }
#endif  // SONAR_IGNORE_STOP
  return starpu_worker_get_hwloc_cpuset(worker_id);
}

auto
bitmap_first_for_affinity(hwloc_const_bitmap_t cpuset) -> int
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto override_fn = hwloc_bitmap_first_override_for_test();
      override_fn != nullptr) {
    return override_fn(cpuset);
  }
#endif  // SONAR_IGNORE_STOP
  return hwloc_bitmap_first(cpuset);
}

auto
bitmap_next_for_affinity(hwloc_const_bitmap_t cpuset, int previous_core) -> int
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto override_fn = hwloc_bitmap_next_override_for_test();
      override_fn != nullptr) {
    return override_fn(cpuset, previous_core);
  }
#endif  // SONAR_IGNORE_STOP
  return hwloc_bitmap_next(cpuset, previous_core);
}

void
bitmap_free_for_affinity(hwloc_bitmap_t cpuset)
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto override_fn = hwloc_bitmap_free_override_for_test();
      override_fn != nullptr) {
    override_fn(cpuset);
    return;
  }
#endif  // SONAR_IGNORE_STOP
  hwloc_bitmap_free(cpuset);
}

auto
format_cpu_core_ranges(const std::vector<int>& cpus) -> std::string
{
  if (cpus.empty()) {
    return {};
  }

  std::string result;
  auto flush_range = [&](int start, int end) {
    if (!result.empty()) {
      result.push_back(',');
    }
    if (start == end) {
      result += std::to_string(start);
    } else {
      result += std::format("{}-{}", start, end);
    }
  };

  int range_start = cpus.front();
  int previous = range_start;
  for (std::size_t idx = 1; idx < cpus.size(); ++idx) {
    const int core = cpus[idx];
    if (core == previous + 1) {
      previous = core;
    } else {
      flush_range(range_start, previous);
      range_start = previous = core;
    }
  }
  flush_range(range_start, previous);
  return result;
}

auto
describe_cpu_affinity(int worker_id) -> std::string
{
  hwloc_cpuset_t cpuset = get_worker_cpuset_for_affinity(worker_id);
  if (cpuset == nullptr) {
    return {};
  }

  std::vector<int> cores;
  for (int core = bitmap_first_for_affinity(cpuset); core != -1;
       core = bitmap_next_for_affinity(cpuset, core)) {
    cores.push_back(core);
  }
  bitmap_free_for_affinity(cpuset);
  return format_cpu_core_ranges(cores);
}

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
using WorkerCountOverrideForTestFn = decltype(&starpu_worker_get_count);
using WorkerTypeOverrideForTestFn = decltype(&starpu_worker_get_type);
using WorkerDeviceIdOverrideForTestFn = decltype(&starpu_worker_get_devid);
using DescribeCpuAffinityOverrideForTestFn = std::string (*)(int);

auto
worker_count_override_for_test() noexcept -> WorkerCountOverrideForTestFn&
{
  static WorkerCountOverrideForTestFn override_fn = nullptr;
  return override_fn;
}

auto
worker_type_override_for_test() noexcept -> WorkerTypeOverrideForTestFn&
{
  static WorkerTypeOverrideForTestFn override_fn = nullptr;
  return override_fn;
}

auto
worker_device_id_override_for_test() noexcept
    -> WorkerDeviceIdOverrideForTestFn&
{
  static WorkerDeviceIdOverrideForTestFn override_fn = nullptr;
  return override_fn;
}

auto
describe_cpu_affinity_override_for_test() noexcept
    -> DescribeCpuAffinityOverrideForTestFn&
{
  static DescribeCpuAffinityOverrideForTestFn override_fn = nullptr;
  return override_fn;
}
#endif  // SONAR_IGNORE_STOP

auto
worker_count_for_inventory() -> int
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto override_fn = worker_count_override_for_test();
      override_fn != nullptr) {
    return static_cast<int>(override_fn());
  }
#endif  // SONAR_IGNORE_STOP
  return static_cast<int>(starpu_worker_get_count());
}

auto
worker_type_for_inventory(int worker_id) -> enum starpu_worker_archtype {
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto override_fn = worker_type_override_for_test();
      override_fn != nullptr){return override_fn(worker_id);}
#endif  // SONAR_IGNORE_STOP
return starpu_worker_get_type(worker_id);
}

auto
worker_device_id_for_inventory(int worker_id) -> int
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto override_fn = worker_device_id_override_for_test();
      override_fn != nullptr) {
    return override_fn(worker_id);
  }
#endif  // SONAR_IGNORE_STOP
  return starpu_worker_get_devid(worker_id);
}

auto
describe_cpu_affinity_for_inventory(int worker_id) -> std::string
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto override_fn = describe_cpu_affinity_override_for_test();
      override_fn != nullptr) {
    return override_fn(worker_id);
  }
#endif  // SONAR_IGNORE_STOP
  return describe_cpu_affinity(worker_id);
}

void
log_worker_inventory(const starpu_server::RuntimeConfig& opts)
{
  const auto total_workers = worker_count_for_inventory();
  starpu_server::log_info(
      opts.verbosity,
      std::format("Configured {} StarPU worker(s).", total_workers));

  for (int worker_id = 0; worker_id < total_workers; ++worker_id) {
    const auto type = worker_type_for_inventory(worker_id);
    const int device_id = worker_device_id_for_inventory(worker_id);
    const std::string device_label =
        device_id >= 0 ? std::to_string(device_id) : "N/A";
    std::string cpu_affinity;
    if (type == STARPU_CPU_WORKER) {
      const std::string affinity =
          describe_cpu_affinity_for_inventory(worker_id);
      if (!affinity.empty()) {
        cpu_affinity = std::format(", cores={}", affinity);
      }
    }
    starpu_server::log_info(
        opts.verbosity,
        std::format(
            "Worker {:2d}: type={}, device id={}{}", worker_id,
            worker_type_label(type), device_label, cpu_affinity));
  }
}

auto
main(int argc, char* argv[]) -> int
{
  try {
    starpu_server::RuntimeConfig opts =
        handle_program_arguments({argv, static_cast<size_t>(argc)});
    RuntimeCleanupGuard cleanup_guard;
    starpu_server::BatchingTraceLogger::instance().configure_from_runtime(opts);
    const bool metrics_ok = starpu_server::init_metrics(opts.metrics_port);
    if (!metrics_ok) {
      starpu_server::log_warning(
          "Metrics server failed to start; continuing without metrics.");
    }
    starpu_server::StarPUSetup starpu(opts);
    log_worker_inventory(opts);
    auto [model_cpu, models_gpu, reference_outputs] =
        prepare_models_and_warmup(opts, starpu);
    starpu_server::InferenceQueue queue(opts.batching.max_queue_size);
    launch_threads(
        opts, starpu, model_cpu, models_gpu, reference_outputs, queue);
    auto& tracer = starpu_server::BatchingTraceLogger::instance();
    tracer.configure(false, "");
    run_trace_plots_if_enabled(opts);
    starpu_server::shutdown_metrics();
    cleanup_guard.Dismiss();
  }
  catch (const starpu_server::InferenceEngineException& e) {
    std::cerr << "\o{33}[1;31m[Inference Error] " << e.what() << "\o{33}[0m\n";
    return 2;
  }
  catch (const std::exception& e) {
    std::cerr << "\o{33}[1;31m[General Error] " << e.what() << "\o{33}[0m\n";
    return -1;
  }

  return 0;
}
