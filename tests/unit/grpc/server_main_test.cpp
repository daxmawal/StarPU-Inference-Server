#include <arpa/inet.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <prometheus/client_metric.h>
#include <prometheus/metric_family.h>
#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <latch>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "../../../src/starpu_task_worker/result_dispatcher_component.hpp"
#include "../../../src/starpu_task_worker/task_runner_internal.hpp"
#include "support/grpc/server/server_main_test_api.hpp"
#include "support/utils/batching_trace_logger_test_api.hpp"
#include "test_batching_config.hpp"
#include "test_helpers.hpp"

namespace {

using namespace ::starpu_server::testing::server_main_api;

starpu_server::testing::ScopedStarpuSilent g_starpu_silent{};

struct ScopedDeathTestStyle {
  ScopedDeathTestStyle()
  {
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  }
};

ScopedDeathTestStyle g_scoped_death_test_style{};

struct ResetInjectedObservabilityNoexceptTag {
  using type = void (*)(
      const std::shared_ptr<starpu_server::RuntimeObservability>&) noexcept;
  friend type get(ResetInjectedObservabilityNoexceptTag);
};

template <typename Tag, typename Tag::type Member>
struct PrivateMemberAccess {
  friend typename Tag::type get(Tag) { return Member; }
};

template struct PrivateMemberAccess<
    ResetInjectedObservabilityNoexceptTag,
    &RuntimeCleanupGuard::reset_injected_observability_noexcept>;

auto
running_under_tsan() -> bool
{
#if defined(__SANITIZE_THREAD__)
  return true;
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
  return true;
#endif
#endif
  return false;
}

struct TempFileGuard {
  std::filesystem::path path;

  ~TempFileGuard()
  {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
};

struct TempDirectoryGuard {
  std::filesystem::path path;

  ~TempDirectoryGuard()
  {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
};

class OStreamCapture {
 public:
  explicit OStreamCapture(std::ostream& stream)
      : stream_(stream), old_buffer_(stream.rdbuf(buffer_.rdbuf()))
  {
  }

  ~OStreamCapture() { stream_.rdbuf(old_buffer_); }

  [[nodiscard]] auto str() const -> std::string { return buffer_.str(); }

 private:
  std::ostream& stream_;
  std::ostringstream buffer_;
  std::streambuf* old_buffer_;
};

auto
make_temp_test_directory(std::string_view stem) -> TempDirectoryGuard
{
  static std::atomic<std::uint64_t> temp_directory_counter{0};
  const auto id =
      temp_directory_counter.fetch_add(1, std::memory_order_relaxed);
  const auto path = std::filesystem::temp_directory_path() /
                    (std::string(stem) + "_" + std::to_string(id));
  std::error_code ec;
  std::filesystem::create_directories(path, ec);
  EXPECT_FALSE(ec) << "Failed to create temporary directory: " << path;
  return TempDirectoryGuard{path};
}

auto
expected_warning_log(std::string_view message) -> std::string
{
  return std::string("\x1b[1;33m[WARNING] ") + std::string(message) +
         "\x1b[0m\n";
}

auto
expected_info_log(std::string_view message) -> std::string
{
  return std::string("\x1b[1;32m[INFO] ") + std::string(message) + "\x1b[0m\n";
}

auto
FindFamily(
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
MetricMatchesLabels(
    const prometheus::ClientMetric& metric,
    const std::vector<std::pair<std::string_view, std::string_view>>& labels)
    -> bool
{
  for (const auto& [label_name, label_value] : labels) {
    bool matched = false;
    for (const auto& label : metric.label) {
      if (label.name == label_name && label.value == label_value) {
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
FindGaugeValue(
    const std::vector<prometheus::MetricFamily>& families,
    std::string_view family_name,
    const std::vector<std::pair<std::string_view, std::string_view>>& labels)
    -> std::optional<double>
{
  const auto* family = FindFamily(families, family_name);
  if (family == nullptr) {
    return std::nullopt;
  }
  for (const auto& metric : family->metric) {
    if (MetricMatchesLabels(metric, labels)) {
      return metric.gauge.value;
    }
  }
  return std::nullopt;
}

struct ScopedTraceLoggerReset {
  ~ScopedTraceLoggerReset()
  {
    starpu_server::BatchingTraceLogger::instance().configure(false, "");
  }
};

struct ScopedResetTraceLoggerForcedThrow {
  explicit ScopedResetTraceLoggerForcedThrow(bool enabled = true) noexcept
  {
    RuntimeCleanupGuard::SetResetTraceLoggerNoexceptForceThrowForTest(enabled);
  }

  ~ScopedResetTraceLoggerForcedThrow()
  {
    RuntimeCleanupGuard::SetResetTraceLoggerNoexceptForceThrowForTest(false);
  }
};

struct ScopedShutdownMetricsForcedThrow {
  explicit ScopedShutdownMetricsForcedThrow(bool enabled = true) noexcept
  {
    RuntimeCleanupGuard::SetShutdownMetricsNoexceptForceThrowForTest(enabled);
  }

  ~ScopedShutdownMetricsForcedThrow()
  {
    RuntimeCleanupGuard::SetShutdownMetricsNoexceptForceThrowForTest(false);
  }
};

struct ScopedSignalPipeForcedPipeFailure {
  explicit ScopedSignalPipeForcedPipeFailure(bool enabled = true) noexcept
  {
    SignalNotificationPipe::SetPipeFailureForTest(enabled);
  }

  ~ScopedSignalPipeForcedPipeFailure()
  {
    SignalNotificationPipe::SetPipeFailureForTest(false);
  }
};

struct ScopedSignalPipeForcedSetNonBlockingFailure {
  explicit ScopedSignalPipeForcedSetNonBlockingFailure(
      bool enabled = true) noexcept
  {
    SignalNotificationPipe::SetSetNonBlockingFailureForTest(enabled);
  }

  ~ScopedSignalPipeForcedSetNonBlockingFailure()
  {
    SignalNotificationPipe::SetSetNonBlockingFailureForTest(false);
  }
};

struct ScopedRunBeforeSubmitHook {
  explicit ScopedRunBeforeSubmitHook(std::function<void()> hook)
  {
    starpu_server::task_runner_internal::set_run_before_submit_hook(
        std::move(hook));
  }

  ~ScopedRunBeforeSubmitHook()
  {
    starpu_server::task_runner_internal::reset_run_before_submit_hook();
  }
};

struct ScopedSubmitInferenceTaskHook {
  explicit ScopedSubmitInferenceTaskHook(std::function<void()> hook)
  {
    starpu_server::task_runner_internal::set_submit_inference_task_hook(
        std::move(hook));
  }

  ~ScopedSubmitInferenceTaskHook()
  {
    starpu_server::task_runner_internal::reset_submit_inference_task_hook();
  }
};

struct ScopedPrepareJobCompletionCallbackHooks {
  explicit ScopedPrepareJobCompletionCallbackHooks(
      starpu_server::ResultDispatcher::PrepareJobCompletionCallbackTestHooks
          hooks)
  {
    starpu_server::ResultDispatcher::SetPrepareJobCompletionCallbackTestHooks(
        std::move(hooks));
  }

  ~ScopedPrepareJobCompletionCallbackHooks()
  {
    starpu_server::ResultDispatcher::
        ClearPrepareJobCompletionCallbackTestHooks();
  }
};

struct ShutdownDrainOverrideState {
  std::chrono::steady_clock::duration timeout = kShutdownDrainTimeout;
  std::chrono::steady_clock::duration wait_step = kShutdownDrainWaitStep;
  int timeout_calls = 0;
  int wait_step_calls = 0;
  int entered_calls = 0;
  int completed_reached_total_calls = 0;
  int deadline_reached_calls = 0;
  int before_wait_calls = 0;
  std::size_t last_total_jobs = 0;
  std::size_t last_completed_jobs = 0;
  std::chrono::steady_clock::duration last_wait_budget{};
  std::mutex mutex;
  std::condition_variable cv;
};

auto
shutdown_drain_override_state() -> ShutdownDrainOverrideState*&
{
  static ShutdownDrainOverrideState* state = nullptr;
  return state;
}

auto
shutdown_drain_timeout_override_stub() -> std::chrono::steady_clock::duration
{
  auto* state = shutdown_drain_override_state();
  if (state == nullptr) {
    return kShutdownDrainTimeout;
  }
  ++state->timeout_calls;
  return state->timeout;
}

auto
shutdown_drain_wait_step_override_stub() -> std::chrono::steady_clock::duration
{
  auto* state = shutdown_drain_override_state();
  if (state == nullptr) {
    return kShutdownDrainWaitStep;
  }
  ++state->wait_step_calls;
  return state->wait_step;
}

void
shutdown_drain_observer_stub(
    ShutdownDrainStageForTest stage, ShutdownDrainProgressForTest progress,
    std::chrono::steady_clock::duration wait_budget)
{
  auto* state = shutdown_drain_override_state();
  if (state == nullptr) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    state->last_total_jobs = progress.total_jobs;
    state->last_completed_jobs = progress.completed_jobs;
    state->last_wait_budget = wait_budget;
    switch (stage) {
      case ShutdownDrainStageForTest::Entered:
        ++state->entered_calls;
        break;
      case ShutdownDrainStageForTest::CompletedReachedTotal:
        ++state->completed_reached_total_calls;
        break;
      case ShutdownDrainStageForTest::DeadlineReached:
        ++state->deadline_reached_calls;
        break;
      case ShutdownDrainStageForTest::BeforeWait:
        ++state->before_wait_calls;
        break;
    }
  }
  state->cv.notify_all();
}

struct ScopedShutdownDrainOverrides {
  explicit ScopedShutdownDrainOverrides(ShutdownDrainOverrideState& state)
  {
    shutdown_drain_override_state() = &state;
    shutdown_drain_timeout_override_for_test() =
        shutdown_drain_timeout_override_stub;
    shutdown_drain_wait_step_override_for_test() =
        shutdown_drain_wait_step_override_stub;
    shutdown_drain_observer_for_test() = shutdown_drain_observer_stub;
  }

  ~ScopedShutdownDrainOverrides()
  {
    shutdown_drain_observer_for_test() = nullptr;
    shutdown_drain_wait_step_override_for_test() = nullptr;
    shutdown_drain_timeout_override_for_test() = nullptr;
    shutdown_drain_override_state() = nullptr;
  }
};

class ScopedEnvironmentVariableUnsetGuard {
 public:
  explicit ScopedEnvironmentVariableUnsetGuard(std::string name)
      : name_(std::move(name))
  {
    if (const char* current = std::getenv(name_.c_str()); current != nullptr) {
      previous_value_ = std::string(current);
    }
    if (::unsetenv(name_.c_str()) != 0) {
      ADD_FAILURE() << "Failed to unset environment variable " << name_;
    }
  }

  ~ScopedEnvironmentVariableUnsetGuard()
  {
    if (previous_value_.has_value()) {
      if (::setenv(name_.c_str(), previous_value_->c_str(), 1) != 0) {
        ADD_FAILURE() << "Failed to restore environment variable " << name_;
      }
      return;
    }
    if (::unsetenv(name_.c_str()) != 0) {
      ADD_FAILURE() << "Failed to unset environment variable " << name_;
    }
  }

  ScopedEnvironmentVariableUnsetGuard(
      const ScopedEnvironmentVariableUnsetGuard&) = delete;
  auto operator=(const ScopedEnvironmentVariableUnsetGuard&)
      -> ScopedEnvironmentVariableUnsetGuard& = delete;
  ScopedEnvironmentVariableUnsetGuard(ScopedEnvironmentVariableUnsetGuard&&) =
      delete;
  auto operator=(ScopedEnvironmentVariableUnsetGuard&&)
      -> ScopedEnvironmentVariableUnsetGuard& = delete;

 private:
  std::string name_;
  std::optional<std::string> previous_value_;
};

struct LibtorchThreadOverrideState {
  int intraop_calls = 0;
  int interop_calls = 0;
  std::optional<int> last_intraop;
  std::optional<int> last_interop;
};

auto
libtorch_thread_override_state() -> LibtorchThreadOverrideState*&
{
  static LibtorchThreadOverrideState* state = nullptr;
  return state;
}

void
set_libtorch_intraop_threads_override_stub(int value)
{
  auto* state = libtorch_thread_override_state();
  if (state == nullptr) {
    return;
  }
  ++state->intraop_calls;
  state->last_intraop = value;
}

void
set_libtorch_interop_threads_override_stub(int value)
{
  auto* state = libtorch_thread_override_state();
  if (state == nullptr) {
    return;
  }
  ++state->interop_calls;
  state->last_interop = value;
}

struct ScopedLibtorchThreadOverrides {
  explicit ScopedLibtorchThreadOverrides(LibtorchThreadOverrideState& state)
  {
    libtorch_thread_override_state() = &state;
    set_libtorch_intraop_threads_override_for_test() =
        set_libtorch_intraop_threads_override_stub;
    set_libtorch_interop_threads_override_for_test() =
        set_libtorch_interop_threads_override_stub;
  }

  ~ScopedLibtorchThreadOverrides()
  {
    set_libtorch_interop_threads_override_for_test() = nullptr;
    set_libtorch_intraop_threads_override_for_test() = nullptr;
    libtorch_thread_override_state() = nullptr;
  }
};

struct ResolvePythonExecutableOverrideState {
  std::vector<std::filesystem::path> candidates;
  bool is_regular_file_result = false;
  bool force_status_error = false;
  int candidates_calls = 0;
  int is_regular_file_calls = 0;
  std::vector<std::filesystem::path> observed_candidates;
};

auto
resolve_python_executable_override_state()
    -> ResolvePythonExecutableOverrideState*&
{
  static ResolvePythonExecutableOverrideState* state = nullptr;
  return state;
}

auto
resolve_python_candidates_override_stub() -> std::vector<std::filesystem::path>
{
  auto* state = resolve_python_executable_override_state();
  if (state == nullptr) {
    return {};
  }
  ++state->candidates_calls;
  return state->candidates;
}

auto
resolve_python_is_regular_file_override_stub(
    const std::filesystem::path& candidate, std::error_code& status_ec) -> bool
{
  auto* state = resolve_python_executable_override_state();
  if (state == nullptr) {
    status_ec.clear();
    return false;
  }
  ++state->is_regular_file_calls;
  state->observed_candidates.push_back(candidate);
  if (state->force_status_error) {
    status_ec = std::error_code(EIO, std::generic_category());
  } else {
    status_ec.clear();
  }
  return state->is_regular_file_result;
}

struct ScopedResolvePythonExecutableOverrides {
  explicit ScopedResolvePythonExecutableOverrides(
      ResolvePythonExecutableOverrideState& state) noexcept
  {
    resolve_python_executable_override_state() = &state;
    hooks_.resolve_python_candidates = resolve_python_candidates_override_stub;
    hooks_.resolve_python_is_regular_file =
        resolve_python_is_regular_file_override_stub;
  }

  ~ScopedResolvePythonExecutableOverrides()
  {
    resolve_python_executable_override_state() = nullptr;
  }

  auto hooks() noexcept -> TracePlotRuntimeHooks& { return hooks_; }
  auto hooks() const noexcept -> const TracePlotRuntimeHooks& { return hooks_; }

 private:
  TracePlotRuntimeHooks hooks_{};
};

struct CandidatePlotScriptsReadSymlinkOverrideState {
  std::filesystem::path resolved_executable_path;
  std::error_code error;
  int call_count = 0;
  std::filesystem::path observed_link_path;
};

auto
candidate_plot_scripts_read_symlink_override_state()
    -> CandidatePlotScriptsReadSymlinkOverrideState*&
{
  static CandidatePlotScriptsReadSymlinkOverrideState* state = nullptr;
  return state;
}

auto
candidate_plot_scripts_read_symlink_override_stub(
    const std::filesystem::path& path,
    std::error_code& ec) -> std::filesystem::path
{
  auto* state = candidate_plot_scripts_read_symlink_override_state();
  if (state == nullptr) {
    ec = std::make_error_code(std::errc::no_such_file_or_directory);
    return {};
  }
  ++state->call_count;
  state->observed_link_path = path;
  ec = state->error;
  return state->resolved_executable_path;
}

struct ScopedCandidatePlotScriptsReadSymlinkOverride {
  explicit ScopedCandidatePlotScriptsReadSymlinkOverride(
      CandidatePlotScriptsReadSymlinkOverrideState& state) noexcept
  {
    candidate_plot_scripts_read_symlink_override_state() = &state;
    hooks_.candidate_plot_scripts_read_symlink =
        candidate_plot_scripts_read_symlink_override_stub;
  }

  ~ScopedCandidatePlotScriptsReadSymlinkOverride()
  {
    candidate_plot_scripts_read_symlink_override_state() = nullptr;
  }

  auto hooks() noexcept -> TracePlotRuntimeHooks& { return hooks_; }
  auto hooks() const noexcept -> const TracePlotRuntimeHooks& { return hooks_; }

 private:
  TracePlotRuntimeHooks hooks_{};
};

struct LocatePlotScriptCandidatesOverrideState {
  std::vector<std::filesystem::path> candidates;
  int call_count = 0;
};

auto
locate_plot_script_candidates_override_state()
    -> LocatePlotScriptCandidatesOverrideState*&
{
  static LocatePlotScriptCandidatesOverrideState* state = nullptr;
  return state;
}

auto
locate_plot_script_candidates_override_stub()
    -> std::vector<std::filesystem::path>
{
  auto* state = locate_plot_script_candidates_override_state();
  if (state == nullptr) {
    return {};
  }
  ++state->call_count;
  return state->candidates;
}

struct ScopedLocatePlotScriptCandidatesOverride {
  explicit ScopedLocatePlotScriptCandidatesOverride(
      LocatePlotScriptCandidatesOverrideState& state) noexcept
  {
    locate_plot_script_candidates_override_state() = &state;
    hooks_.locate_plot_script_candidates =
        locate_plot_script_candidates_override_stub;
  }

  ~ScopedLocatePlotScriptCandidatesOverride()
  {
    locate_plot_script_candidates_override_state() = nullptr;
  }

  auto hooks() noexcept -> TracePlotRuntimeHooks& { return hooks_; }
  auto hooks() const noexcept -> const TracePlotRuntimeHooks& { return hooks_; }

 private:
  TracePlotRuntimeHooks hooks_{};
};

struct WaitForSignalNotificationReadOverrideState {
  std::vector<ssize_t> read_results;
  std::vector<int> errnos;
  std::size_t next_index = 0;
  int call_count = 0;
  int last_read_fd = -1;
  std::size_t last_buffer_size = 0;
};

auto
wait_for_signal_notification_read_override_state()
    -> WaitForSignalNotificationReadOverrideState*&
{
  static WaitForSignalNotificationReadOverrideState* state = nullptr;
  return state;
}

auto
wait_for_signal_notification_read_override_stub(
    int read_fd, void* /*buffer*/, std::size_t buffer_size) -> ssize_t
{
  auto* state = wait_for_signal_notification_read_override_state();
  if (state == nullptr) {
    return 0;
  }
  ++state->call_count;
  state->last_read_fd = read_fd;
  state->last_buffer_size = buffer_size;

  if (state->next_index >= state->read_results.size()) {
    errno = 0;
    return 0;
  }

  const auto index = state->next_index++;
  errno = index < state->errnos.size() ? state->errnos[index] : 0;
  return state->read_results[index];
}

struct ScopedWaitForSignalNotificationReadOverride {
  explicit ScopedWaitForSignalNotificationReadOverride(
      WaitForSignalNotificationReadOverrideState& state) noexcept
  {
    wait_for_signal_notification_read_override_state() = &state;
    wait_for_signal_notification_read_override_for_test() =
        wait_for_signal_notification_read_override_stub;
  }

  ~ScopedWaitForSignalNotificationReadOverride()
  {
    wait_for_signal_notification_read_override_for_test() = nullptr;
    wait_for_signal_notification_read_override_state() = nullptr;
  }
};

struct WaitPidNoHangOverrideState {
  std::vector<pid_t> waitpid_results;
  std::vector<int> statuses;
  std::vector<int> errnos;
  std::size_t next_index = 0;
  int call_count = 0;
  pid_t last_pid = -1;
  int last_options = 0;
};

auto
waitpid_nohang_override_state() -> WaitPidNoHangOverrideState*&
{
  static WaitPidNoHangOverrideState* state = nullptr;
  return state;
}

auto
waitpid_nohang_override_stub(pid_t pid, int* status, int options) -> pid_t
{
  auto* state = waitpid_nohang_override_state();
  if (state == nullptr) {
    errno = ECHILD;
    if (status != nullptr) {
      *status = 0;
    }
    return -1;
  }

  ++state->call_count;
  state->last_pid = pid;
  state->last_options = options;

  if (state->next_index >= state->waitpid_results.size()) {
    errno = 0;
    if (status != nullptr) {
      *status = 0;
    }
    return 0;
  }

  const auto index = state->next_index++;
  errno = index < state->errnos.size() ? state->errnos[index] : 0;
  if (status != nullptr) {
    *status = index < state->statuses.size() ? state->statuses[index] : 0;
  }
  return state->waitpid_results[index];
}

struct ScopedWaitPidNoHangOverride {
  explicit ScopedWaitPidNoHangOverride(
      WaitPidNoHangOverrideState& state) noexcept
  {
    waitpid_nohang_override_state() = &state;
    hooks_.waitpid_nohang = waitpid_nohang_override_stub;
  }

  ~ScopedWaitPidNoHangOverride() { waitpid_nohang_override_state() = nullptr; }

  auto hooks() noexcept -> TracePlotRuntimeHooks& { return hooks_; }
  auto hooks() const noexcept -> const TracePlotRuntimeHooks& { return hooks_; }

 private:
  TracePlotRuntimeHooks hooks_{};
};

struct WaitPidBlockingOverrideState {
  std::vector<pid_t> waitpid_results;
  std::vector<int> statuses;
  std::vector<int> errnos;
  std::size_t next_index = 0;
  int call_count = 0;
  pid_t last_pid = -1;
  int last_options = -1;
};

auto
waitpid_blocking_override_state() -> WaitPidBlockingOverrideState*&
{
  static WaitPidBlockingOverrideState* state = nullptr;
  return state;
}

auto
waitpid_blocking_override_stub(pid_t pid, int* status, int options) -> pid_t
{
  auto* state = waitpid_blocking_override_state();
  if (state == nullptr) {
    errno = ECHILD;
    if (status != nullptr) {
      *status = 0;
    }
    return -1;
  }

  ++state->call_count;
  state->last_pid = pid;
  state->last_options = options;

  if (state->next_index >= state->waitpid_results.size()) {
    errno = ECHILD;
    if (status != nullptr) {
      *status = 0;
    }
    return -1;
  }

  const auto index = state->next_index++;
  errno = index < state->errnos.size() ? state->errnos[index] : 0;
  if (status != nullptr) {
    *status = index < state->statuses.size() ? state->statuses[index] : 0;
  }
  return state->waitpid_results[index];
}

struct ScopedWaitPidBlockingOverride {
  explicit ScopedWaitPidBlockingOverride(
      WaitPidBlockingOverrideState& state) noexcept
  {
    waitpid_blocking_override_state() = &state;
    hooks_.waitpid_blocking = waitpid_blocking_override_stub;
  }

  ~ScopedWaitPidBlockingOverride()
  {
    waitpid_blocking_override_state() = nullptr;
  }

  auto hooks() noexcept -> TracePlotRuntimeHooks& { return hooks_; }
  auto hooks() const noexcept -> const TracePlotRuntimeHooks& { return hooks_; }

 private:
  TracePlotRuntimeHooks hooks_{};
};

struct WaitForPlotProcessOverrideState {
  WaitOutcomeResult wait_result{WaitOutcome::Error, std::nullopt};
  std::optional<int> terminate_result = std::nullopt;
  int wait_calls = 0;
  int terminate_calls = 0;
  pid_t wait_pid = -1;
  pid_t terminate_pid = -1;
  std::chrono::steady_clock::duration wait_timeout{};
};

auto
wait_for_plot_process_override_state() -> WaitForPlotProcessOverrideState*&
{
  static WaitForPlotProcessOverrideState* state = nullptr;
  return state;
}

auto
wait_for_plot_process_wait_override_stub(
    pid_t pid, std::chrono::steady_clock::duration timeout) -> WaitOutcomeResult
{
  auto* state = wait_for_plot_process_override_state();
  if (state == nullptr) {
    return {WaitOutcome::Error, std::nullopt};
  }
  ++state->wait_calls;
  state->wait_pid = pid;
  state->wait_timeout = timeout;
  return state->wait_result;
}

auto
terminate_and_wait_override_stub(pid_t pid) -> std::optional<int>
{
  auto* state = wait_for_plot_process_override_state();
  if (state == nullptr) {
    return std::nullopt;
  }
  ++state->terminate_calls;
  state->terminate_pid = pid;
  return state->terminate_result;
}

struct ScopedWaitForPlotProcessOverrides {
  explicit ScopedWaitForPlotProcessOverrides(
      WaitForPlotProcessOverrideState& state) noexcept
  {
    wait_for_plot_process_override_state() = &state;
    hooks_.wait_for_plot_process_wait =
        wait_for_plot_process_wait_override_stub;
    hooks_.terminate_and_wait = terminate_and_wait_override_stub;
  }

  ~ScopedWaitForPlotProcessOverrides()
  {
    wait_for_plot_process_override_state() = nullptr;
  }

  auto hooks() noexcept -> TracePlotRuntimeHooks& { return hooks_; }
  auto hooks() const noexcept -> const TracePlotRuntimeHooks& { return hooks_; }

 private:
  TracePlotRuntimeHooks hooks_{};
};

struct RunPlotScriptForkOverrideState {
  pid_t result = -1;
  int error_number = EAGAIN;
  int call_count = 0;
};

auto
run_plot_script_fork_override_state() -> RunPlotScriptForkOverrideState*&
{
  static RunPlotScriptForkOverrideState* state = nullptr;
  return state;
}

auto
run_plot_script_fork_override_stub() -> pid_t
{
  auto* state = run_plot_script_fork_override_state();
  if (state == nullptr) {
    errno = EAGAIN;
    return -1;
  }
  ++state->call_count;
  errno = state->error_number;
  return state->result;
}

struct ScopedRunPlotScriptForkOverride {
  explicit ScopedRunPlotScriptForkOverride(
      RunPlotScriptForkOverrideState& state) noexcept
  {
    run_plot_script_fork_override_state() = &state;
    hooks_.run_plot_script_fork = run_plot_script_fork_override_stub;
  }

  ~ScopedRunPlotScriptForkOverride()
  {
    run_plot_script_fork_override_state() = nullptr;
  }

  auto hooks() noexcept -> TracePlotRuntimeHooks& { return hooks_; }
  auto hooks() const noexcept -> const TracePlotRuntimeHooks& { return hooks_; }

 private:
  TracePlotRuntimeHooks hooks_{};
};

struct PlotScriptOverrideState {
  std::optional<std::filesystem::path> summary_path_result;
  std::optional<std::filesystem::path> locate_result;
  std::optional<int> run_result;
  int summary_path_calls = 0;
  int locate_calls = 0;
  int run_calls = 0;
  std::filesystem::path run_script_path;
  std::filesystem::path run_summary_path;
  std::filesystem::path run_output_path;
};

auto
plot_script_override_state() -> PlotScriptOverrideState*&
{
  static PlotScriptOverrideState* state = nullptr;
  return state;
}

auto
trace_summary_file_path_override_stub() -> std::optional<std::filesystem::path>
{
  auto* state = plot_script_override_state();
  if (state == nullptr) {
    return std::nullopt;
  }
  ++state->summary_path_calls;
  return state->summary_path_result;
}

auto
locate_plot_script_override_stub(const starpu_server::RuntimeConfig&)
    -> std::optional<std::filesystem::path>
{
  auto* state = plot_script_override_state();
  if (state == nullptr) {
    return std::nullopt;
  }
  ++state->locate_calls;
  return state->locate_result;
}

auto
run_plot_script_override_stub(
    const std::filesystem::path& script_path,
    const std::filesystem::path& summary_path,
    const std::filesystem::path& output_path) -> std::optional<int>
{
  auto* state = plot_script_override_state();
  if (state == nullptr) {
    return std::nullopt;
  }
  ++state->run_calls;
  state->run_script_path = script_path;
  state->run_summary_path = summary_path;
  state->run_output_path = output_path;
  return state->run_result;
}

struct ScopedPlotScriptOverrides {
  explicit ScopedPlotScriptOverrides(PlotScriptOverrideState& state) noexcept
  {
    plot_script_override_state() = &state;
    hooks_.trace_summary_file_path = trace_summary_file_path_override_stub;
    hooks_.locate_plot_script = locate_plot_script_override_stub;
    hooks_.run_plot_script = run_plot_script_override_stub;
  }

  ~ScopedPlotScriptOverrides() { plot_script_override_state() = nullptr; }

  auto hooks() noexcept -> TracePlotRuntimeHooks& { return hooks_; }
  auto hooks() const noexcept -> const TracePlotRuntimeHooks& { return hooks_; }

 private:
  TracePlotRuntimeHooks hooks_{};
};

auto
merge_trace_plot_runtime_hooks(
    std::initializer_list<const TracePlotRuntimeHooks*> hook_sets)
    -> TracePlotRuntimeHooks
{
  TracePlotRuntimeHooks merged{};
  for (const auto* hooks : hook_sets) {
    if (hooks == nullptr) {
      continue;
    }
    if (hooks->resolve_python_candidates != nullptr) {
      merged.resolve_python_candidates = hooks->resolve_python_candidates;
    }
    if (hooks->resolve_python_is_regular_file != nullptr) {
      merged.resolve_python_is_regular_file =
          hooks->resolve_python_is_regular_file;
    }
    if (hooks->waitpid_nohang != nullptr) {
      merged.waitpid_nohang = hooks->waitpid_nohang;
    }
    if (hooks->waitpid_blocking != nullptr) {
      merged.waitpid_blocking = hooks->waitpid_blocking;
    }
    if (hooks->wait_for_plot_process_wait != nullptr) {
      merged.wait_for_plot_process_wait = hooks->wait_for_plot_process_wait;
    }
    if (hooks->terminate_and_wait != nullptr) {
      merged.terminate_and_wait = hooks->terminate_and_wait;
    }
    if (hooks->run_plot_script != nullptr) {
      merged.run_plot_script = hooks->run_plot_script;
    }
    if (hooks->run_plot_script_fork != nullptr) {
      merged.run_plot_script_fork = hooks->run_plot_script_fork;
    }
    if (hooks->locate_plot_script != nullptr) {
      merged.locate_plot_script = hooks->locate_plot_script;
    }
    if (hooks->trace_summary_file_path != nullptr) {
      merged.trace_summary_file_path = hooks->trace_summary_file_path;
    }
    if (hooks->candidate_plot_scripts_read_symlink != nullptr) {
      merged.candidate_plot_scripts_read_symlink =
          hooks->candidate_plot_scripts_read_symlink;
    }
    if (hooks->locate_plot_script_candidates != nullptr) {
      merged.locate_plot_script_candidates =
          hooks->locate_plot_script_candidates;
    }
  }
  return merged;
}

using ModelPreparationTuple = std::tuple<
    torch::jit::script::Module, std::vector<torch::jit::script::Module>,
    std::vector<torch::Tensor>>;

struct PrepareModelsAndWarmupOverrideState {
  std::optional<ModelPreparationTuple> load_result;
  bool throw_from_warmup = false;
  std::string warmup_exception_message = "warmup failure";
  bool append_gpu_model_in_warmup = false;
  int load_calls = 0;
  int warmup_calls = 0;
  const starpu_server::RuntimeConfig* load_opts = nullptr;
  const starpu_server::RuntimeConfig* warmup_opts = nullptr;
  starpu_server::StarPUSetup* warmup_starpu = nullptr;
  size_t warmup_gpu_model_count = 0;
  size_t warmup_reference_output_count = 0;
};

auto
prepare_models_and_warmup_override_state()
    -> PrepareModelsAndWarmupOverrideState*&
{
  static PrepareModelsAndWarmupOverrideState* state = nullptr;
  return state;
}

auto
load_model_and_reference_output_override_stub(
    const starpu_server::RuntimeConfig& opts)
    -> std::optional<ModelPreparationTuple>
{
  auto* state = prepare_models_and_warmup_override_state();
  if (state == nullptr) {
    return std::nullopt;
  }
  ++state->load_calls;
  state->load_opts = &opts;
  return std::move(state->load_result);
}

void
run_warmup_override_stub(
    const starpu_server::RuntimeConfig& opts,
    starpu_server::StarPUSetup& starpu,
    torch::jit::script::Module& /*model_cpu*/,
    std::vector<torch::jit::script::Module>& models_gpu,
    const std::vector<torch::Tensor>& reference_outputs)
{
  auto* state = prepare_models_and_warmup_override_state();
  if (state == nullptr) {
    return;
  }
  ++state->warmup_calls;
  state->warmup_opts = &opts;
  state->warmup_starpu = &starpu;
  state->warmup_gpu_model_count = models_gpu.size();
  state->warmup_reference_output_count = reference_outputs.size();
  if (state->append_gpu_model_in_warmup) {
    models_gpu.emplace_back("gpu_added_by_warmup");
  }
  if (state->throw_from_warmup) {
    throw std::runtime_error(state->warmup_exception_message);
  }
}

struct ScopedPrepareModelsAndWarmupOverrides {
  explicit ScopedPrepareModelsAndWarmupOverrides(
      PrepareModelsAndWarmupOverrideState& state) noexcept
  {
    prepare_models_and_warmup_override_state() = &state;
    load_model_and_reference_output_override_for_test() =
        load_model_and_reference_output_override_stub;
    run_warmup_override_for_test() = run_warmup_override_stub;
  }

  ~ScopedPrepareModelsAndWarmupOverrides()
  {
    run_warmup_override_for_test() = nullptr;
    load_model_and_reference_output_override_for_test() = nullptr;
    prepare_models_and_warmup_override_state() = nullptr;
  }
};

struct ScopedLoadModelAndReferenceOutputOverride {
  explicit ScopedLoadModelAndReferenceOutputOverride(
      PrepareModelsAndWarmupOverrideState& state) noexcept
  {
    prepare_models_and_warmup_override_state() = &state;
    load_model_and_reference_output_override_for_test() =
        load_model_and_reference_output_override_stub;
  }

  ~ScopedLoadModelAndReferenceOutputOverride()
  {
    load_model_and_reference_output_override_for_test() = nullptr;
    prepare_models_and_warmup_override_state() = nullptr;
  }
};

struct DescribeCpuAffinityOverrideState {
  hwloc_cpuset_t cpuset_to_return = nullptr;
  std::vector<int> cores;
  std::size_t next_index = 0;
  int provider_calls = 0;
  int first_calls = 0;
  int next_calls = 0;
  int free_calls = 0;
  int requested_worker_id = -1;
  hwloc_const_bitmap_t first_cpuset = nullptr;
  hwloc_const_bitmap_t next_cpuset = nullptr;
  hwloc_bitmap_t freed_cpuset = nullptr;
};

auto
describe_cpu_affinity_override_state() -> DescribeCpuAffinityOverrideState*&
{
  static DescribeCpuAffinityOverrideState* state = nullptr;
  return state;
}

auto
worker_cpuset_provider_override_stub(int worker_id) -> hwloc_cpuset_t
{
  auto* state = describe_cpu_affinity_override_state();
  if (state == nullptr) {
    return nullptr;
  }
  ++state->provider_calls;
  state->requested_worker_id = worker_id;
  state->next_index = 0;
  return state->cpuset_to_return;
}

auto
hwloc_bitmap_first_override_stub(hwloc_const_bitmap_t cpuset) -> int
{
  auto* state = describe_cpu_affinity_override_state();
  if (state == nullptr) {
    return -1;
  }
  ++state->first_calls;
  state->first_cpuset = cpuset;
  if (state->cores.empty()) {
    return -1;
  }
  state->next_index = 1;
  return state->cores.front();
}

auto
hwloc_bitmap_next_override_stub(
    hwloc_const_bitmap_t cpuset, int /*previous_core*/) -> int
{
  auto* state = describe_cpu_affinity_override_state();
  if (state == nullptr) {
    return -1;
  }
  ++state->next_calls;
  state->next_cpuset = cpuset;
  if (state->next_index >= state->cores.size()) {
    return -1;
  }
  return state->cores[state->next_index++];
}

void
hwloc_bitmap_free_override_stub(hwloc_bitmap_t cpuset)
{
  auto* state = describe_cpu_affinity_override_state();
  if (state == nullptr) {
    return;
  }
  ++state->free_calls;
  state->freed_cpuset = cpuset;
}

struct ScopedDescribeCpuAffinityOverrides {
  explicit ScopedDescribeCpuAffinityOverrides(
      DescribeCpuAffinityOverrideState& state) noexcept
  {
    describe_cpu_affinity_override_state() = &state;
    worker_cpuset_provider_override_for_test() =
        worker_cpuset_provider_override_stub;
    hwloc_bitmap_first_override_for_test() = hwloc_bitmap_first_override_stub;
    hwloc_bitmap_next_override_for_test() = hwloc_bitmap_next_override_stub;
    hwloc_bitmap_free_override_for_test() = hwloc_bitmap_free_override_stub;
  }

  ~ScopedDescribeCpuAffinityOverrides()
  {
    hwloc_bitmap_free_override_for_test() = nullptr;
    hwloc_bitmap_next_override_for_test() = nullptr;
    hwloc_bitmap_first_override_for_test() = nullptr;
    worker_cpuset_provider_override_for_test() = nullptr;
    describe_cpu_affinity_override_state() = nullptr;
  }
};

struct LogWorkerInventoryOverrideState {
  int worker_count = 0;
  std::vector<enum starpu_worker_archtype> worker_types;
  std::vector<int> device_ids;
  std::vector<std::string> cpu_affinities;
  int worker_count_calls = 0;
  int worker_type_calls = 0;
  int worker_device_id_calls = 0;
  int describe_cpu_affinity_calls = 0;
  std::vector<int> requested_worker_ids_for_type;
  std::vector<int> requested_worker_ids_for_device;
  std::vector<int> requested_worker_ids_for_affinity;
};

auto
log_worker_inventory_override_state() -> LogWorkerInventoryOverrideState*&
{
  static LogWorkerInventoryOverrideState* state = nullptr;
  return state;
}

auto
worker_count_override_stub() -> decltype(starpu_worker_get_count())
{
  auto* state = log_worker_inventory_override_state();
  if (state == nullptr) {
    return 0;
  }
  ++state->worker_count_calls;
  return static_cast<decltype(starpu_worker_get_count())>(state->worker_count);
}

auto
worker_type_override_stub(int worker_id)
    -> decltype(starpu_worker_get_type(worker_id))
{
  auto* state = log_worker_inventory_override_state();
  if (state == nullptr) {
    return STARPU_CPU_WORKER;
  }
  ++state->worker_type_calls;
  state->requested_worker_ids_for_type.push_back(worker_id);
  if (worker_id < 0 ||
      static_cast<std::size_t>(worker_id) >= state->worker_types.size()) {
    return STARPU_CPU_WORKER;
  }
  return state->worker_types[static_cast<std::size_t>(worker_id)];
}

auto
worker_device_id_override_stub(int worker_id)
    -> decltype(starpu_worker_get_devid(worker_id))
{
  auto* state = log_worker_inventory_override_state();
  if (state == nullptr) {
    return -1;
  }
  ++state->worker_device_id_calls;
  state->requested_worker_ids_for_device.push_back(worker_id);
  if (worker_id < 0 ||
      static_cast<std::size_t>(worker_id) >= state->device_ids.size()) {
    return -1;
  }
  return state->device_ids[static_cast<std::size_t>(worker_id)];
}

auto
describe_cpu_affinity_override_stub(int worker_id) -> std::string
{
  auto* state = log_worker_inventory_override_state();
  if (state == nullptr) {
    return {};
  }
  ++state->describe_cpu_affinity_calls;
  state->requested_worker_ids_for_affinity.push_back(worker_id);
  if (worker_id < 0 ||
      static_cast<std::size_t>(worker_id) >= state->cpu_affinities.size()) {
    return {};
  }
  return state->cpu_affinities[static_cast<std::size_t>(worker_id)];
}

struct ScopedLogWorkerInventoryOverrides {
  explicit ScopedLogWorkerInventoryOverrides(
      LogWorkerInventoryOverrideState& state) noexcept
  {
    log_worker_inventory_override_state() = &state;
    worker_count_override_for_test() = worker_count_override_stub;
    worker_type_override_for_test() = worker_type_override_stub;
    worker_device_id_override_for_test() = worker_device_id_override_stub;
    describe_cpu_affinity_override_for_test() =
        describe_cpu_affinity_override_stub;
  }

  ~ScopedLogWorkerInventoryOverrides()
  {
    describe_cpu_affinity_override_for_test() = nullptr;
    worker_device_id_override_for_test() = nullptr;
    worker_type_override_for_test() = nullptr;
    worker_count_override_for_test() = nullptr;
    log_worker_inventory_override_state() = nullptr;
  }
};

struct ScopedDescribeCpuAffinityInventoryOverride {
  explicit ScopedDescribeCpuAffinityInventoryOverride(
      DescribeCpuAffinityOverrideForTestFn override_fn) noexcept
      : previous_(describe_cpu_affinity_override_for_test())
  {
    describe_cpu_affinity_override_for_test() = override_fn;
  }

  ~ScopedDescribeCpuAffinityInventoryOverride()
  {
    describe_cpu_affinity_override_for_test() = previous_;
  }

 private:
  DescribeCpuAffinityOverrideForTestFn previous_ = nullptr;
};

auto
gpu_replication_worker_query_stub(
    unsigned int device_id, int* worker_ids,
    enum starpu_worker_archtype worker_type) -> int
{
  if (worker_type != STARPU_CUDA_WORKER || worker_ids == nullptr) {
    return 0;
  }
  if (device_id == 0U) {
    worker_ids[0] = 7;
    worker_ids[1] = 9;
    return 2;
  }
  if (device_id == 1U) {
    worker_ids[0] = 11;
    return 1;
  }
  return 0;
}

struct ScopedWorkerStreamQueryOverride {
  ScopedWorkerStreamQueryOverride()
  {
    starpu_server::StarPUSetup::set_worker_stream_query_fn(
        &gpu_replication_worker_query_stub);
  }

  ~ScopedWorkerStreamQueryOverride()
  {
    starpu_server::StarPUSetup::reset_worker_stream_query_fn();
  }
};

struct ScopedWorkerDeviceIdInventoryOverride {
  explicit ScopedWorkerDeviceIdInventoryOverride(
      WorkerDeviceIdOverrideForTestFn override_fn) noexcept
      : previous_(worker_device_id_override_for_test())
  {
    worker_device_id_override_for_test() = override_fn;
  }

  ~ScopedWorkerDeviceIdInventoryOverride()
  {
    worker_device_id_override_for_test() = previous_;
  }

 private:
  WorkerDeviceIdOverrideForTestFn previous_ = nullptr;
};

struct ScopedWorkerTypeInventoryOverride {
  explicit ScopedWorkerTypeInventoryOverride(
      WorkerTypeOverrideForTestFn override_fn) noexcept
      : previous_(worker_type_override_for_test())
  {
    worker_type_override_for_test() = override_fn;
  }

  ~ScopedWorkerTypeInventoryOverride()
  {
    worker_type_override_for_test() = previous_;
  }

 private:
  WorkerTypeOverrideForTestFn previous_ = nullptr;
};

struct ScopedWorkerCountInventoryOverride {
  explicit ScopedWorkerCountInventoryOverride(
      WorkerCountOverrideForTestFn override_fn) noexcept
      : previous_(worker_count_override_for_test())
  {
    worker_count_override_for_test() = override_fn;
  }

  ~ScopedWorkerCountInventoryOverride()
  {
    worker_count_override_for_test() = previous_;
  }

 private:
  WorkerCountOverrideForTestFn previous_ = nullptr;
};

struct ScopedBitmapFreeAffinityOverride {
  explicit ScopedBitmapFreeAffinityOverride(
      HwlocBitmapFreeOverrideForTestFn override_fn) noexcept
      : previous_(hwloc_bitmap_free_override_for_test())
  {
    hwloc_bitmap_free_override_for_test() = override_fn;
  }

  ~ScopedBitmapFreeAffinityOverride()
  {
    hwloc_bitmap_free_override_for_test() = previous_;
  }

 private:
  HwlocBitmapFreeOverrideForTestFn previous_ = nullptr;
};

struct ScopedBitmapNextAffinityOverride {
  explicit ScopedBitmapNextAffinityOverride(
      HwlocBitmapNextOverrideForTestFn override_fn) noexcept
      : previous_(hwloc_bitmap_next_override_for_test())
  {
    hwloc_bitmap_next_override_for_test() = override_fn;
  }

  ~ScopedBitmapNextAffinityOverride()
  {
    hwloc_bitmap_next_override_for_test() = previous_;
  }

 private:
  HwlocBitmapNextOverrideForTestFn previous_ = nullptr;
};

struct ScopedBitmapFirstAffinityOverride {
  explicit ScopedBitmapFirstAffinityOverride(
      HwlocBitmapFirstOverrideForTestFn override_fn) noexcept
      : previous_(hwloc_bitmap_first_override_for_test())
  {
    hwloc_bitmap_first_override_for_test() = override_fn;
  }

  ~ScopedBitmapFirstAffinityOverride()
  {
    hwloc_bitmap_first_override_for_test() = previous_;
  }

 private:
  HwlocBitmapFirstOverrideForTestFn previous_ = nullptr;
};

struct ScopedWorkerCpusetAffinityOverride {
  explicit ScopedWorkerCpusetAffinityOverride(
      WorkerCpusetProviderOverrideForTestFn override_fn) noexcept
      : previous_(worker_cpuset_provider_override_for_test())
  {
    worker_cpuset_provider_override_for_test() = override_fn;
  }

  ~ScopedWorkerCpusetAffinityOverride()
  {
    worker_cpuset_provider_override_for_test() = previous_;
  }

 private:
  WorkerCpusetProviderOverrideForTestFn previous_ = nullptr;
};

struct OccupiedLoopbackPort {
  int socket_fd = -1;
  int port = -1;

  OccupiedLoopbackPort() = default;
  OccupiedLoopbackPort(const OccupiedLoopbackPort&) = delete;
  auto operator=(const OccupiedLoopbackPort&) -> OccupiedLoopbackPort& = delete;
  OccupiedLoopbackPort(OccupiedLoopbackPort&& other) noexcept
      : socket_fd(other.socket_fd), port(other.port)
  {
    other.socket_fd = -1;
    other.port = -1;
  }
  auto operator=(OccupiedLoopbackPort&& other) noexcept -> OccupiedLoopbackPort&
  {
    if (this == &other) {
      return *this;
    }
    if (socket_fd >= 0) {
      ::close(socket_fd);
    }
    socket_fd = other.socket_fd;
    port = other.port;
    other.socket_fd = -1;
    other.port = -1;
    return *this;
  }
  ~OccupiedLoopbackPort()
  {
    if (socket_fd >= 0) {
      ::close(socket_fd);
    }
  }
};

struct HandleProgramArgumentsFatalOverrideState {
  std::string message;
  int call_count = 0;
};

auto
handle_program_arguments_fatal_override_state()
    -> HandleProgramArgumentsFatalOverrideState*&
{
  static HandleProgramArgumentsFatalOverrideState* state = nullptr;
  return state;
}

void
handle_program_arguments_fatal_override_stub(std::string_view message)
{
  auto* state = handle_program_arguments_fatal_override_state();
  if (state != nullptr) {
    ++state->call_count;
    state->message = std::string(message);
  }
  throw std::runtime_error(std::string(message));
}

void
handle_program_arguments_fatal_override_noop(std::string_view)
{
}

[[noreturn]] void
handle_program_arguments_fatal_with_exit_terminate_handler(
    std::string_view message)
{
  handle_program_arguments_fatal_override_for_test() = nullptr;
  std::set_terminate([] { std::exit(87); });
  handle_program_arguments_fatal(message);
}

[[noreturn]] void
handle_program_arguments_fatal_override_with_exit_terminate_handler(
    std::string_view message)
{
  handle_program_arguments_fatal_override_for_test() =
      handle_program_arguments_fatal_override_noop;
  std::set_terminate([] { std::exit(88); });
  handle_program_arguments_fatal(message);
}

struct ScopedHandleProgramArgumentsFatalOverride {
  explicit ScopedHandleProgramArgumentsFatalOverride(
      HandleProgramArgumentsFatalOverrideState& state) noexcept
  {
    handle_program_arguments_fatal_override_state() = &state;
    handle_program_arguments_fatal_override_for_test() =
        handle_program_arguments_fatal_override_stub;
  }

  ~ScopedHandleProgramArgumentsFatalOverride()
  {
    handle_program_arguments_fatal_override_for_test() = nullptr;
    handle_program_arguments_fatal_override_state() = nullptr;
  }
};

auto
fixture_model_path() -> std::filesystem::path
{
  const auto relative = std::filesystem::path(__FILE__).parent_path() / ".." /
                        ".." / "e2e" / "fixtures" / "simple_model.ts";
  std::error_code ec;
  const auto canonical = std::filesystem::weakly_canonical(relative, ec);
  if (!ec) {
    return canonical;
  }
  return relative;
}

auto
write_temp_config_file() -> TempFileGuard
{
  static std::atomic<std::uint64_t> temp_file_counter{0};
  const auto suffix = temp_file_counter.fetch_add(1, std::memory_order_relaxed);
  TempFileGuard guard{
      std::filesystem::temp_directory_path() /
      ("server_main_test_" + std::to_string(suffix) + ".yml")};

  std::ofstream cfg(guard.path);
  EXPECT_TRUE(cfg.is_open());

  const auto model_path = fixture_model_path();
  cfg << "name: unit_server_main\n";
  cfg << "model: " << model_path.string() << "\n";
  cfg << "inputs:\n";
  cfg << "  - { name: input0, data_type: TYPE_FP32, dims: [1, 2] }\n";
  cfg << "outputs:\n";
  cfg << "  - { name: output0, data_type: TYPE_FP32, dims: [1, 2] }\n";
  cfg << "pool_size: 1\n";
  cfg << "batching_strategy: adaptive\n";
  cfg << "adaptive_batching:\n";
  cfg << "  max_batch_size: 1\n";
  cfg << "batch_coalesce_timeout_ms: 0\n";
  cfg.close();

  EXPECT_TRUE(std::filesystem::exists(guard.path));
  return guard;
}

auto
pick_unused_port() -> int
{
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    return -1;
  }

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;

  if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    ::close(fd);
    return -1;
  }

  socklen_t addr_len = sizeof(addr);
  if (::getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &addr_len) != 0) {
    ::close(fd);
    return -1;
  }

  const int port = ntohs(addr.sin_port);
  ::close(fd);
  return port;
}

auto
occupy_loopback_port() -> OccupiedLoopbackPort
{
  OccupiedLoopbackPort occupied;
  occupied.socket_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (occupied.socket_fd < 0) {
    return occupied;
  }

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;
  if (::bind(
          occupied.socket_fd, reinterpret_cast<sockaddr*>(&addr),
          sizeof(addr)) != 0) {
    ::close(occupied.socket_fd);
    occupied.socket_fd = -1;
    return occupied;
  }
  if (::listen(occupied.socket_fd, 1) != 0) {
    ::close(occupied.socket_fd);
    occupied.socket_fd = -1;
    return occupied;
  }

  socklen_t addr_len = sizeof(addr);
  if (::getsockname(
          occupied.socket_fd, reinterpret_cast<sockaddr*>(&addr), &addr_len) !=
      0) {
    ::close(occupied.socket_fd);
    occupied.socket_fd = -1;
    return occupied;
  }
  occupied.port = ntohs(addr.sin_port);
  return occupied;
}

auto
wait_for_channel_ready(
    const std::string& address,
    std::chrono::system_clock::duration timeout) -> bool
{
  auto channel =
      grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
  return channel->WaitForConnected(std::chrono::system_clock::now() + timeout);
}

auto
make_temp_test_path(const std::string& stem, const std::string& extension)
    -> std::filesystem::path
{
  static std::atomic<std::uint64_t> temp_artifact_counter{0};
  const auto id = temp_artifact_counter.fetch_add(1, std::memory_order_relaxed);
  return std::filesystem::temp_directory_path() /
         (stem + "_" + std::to_string(id) + extension);
}

auto
make_runtime_config_for_starpu_setup() -> starpu_server::RuntimeConfig
{
  starpu_server::RuntimeConfig opts;
  opts.congestion.enabled = false;
  opts.batching.max_queue_size = 8;
  return opts;
}

auto
spawn_exiting_child(int exit_code) -> pid_t
{
  const pid_t pid = ::fork();
  if (pid == 0) {
    _exit(exit_code);
  }
  return pid;
}

auto
spawn_sleeping_child(bool ignore_sigterm) -> pid_t
{
  int ready_pipe[2]{-1, -1};
  if (::pipe(ready_pipe) != 0) {
    return -1;
  }

  const pid_t pid = ::fork();
  if (pid == 0) {
    ::close(ready_pipe[0]);
    if (ignore_sigterm) {
      sigset_t blocked{};
      ::sigemptyset(&blocked);
      ::sigaddset(&blocked, SIGTERM);
      (void)::sigprocmask(SIG_BLOCK, &blocked, nullptr);
    }
    const char ready = '1';
    (void)::write(ready_pipe[1], &ready, 1);
    ::close(ready_pipe[1]);
    while (true) {
      ::pause();
    }
  }
  ::close(ready_pipe[1]);
  if (pid < 0) {
    ::close(ready_pipe[0]);
    return -1;
  }
  char ready = 0;
  const auto bytes_read = ::read(ready_pipe[0], &ready, 1);
  ::close(ready_pipe[0]);
  if (bytes_read != 1) {
    (void)::kill(pid, SIGKILL);
    (void)wait_for_exit_blocking(pid);
    return -1;
  }
  return pid;
}

TEST(ServerMainArgs, HandleProgramArgumentsParsesLongConfigFlag)
{
  auto config_guard = write_temp_config_file();

  const std::string arg0 = "starpu_server";
  const std::string arg1 = "--config";
  const std::string arg2 = config_guard.path.string();
  const std::array<const char*, 3> argv{
      arg0.c_str(), arg1.c_str(), arg2.c_str()};

  auto cfg = handle_program_arguments({argv.data(), argv.size()});

  EXPECT_TRUE(cfg.valid);
  EXPECT_EQ(cfg.name, "unit_server_main");
  ASSERT_TRUE(cfg.model.has_value());
  EXPECT_EQ(cfg.model->inputs.size(), 1U);
  EXPECT_EQ(cfg.model->outputs.size(), 1U);
  EXPECT_EQ(cfg.batching.resolved_max_batch_size, 1);
}

TEST(ServerMainArgs, HandleProgramArgumentsParsesShortConfigFlag)
{
  auto config_guard = write_temp_config_file();

  const std::string arg0 = "starpu_server";
  const std::string arg1 = "-c";
  const std::string arg2 = config_guard.path.string();
  const std::array<const char*, 3> argv{
      arg0.c_str(), arg1.c_str(), arg2.c_str()};

  auto cfg = handle_program_arguments({argv.data(), argv.size()});

  EXPECT_TRUE(cfg.valid);
  EXPECT_EQ(cfg.name, "unit_server_main");
}

TEST(
    ServerMainArgs,
    HandleProgramArgumentsReportsMissingConfigValueWhenRemainingIsEmpty)
{
  HandleProgramArgumentsFatalOverrideState fatal_state;
  ScopedHandleProgramArgumentsFatalOverride fatal_override(fatal_state);
  const std::array<const char*, 2> argv{"starpu_server", "--config"};

  EXPECT_THROW(
      { (void)handle_program_arguments({argv.data(), argv.size()}); },
      std::runtime_error);

  EXPECT_EQ(fatal_state.call_count, 1);
  EXPECT_EQ(fatal_state.message, "Missing value for --config argument.\n");
}

TEST(
    ServerMainArgs,
    HandleProgramArgumentsReportsMissingConfigValueWhenRemainingFrontIsNullptr)
{
  HandleProgramArgumentsFatalOverrideState fatal_state;
  ScopedHandleProgramArgumentsFatalOverride fatal_override(fatal_state);
  const std::array<const char*, 3> argv{"starpu_server", "--config", nullptr};

  EXPECT_THROW(
      { (void)handle_program_arguments({argv.data(), argv.size()}); },
      std::runtime_error);

  EXPECT_EQ(fatal_state.call_count, 1);
  EXPECT_EQ(fatal_state.message, "Missing value for --config argument.\n");
}

TEST(ServerMainArgs, HandleProgramArgumentsReportsUnexpectedNullRawArgument)
{
  HandleProgramArgumentsFatalOverrideState fatal_state;
  ScopedHandleProgramArgumentsFatalOverride fatal_override(fatal_state);
  const std::array<const char*, 2> argv{"starpu_server", nullptr};

  EXPECT_THROW(
      { (void)handle_program_arguments({argv.data(), argv.size()}); },
      std::runtime_error);

  EXPECT_EQ(fatal_state.call_count, 1);
  EXPECT_EQ(fatal_state.message, "Unexpected null program argument.\n");
}

TEST(ServerMainArgs, HandleProgramArgumentsReportsMissingRequiredConfigPath)
{
  HandleProgramArgumentsFatalOverrideState fatal_state;
  ScopedHandleProgramArgumentsFatalOverride fatal_override(fatal_state);
  const std::array<const char*, 1> argv{"starpu_server"};

  EXPECT_THROW(
      { (void)handle_program_arguments({argv.data(), argv.size()}); },
      std::runtime_error);

  EXPECT_EQ(fatal_state.call_count, 1);
  EXPECT_EQ(fatal_state.message, "Missing required --config argument.\n");
}

TEST(ServerMainArgs, HandleProgramArgumentsReportsInvalidLoadedConfig)
{
  HandleProgramArgumentsFatalOverrideState fatal_state;
  ScopedHandleProgramArgumentsFatalOverride fatal_override(fatal_state);
  const auto missing_config =
      make_temp_test_path("server_main_missing_config_non_death", ".yaml");
  std::error_code remove_ec;
  std::filesystem::remove(missing_config, remove_ec);
  ASSERT_TRUE(remove_ec.value() == 0 || remove_ec.value() == ENOENT);

  const std::string config_path = missing_config.string();
  const std::array<const char*, 3> argv{
      "starpu_server", "--config", config_path.c_str()};

  OStreamCapture capture_err(std::cerr);
  EXPECT_THROW(
      { (void)handle_program_arguments({argv.data(), argv.size()}); },
      std::runtime_error);

  EXPECT_EQ(fatal_state.call_count, 1);
  EXPECT_EQ(fatal_state.message, "Invalid configuration file.\n");
  EXPECT_NE(capture_err.str().find("Failed to load config"), std::string::npos);
}

TEST(
    ServerMainArgs, HandleProgramArgumentsReportsUnknownArgumentWithFullMessage)
{
  HandleProgramArgumentsFatalOverrideState fatal_state;
  ScopedHandleProgramArgumentsFatalOverride fatal_override(fatal_state);
  const std::array<const char*, 2> argv{"starpu_server", "--unknown"};

  EXPECT_THROW(
      { (void)handle_program_arguments({argv.data(), argv.size()}); },
      std::runtime_error);

  EXPECT_EQ(fatal_state.call_count, 1);
  EXPECT_EQ(
      fatal_state.message,
      "Unknown argument '--unknown'. Only --config/-c is supported; all other "
      "settings must live in the YAML file.\n");
}

TEST(ServerMainArgs, HandleProgramArgumentsDiesWhenConfigValueIsMissing)
{
  const std::array<const char*, 2> argv{"starpu_server", "--config"};
  EXPECT_DEATH(
      { (void)handle_program_arguments({argv.data(), argv.size()}); },
      "Missing value for --config argument\\.");
}

TEST(ServerMainArgs, HandleProgramArgumentsDiesWhenConfigValueIsNullptr)
{
  const std::array<const char*, 3> argv{"starpu_server", "--config", nullptr};
  EXPECT_DEATH(
      { (void)handle_program_arguments({argv.data(), argv.size()}); },
      "Missing value for --config argument\\.");
}

TEST(ServerMainArgs, HandleProgramArgumentsDiesOnUnexpectedNullRawArgument)
{
  const std::array<const char*, 2> argv{"starpu_server", nullptr};
  EXPECT_DEATH(
      { (void)handle_program_arguments({argv.data(), argv.size()}); },
      "Unexpected null program argument\\.");
}

TEST(ServerMainArgs, HandleProgramArgumentsDiesOnUnknownArgument)
{
  const std::array<const char*, 2> argv{"starpu_server", "--unknown"};
  EXPECT_DEATH(
      { (void)handle_program_arguments({argv.data(), argv.size()}); },
      "Unknown argument '--unknown'");
}

TEST(ServerMainArgs, HandleProgramArgumentsDiesWhenConfigFlagIsMissing)
{
  const std::array<const char*, 1> argv{"starpu_server"};
  EXPECT_DEATH(
      { (void)handle_program_arguments({argv.data(), argv.size()}); },
      "Missing required --config argument\\.");
}

TEST(ServerMainArgs, HandleProgramArgumentsDiesWhenLoadedConfigIsInvalid)
{
  const auto missing_config =
      make_temp_test_path("server_main_missing_config", ".yaml");
  std::error_code remove_ec;
  std::filesystem::remove(missing_config, remove_ec);
  ASSERT_TRUE(remove_ec.value() == 0 || remove_ec.value() == ENOENT);

  const std::string config_path = missing_config.string();
  const std::array<const char*, 3> argv{
      "starpu_server", "--config", config_path.c_str()};
  EXPECT_DEATH(
      { (void)handle_program_arguments({argv.data(), argv.size()}); },
      "Invalid configuration file\\.");
}

TEST(
    ServerMainArgs,
    HandleProgramArgumentsFatalLogsFatalMessageWhenOverrideIsNotInstalled)
{
  EXPECT_EXIT(
      handle_program_arguments_fatal_with_exit_terminate_handler(
          "fatal without override"),
      ::testing::ExitedWithCode(87), "fatal without override");
}

TEST(
    ServerMainArgs,
    HandleProgramArgumentsFatalTerminatesAfterReturningTestOverride)
{
  EXPECT_EXIT(
      handle_program_arguments_fatal_override_with_exit_terminate_handler(
          "fatal through override"),
      ::testing::ExitedWithCode(88), ".*");
}

TEST(ServerMainSignal, SignalHandlerSetsStopFlag)
{
  signal_stop_requested_flag() = 0;
  signal_handler(SIGINT);
  EXPECT_EQ(signal_stop_requested_flag(), 1);
  signal_stop_requested_flag() = 0;
}

TEST(ServerMainThreadExceptionState, CaptureStoresFirstExceptionAndThreadName)
{
  ThreadExceptionState state;
  state.capture(
      "grpc-server",
      std::make_exception_ptr(std::runtime_error("grpc thread failure")));

  auto [captured_exception, thread_name] = state.take();
  ASSERT_NE(captured_exception, nullptr);
  EXPECT_EQ(thread_name, "grpc-server");

  try {
    std::rethrow_exception(captured_exception);
    FAIL() << "Expected runtime_error to be rethrown.";
  }
  catch (const std::runtime_error& error) {
    EXPECT_EQ(std::string(error.what()), "grpc thread failure");
  }
  catch (...) {
    FAIL() << "Expected std::runtime_error.";
  }
}

TEST(
    ServerMainThreadExceptionState,
    CaptureKeepsFirstExceptionWhenCalledMultipleTimes)
{
  ThreadExceptionState state;
  state.capture(
      "signal-notifier",
      std::make_exception_ptr(std::logic_error("first failure")));
  state.capture(
      "starpu-worker",
      std::make_exception_ptr(std::runtime_error("second failure")));

  auto [captured_exception, thread_name] = state.take();
  ASSERT_NE(captured_exception, nullptr);
  EXPECT_EQ(thread_name, "signal-notifier");

  try {
    std::rethrow_exception(captured_exception);
    FAIL() << "Expected logic_error to be rethrown.";
  }
  catch (const std::logic_error& error) {
    EXPECT_EQ(std::string(error.what()), "first failure");
  }
  catch (...) {
    FAIL() << "Expected std::logic_error.";
  }
}

TEST(
    ServerMainThreadExceptionState,
    RethrowThreadExceptionIfAnyWrapsStdExceptionWithThreadName)
{
  ThreadExceptionState state;
  state.capture(
      "grpc-server",
      std::make_exception_ptr(std::runtime_error("grpc thread failure")));

  try {
    rethrow_thread_exception_if_any(state);
    FAIL() << "Expected std::runtime_error.";
  }
  catch (const std::runtime_error& error) {
    EXPECT_EQ(
        std::string(error.what()),
        "Thread 'grpc-server' terminated with exception: grpc thread failure");
  }
  catch (...) {
    FAIL() << "Expected std::runtime_error.";
  }
}

TEST(
    ServerMainThreadExceptionState,
    RethrowThreadExceptionIfAnyWrapsNonStdExceptionWithUnknownMessage)
{
  ThreadExceptionState state;
  state.capture("signal-notifier", std::make_exception_ptr(42));

  try {
    rethrow_thread_exception_if_any(state);
    FAIL() << "Expected std::runtime_error.";
  }
  catch (const std::runtime_error& error) {
    EXPECT_EQ(
        std::string(error.what()),
        "Thread 'signal-notifier' terminated with unknown exception.");
  }
  catch (...) {
    FAIL() << "Expected std::runtime_error.";
  }
}

TEST(
    ServerMainThreadEntry,
    CapturesStdExceptionRequestsStopAndShutsDownQueueWhenQueueProvided)
{
  ThreadExceptionState state;
  ServerContext server_ctx;
  starpu_server::InferenceQueue queue(4);
  OStreamCapture capture_err(std::cerr);
  ASSERT_FALSE(queue.is_shutdown());
  EXPECT_FALSE(server_ctx.stop_requested.load(std::memory_order_relaxed));

  EXPECT_NO_THROW(run_thread_entry_with_exception_capture(
      "batching-thread", state, server_ctx, &queue,
      []() { throw std::runtime_error("std failure"); }));

  EXPECT_TRUE(server_ctx.stop_requested.load(std::memory_order_relaxed));
  EXPECT_TRUE(queue.is_shutdown());

  auto [captured_exception, thread_name] = state.take();
  ASSERT_NE(captured_exception, nullptr);
  EXPECT_EQ(thread_name, "batching-thread");
  try {
    std::rethrow_exception(captured_exception);
    FAIL() << "Expected std::runtime_error to be rethrown.";
  }
  catch (const std::runtime_error& error) {
    EXPECT_EQ(std::string(error.what()), "std failure");
  }
  catch (...) {
    FAIL() << "Expected std::runtime_error.";
  }

  EXPECT_NE(
      capture_err.str().find(
          "Unhandled exception escaped 'batching-thread' thread: std failure"),
      std::string::npos);
}

TEST(
    ServerMainThreadEntry,
    CapturesNonStdExceptionRequestsStopAndShutsDownQueueWhenQueueProvided)
{
  ThreadExceptionState state;
  ServerContext server_ctx;
  starpu_server::InferenceQueue queue(4);
  OStreamCapture capture_err(std::cerr);
  ASSERT_FALSE(queue.is_shutdown());
  EXPECT_FALSE(server_ctx.stop_requested.load(std::memory_order_relaxed));

  EXPECT_NO_THROW(run_thread_entry_with_exception_capture(
      "signal-thread", state, server_ctx, &queue, []() { throw 123; }));

  EXPECT_TRUE(server_ctx.stop_requested.load(std::memory_order_relaxed));
  EXPECT_TRUE(queue.is_shutdown());

  auto [captured_exception, thread_name] = state.take();
  ASSERT_NE(captured_exception, nullptr);
  EXPECT_EQ(thread_name, "signal-thread");
  try {
    std::rethrow_exception(captured_exception);
    FAIL() << "Expected non-standard exception to be rethrown.";
  }
  catch (int value) {
    EXPECT_EQ(value, 123);
  }
  catch (...) {
    FAIL() << "Expected integer exception.";
  }

  EXPECT_NE(
      capture_err.str().find(
          "Unhandled non-standard exception escaped 'signal-thread' thread."),
      std::string::npos);
}

TEST(
    ServerMainThreadEntry,
    CapturesStdExceptionInvokesExceptionHookAndMarksServerStopped)
{
  ThreadExceptionState state;
  ServerContext server_ctx;
  starpu_server::InferenceQueue queue(4);
  std::atomic<bool> hook_called{false};
  OStreamCapture capture_err(std::cerr);
  ASSERT_FALSE(queue.is_shutdown());
  EXPECT_FALSE(server_ctx.stop_requested.load(std::memory_order_relaxed));
  {
    std::lock_guard<std::mutex> lock(server_ctx.server_mutex);
    EXPECT_FALSE(server_ctx.server_startup_observed);
    EXPECT_EQ(server_ctx.server, nullptr);
  }

  EXPECT_NO_THROW(run_thread_entry_with_exception_capture(
      "grpc-server", state, server_ctx, &queue,
      []() { throw std::runtime_error("grpc startup failure"); },
      [&server_ctx, &hook_called]() {
        hook_called.store(true, std::memory_order_relaxed);
        mark_server_stopped(server_ctx);
      }));

  EXPECT_TRUE(hook_called.load(std::memory_order_relaxed));
  EXPECT_TRUE(server_ctx.stop_requested.load(std::memory_order_relaxed));
  EXPECT_TRUE(queue.is_shutdown());
  {
    std::lock_guard<std::mutex> lock(server_ctx.server_mutex);
    EXPECT_TRUE(server_ctx.server_startup_observed);
    EXPECT_EQ(server_ctx.server, nullptr);
  }

  auto [captured_exception, thread_name] = state.take();
  ASSERT_NE(captured_exception, nullptr);
  EXPECT_EQ(thread_name, "grpc-server");
  try {
    std::rethrow_exception(captured_exception);
    FAIL() << "Expected std::runtime_error to be rethrown.";
  }
  catch (const std::runtime_error& error) {
    EXPECT_EQ(std::string(error.what()), "grpc startup failure");
  }
  catch (...) {
    FAIL() << "Expected std::runtime_error.";
  }

  EXPECT_NE(
      capture_err.str().find(
          "Unhandled exception escaped 'grpc-server' thread: grpc startup "
          "failure"),
      std::string::npos);
}

TEST(
    ServerMainThreadEntry,
    CapturesStdExceptionLogsStdExceptionThrownByExceptionHook)
{
  ThreadExceptionState state;
  ServerContext server_ctx;
  starpu_server::InferenceQueue queue(4);
  OStreamCapture capture_err(std::cerr);

  EXPECT_NO_THROW(run_thread_entry_with_exception_capture(
      "grpc-server", state, server_ctx, &queue,
      []() { throw std::runtime_error("grpc startup failure"); },
      []() { throw std::logic_error("hook failure"); }));

  EXPECT_TRUE(server_ctx.stop_requested.load(std::memory_order_relaxed));
  EXPECT_TRUE(queue.is_shutdown());

  auto [captured_exception, thread_name] = state.take();
  ASSERT_NE(captured_exception, nullptr);
  EXPECT_EQ(thread_name, "grpc-server");
  EXPECT_NE(
      capture_err.str().find(
          "Unhandled exception escaped 'grpc-server' exception hook: "
          "hook failure"),
      std::string::npos);
}

TEST(
    ServerMainThreadEntry,
    CapturesStdExceptionLogsNonStdExceptionThrownByExceptionHook)
{
  ThreadExceptionState state;
  ServerContext server_ctx;
  starpu_server::InferenceQueue queue(4);
  OStreamCapture capture_err(std::cerr);

  EXPECT_NO_THROW(run_thread_entry_with_exception_capture(
      "grpc-server", state, server_ctx, &queue,
      []() { throw std::runtime_error("grpc startup failure"); },
      []() { throw 99; }));

  EXPECT_TRUE(server_ctx.stop_requested.load(std::memory_order_relaxed));
  EXPECT_TRUE(queue.is_shutdown());

  auto [captured_exception, thread_name] = state.take();
  ASSERT_NE(captured_exception, nullptr);
  EXPECT_EQ(thread_name, "grpc-server");
  EXPECT_NE(
      capture_err.str().find(
          "Unhandled non-standard exception escaped 'grpc-server' "
          "exception hook."),
      std::string::npos);
}

TEST(
    ServerMainModelPreparation,
    PrepareModelsAndWarmupThrowsModelLoadingExceptionWhenLoadingFails)
{
  auto opts = make_runtime_config_for_starpu_setup();
  starpu_server::StarPUSetup starpu(opts);

  PrepareModelsAndWarmupOverrideState override_state;
  ScopedPrepareModelsAndWarmupOverrides overrides(override_state);

  try {
    (void)prepare_models_and_warmup(opts, starpu);
    FAIL() << "Expected ModelLoadingException.";
  }
  catch (const starpu_server::ModelLoadingException& error) {
    EXPECT_EQ(
        std::string(error.what()), "Failed to load model or reference outputs");
  }
  catch (...) {
    FAIL() << "Expected starpu_server::ModelLoadingException.";
  }

  EXPECT_EQ(override_state.load_calls, 1);
  EXPECT_EQ(override_state.warmup_calls, 0);
  EXPECT_EQ(override_state.load_opts, &opts);
}

TEST(
    ServerMainModelPreparation,
    PrepareModelsAndWarmupRunsWarmupAndReturnsModels)
{
  auto opts = make_runtime_config_for_starpu_setup();
  starpu_server::StarPUSetup starpu(opts);

  PrepareModelsAndWarmupOverrideState override_state;
  override_state.load_result = ModelPreparationTuple{
      torch::jit::script::Module("cpu_model"),
      std::vector<torch::jit::script::Module>{
          torch::jit::script::Module("gpu_model_0")},
      std::vector<torch::Tensor>{
          torch::ones({2, 2}, torch::dtype(torch::kFloat32))}};
  override_state.append_gpu_model_in_warmup = true;
  ScopedPrepareModelsAndWarmupOverrides overrides(override_state);

  auto [model_cpu, models_gpu, reference_outputs] =
      prepare_models_and_warmup(opts, starpu);

  (void)model_cpu;
  EXPECT_EQ(override_state.load_calls, 1);
  EXPECT_EQ(override_state.warmup_calls, 1);
  EXPECT_EQ(override_state.load_opts, &opts);
  EXPECT_EQ(override_state.warmup_opts, &opts);
  EXPECT_EQ(override_state.warmup_starpu, &starpu);
  EXPECT_EQ(override_state.warmup_gpu_model_count, 1U);
  EXPECT_EQ(override_state.warmup_reference_output_count, 1U);
  EXPECT_EQ(models_gpu.size(), 2U);
  ASSERT_EQ(reference_outputs.size(), 1U);
  EXPECT_TRUE(torch::equal(
      reference_outputs[0],
      torch::ones({2, 2}, torch::dtype(torch::kFloat32))));
}

TEST(
    ServerMainModelPreparation, PrepareModelsAndWarmupPropagatesWarmupException)
{
  auto opts = make_runtime_config_for_starpu_setup();
  starpu_server::StarPUSetup starpu(opts);

  PrepareModelsAndWarmupOverrideState override_state;
  override_state.load_result = ModelPreparationTuple{
      torch::jit::script::Module("cpu_model"),
      std::vector<torch::jit::script::Module>{}, std::vector<torch::Tensor>{}};
  override_state.throw_from_warmup = true;
  override_state.warmup_exception_message = "forced warmup failure";
  ScopedPrepareModelsAndWarmupOverrides overrides(override_state);

  try {
    (void)prepare_models_and_warmup(opts, starpu);
    FAIL() << "Expected runtime_error from warmup.";
  }
  catch (const std::runtime_error& error) {
    EXPECT_EQ(std::string(error.what()), "forced warmup failure");
  }
  catch (...) {
    FAIL() << "Expected std::runtime_error.";
  }

  EXPECT_EQ(override_state.load_calls, 1);
  EXPECT_EQ(override_state.warmup_calls, 1);
}

TEST(
    ServerMainModelPreparation,
    PrepareModelsAndWarmupCallsRealLoadModelAndThrowsWhenLoadFails)
{
  auto opts = make_runtime_config_for_starpu_setup();
  opts.model = starpu_server::ModelConfig{};
  const auto missing_model =
      make_temp_test_path("prepare_models_and_warmup_missing_model", ".ts");
  std::error_code remove_ec;
  std::filesystem::remove(missing_model, remove_ec);
  ASSERT_TRUE(remove_ec.value() == 0 || remove_ec.value() == ENOENT);
  opts.model->path = missing_model.string();

  starpu_server::StarPUSetup starpu(opts);
  load_model_and_reference_output_override_for_test() = nullptr;
  run_warmup_override_for_test() = nullptr;

  OStreamCapture capture_err(std::cerr);
  EXPECT_THROW(
      (void)prepare_models_and_warmup(opts, starpu),
      starpu_server::ModelLoadingException);
  EXPECT_NE(
      capture_err.str().find("Failed to load model or run reference inference"),
      std::string::npos);
}

TEST(
    ServerMainModelPreparation,
    PrepareModelsAndWarmupCallsRealRunWarmupWhenOverrideIsUnset)
{
  auto opts = make_runtime_config_for_starpu_setup();
  opts.batching.warmup_request_nb = 0;
  opts.batching.warmup_batches_per_worker = 0;

  starpu_server::StarPUSetup starpu(opts);
  PrepareModelsAndWarmupOverrideState override_state;
  override_state.load_result = ModelPreparationTuple{
      torch::jit::script::Module("cpu_model"),
      std::vector<torch::jit::script::Module>{},
      std::vector<torch::Tensor>{
          torch::ones({1, 2}, torch::dtype(torch::kFloat32))}};
  ScopedLoadModelAndReferenceOutputOverride load_override(override_state);
  run_warmup_override_for_test() = nullptr;

  auto [model_cpu, models_gpu, reference_outputs] =
      prepare_models_and_warmup(opts, starpu);
  (void)model_cpu;

  EXPECT_EQ(override_state.load_calls, 1);
  EXPECT_EQ(override_state.load_opts, &opts);
  EXPECT_TRUE(models_gpu.empty());
  ASSERT_EQ(reference_outputs.size(), 1U);
  EXPECT_EQ(reference_outputs[0].sizes().size(), 2U);
  EXPECT_EQ(reference_outputs[0].size(0), 1);
  EXPECT_EQ(reference_outputs[0].size(1), 2);
}

TEST(ServerMainModelNameResolution, UsesModelNameWhenConfiguredAndNonEmpty)
{
  starpu_server::RuntimeConfig opts;
  opts.name = "fallback_name";
  opts.model = starpu_server::ModelConfig{};
  opts.model->name = "configured_model_name";

  EXPECT_EQ(resolve_default_model_name(opts), "configured_model_name");
}

TEST(ServerMainModelNameResolution, FallsBackToConfigNameWhenModelNameIsEmpty)
{
  starpu_server::RuntimeConfig opts;
  opts.name = "fallback_name";
  opts.model = starpu_server::ModelConfig{};
  opts.model->name = "";

  EXPECT_EQ(resolve_default_model_name(opts), "fallback_name");
}

TEST(ServerMainModelNameResolution, FallsBackToConfigNameWhenModelIsNotSet)
{
  starpu_server::RuntimeConfig opts;
  opts.name = "fallback_name";
  opts.model.reset();

  EXPECT_EQ(resolve_default_model_name(opts), "fallback_name");
}

TEST(ServerMainSchedulerResolution, UsesSchedulerFromStarpuEnvMap)
{
  starpu_server::RuntimeConfig opts;
  opts.starpu_env.emplace(
      std::string(starpu_server::kStarpuSchedulerEnvVar), "dmda");

  EXPECT_EQ(resolve_starpu_scheduler(opts), "dmda (from starpu_env)");
}

TEST(ServerMainSchedulerResolution, UsesDefaultSchedulerWhenUnsetEverywhere)
{
  ScopedEnvironmentVariableUnsetGuard unset_scheduler_env{
      std::string(starpu_server::kStarpuSchedulerEnvVar)};
  starpu_server::RuntimeConfig opts;
  opts.starpu_env.clear();

  EXPECT_EQ(
      resolve_starpu_scheduler(opts),
      std::format("{} (default)", starpu_server::kDefaultStarpuScheduler));
}

TEST(ServerMainLibtorchSettings, AppliesConfiguredThreadCounts)
{
  LibtorchThreadOverrideState state;
  const ScopedLibtorchThreadOverrides overrides(state);

  starpu_server::RuntimeConfig opts;
  opts.libtorch.intraop_threads = 2;
  opts.libtorch.interop_threads = 3;

  apply_libtorch_runtime_settings(opts);

  EXPECT_EQ(state.intraop_calls, 1);
  EXPECT_EQ(state.interop_calls, 1);
  ASSERT_TRUE(state.last_intraop.has_value());
  ASSERT_TRUE(state.last_interop.has_value());
  EXPECT_EQ(*state.last_intraop, 2);
  EXPECT_EQ(*state.last_interop, 3);
}

TEST(ServerMainLibtorchSettings, SkipsUnsetThreadCounts)
{
  LibtorchThreadOverrideState state;
  const ScopedLibtorchThreadOverrides overrides(state);

  starpu_server::RuntimeConfig opts;

  apply_libtorch_runtime_settings(opts);

  EXPECT_EQ(state.intraop_calls, 0);
  EXPECT_EQ(state.interop_calls, 0);
  EXPECT_FALSE(state.last_intraop.has_value());
  EXPECT_FALSE(state.last_interop.has_value());
}

TEST(ServerMainWorkerTypeLabel, ReturnsCpuLabelForCpuWorker)
{
  EXPECT_EQ(worker_type_label(STARPU_CPU_WORKER), "CPU");
}

TEST(ServerMainWorkerTypeLabel, ReturnsCudaLabelForCudaWorker)
{
  EXPECT_EQ(worker_type_label(STARPU_CUDA_WORKER), "CUDA");
}

TEST(ServerMainWorkerTypeLabel, ReturnsOtherLabelForUnknownWorkerType)
{
  EXPECT_EQ(worker_type_label(STARPU_OPENCL_WORKER), "Other(2)");
}

TEST(ServerMainWorkerInventory, FormatIntListReturnsNoneWhenEmpty)
{
  EXPECT_EQ(format_int_list({}), "none");
}

TEST(ServerMainWorkerInventory, ResolveModelLabelDefaultsWithoutModel)
{
  starpu_server::RuntimeConfig opts;
  EXPECT_EQ(resolve_model_label_for_startup_metrics(opts), "default");
}

TEST(ServerMainWorkerInventory, ResolveModelLabelFallsBackToPathWhenNameEmpty)
{
  starpu_server::RuntimeConfig opts;
  opts.model = starpu_server::ModelConfig{};
  opts.model->path = "/tmp/model_for_startup_metrics.ts";

  EXPECT_EQ(
      resolve_model_label_for_startup_metrics(opts),
      "/tmp/model_for_startup_metrics.ts");
}

TEST(ServerMainCpuCoreRanges, ReturnsEmptyStringForEmptyInput)
{
  EXPECT_EQ(format_cpu_core_ranges({}), "");
}

TEST(ServerMainCpuCoreRanges, FormatsSingleCpuAsSingleValue)
{
  EXPECT_EQ(format_cpu_core_ranges({7}), "7");
}

TEST(ServerMainCpuCoreRanges, FormatsMixedContiguousAndSingleRanges)
{
  EXPECT_EQ(format_cpu_core_ranges({1, 2, 3, 5, 7, 8}), "1-3,5,7-8");
}

TEST(ServerMainCpuCoreRanges, FormatsMultipleSingleValuesWithCommas)
{
  EXPECT_EQ(format_cpu_core_ranges({2, 4, 6}), "2,4,6");
}

TEST(ServerMainCpuAffinity, ReturnsEmptyStringWhenCpusetIsNull)
{
  DescribeCpuAffinityOverrideState override_state;
  override_state.cpuset_to_return = nullptr;
  ScopedDescribeCpuAffinityOverrides overrides(override_state);

  EXPECT_EQ(describe_cpu_affinity(11), "");
  EXPECT_EQ(override_state.provider_calls, 1);
  EXPECT_EQ(override_state.requested_worker_id, 11);
  EXPECT_EQ(override_state.first_calls, 0);
  EXPECT_EQ(override_state.next_calls, 0);
  EXPECT_EQ(override_state.free_calls, 0);
}

TEST(ServerMainCpuAffinity, ReturnsEmptyStringWhenCpusetContainsNoCore)
{
  DescribeCpuAffinityOverrideState override_state;
  override_state.cpuset_to_return =
      reinterpret_cast<hwloc_cpuset_t>(static_cast<std::uintptr_t>(0x1));
  override_state.cores = {};
  ScopedDescribeCpuAffinityOverrides overrides(override_state);

  EXPECT_EQ(describe_cpu_affinity(3), "");
  EXPECT_EQ(override_state.provider_calls, 1);
  EXPECT_EQ(override_state.requested_worker_id, 3);
  EXPECT_EQ(override_state.first_calls, 1);
  EXPECT_EQ(override_state.next_calls, 0);
  EXPECT_EQ(override_state.free_calls, 1);
  EXPECT_EQ(override_state.first_cpuset, override_state.cpuset_to_return);
  EXPECT_EQ(override_state.freed_cpuset, override_state.cpuset_to_return);
}

TEST(ServerMainCpuAffinity, FormatsCoresAndFreesCpusetWhenCoresArePresent)
{
  DescribeCpuAffinityOverrideState override_state;
  override_state.cpuset_to_return =
      reinterpret_cast<hwloc_cpuset_t>(static_cast<std::uintptr_t>(0x2));
  override_state.cores = {0, 1, 2, 4, 6, 7};
  ScopedDescribeCpuAffinityOverrides overrides(override_state);

  EXPECT_EQ(describe_cpu_affinity(5), "0-2,4,6-7");
  EXPECT_EQ(override_state.provider_calls, 1);
  EXPECT_EQ(override_state.requested_worker_id, 5);
  EXPECT_EQ(override_state.first_calls, 1);
  EXPECT_EQ(override_state.next_calls, 6);
  EXPECT_EQ(override_state.free_calls, 1);
  EXPECT_EQ(override_state.first_cpuset, override_state.cpuset_to_return);
  EXPECT_EQ(override_state.next_cpuset, override_state.cpuset_to_return);
  EXPECT_EQ(override_state.freed_cpuset, override_state.cpuset_to_return);
}

TEST(ServerMainCpuAffinity, BitmapFreeForAffinityFallsBackToHwlocBitmapFree)
{
  ScopedBitmapFreeAffinityOverride free_override_guard(nullptr);

  hwloc_bitmap_t cpuset = hwloc_bitmap_alloc();
  ASSERT_NE(cpuset, nullptr);
  ASSERT_EQ(hwloc_bitmap_set(cpuset, 0), 0);

  EXPECT_NO_THROW(bitmap_free_for_affinity(cpuset));
}

TEST(ServerMainCpuAffinity, BitmapNextForAffinityFallsBackToHwlocBitmapNext)
{
  ScopedBitmapNextAffinityOverride next_override_guard(nullptr);

  hwloc_bitmap_t cpuset = hwloc_bitmap_alloc();
  ASSERT_NE(cpuset, nullptr);
  ASSERT_EQ(hwloc_bitmap_set(cpuset, 1), 0);
  ASSERT_EQ(hwloc_bitmap_set(cpuset, 4), 0);

  EXPECT_EQ(bitmap_next_for_affinity(cpuset, 1), 4);
  EXPECT_EQ(bitmap_next_for_affinity(cpuset, 4), -1);

  hwloc_bitmap_free(cpuset);
}

TEST(ServerMainCpuAffinity, BitmapFirstForAffinityFallsBackToHwlocBitmapFirst)
{
  ScopedBitmapFirstAffinityOverride first_override_guard(nullptr);

  hwloc_bitmap_t cpuset = hwloc_bitmap_alloc();
  ASSERT_NE(cpuset, nullptr);
  ASSERT_EQ(hwloc_bitmap_set(cpuset, 3), 0);
  ASSERT_EQ(hwloc_bitmap_set(cpuset, 7), 0);

  EXPECT_EQ(bitmap_first_for_affinity(cpuset), 3);

  hwloc_bitmap_free(cpuset);
}

TEST(
    ServerMainCpuAffinity,
    GetWorkerCpusetForAffinityFallsBackToStarpuWhenNoOverride)
{
  auto opts = make_runtime_config_for_starpu_setup();
  starpu_server::StarPUSetup starpu(opts);
  (void)starpu;

  ScopedWorkerCpusetAffinityOverride cpuset_override_guard(nullptr);

  const int worker_count = static_cast<int>(starpu_worker_get_count());
  if (worker_count <= 0) {
    GTEST_SKIP() << "No StarPU workers available for cpuset query";
  }

  constexpr int worker_id = 0;
  hwloc_cpuset_t expected = starpu_worker_get_hwloc_cpuset(worker_id);
  hwloc_cpuset_t actual = get_worker_cpuset_for_affinity(worker_id);

  if (expected == nullptr || actual == nullptr) {
    EXPECT_EQ(actual, expected);
    if (actual != nullptr) {
      hwloc_bitmap_free(actual);
    }
    if (expected != nullptr && expected != actual) {
      hwloc_bitmap_free(expected);
    }
    return;
  }

  EXPECT_EQ(hwloc_bitmap_isequal(actual, expected), 1);

  if (actual == expected) {
    hwloc_bitmap_free(actual);
  } else {
    hwloc_bitmap_free(actual);
    hwloc_bitmap_free(expected);
  }
}

TEST(
    ServerMainWorkerInventory,
    DescribeCpuAffinityForInventoryFallsBackToDescribeCpuAffinityWhenNoOverride)
{
  DescribeCpuAffinityOverrideState override_state;
  override_state.cpuset_to_return =
      reinterpret_cast<hwloc_cpuset_t>(static_cast<std::uintptr_t>(0x3));
  override_state.cores = {1, 2, 3, 8};
  ScopedDescribeCpuAffinityOverrides affinity_overrides(override_state);
  ScopedDescribeCpuAffinityInventoryOverride inventory_override_guard(nullptr);

  EXPECT_EQ(describe_cpu_affinity_for_inventory(13), "1-3,8");
  EXPECT_EQ(override_state.provider_calls, 1);
  EXPECT_EQ(override_state.requested_worker_id, 13);
  EXPECT_EQ(override_state.first_calls, 1);
  EXPECT_EQ(override_state.next_calls, 4);
  EXPECT_EQ(override_state.free_calls, 1);
  EXPECT_EQ(override_state.first_cpuset, override_state.cpuset_to_return);
  EXPECT_EQ(override_state.next_cpuset, override_state.cpuset_to_return);
  EXPECT_EQ(override_state.freed_cpuset, override_state.cpuset_to_return);
}

TEST(
    ServerMainWorkerInventory,
    WorkerDeviceIdForInventoryFallsBackToStarpuWhenNoOverride)
{
  auto opts = make_runtime_config_for_starpu_setup();
  starpu_server::StarPUSetup starpu(opts);
  (void)starpu;

  ScopedWorkerDeviceIdInventoryOverride override_guard(nullptr);

  const int worker_count = static_cast<int>(starpu_worker_get_count());
  if (worker_count <= 0) {
    GTEST_SKIP() << "No StarPU workers available for device id query";
  }

  constexpr int worker_id = 0;
  const int expected = starpu_worker_get_devid(worker_id);
  EXPECT_EQ(worker_device_id_for_inventory(worker_id), expected);
}

TEST(
    ServerMainWorkerInventory,
    WorkerTypeForInventoryFallsBackToStarpuWhenNoOverride)
{
  auto opts = make_runtime_config_for_starpu_setup();
  starpu_server::StarPUSetup starpu(opts);
  (void)starpu;

  ScopedWorkerTypeInventoryOverride override_guard(nullptr);

  const int worker_count = static_cast<int>(starpu_worker_get_count());
  if (worker_count <= 0) {
    GTEST_SKIP() << "No StarPU workers available for type query";
  }

  constexpr int worker_id = 0;
  const auto expected = starpu_worker_get_type(worker_id);
  EXPECT_EQ(worker_type_for_inventory(worker_id), expected);
}

TEST(
    ServerMainWorkerInventory,
    WorkerCountForInventoryFallsBackToStarpuWhenNoOverride)
{
  auto opts = make_runtime_config_for_starpu_setup();
  starpu_server::StarPUSetup starpu(opts);
  (void)starpu;

  ScopedWorkerCountInventoryOverride override_guard(nullptr);

  const int expected = static_cast<int>(starpu_worker_get_count());
  EXPECT_EQ(worker_count_for_inventory(), expected);
}

TEST(ServerMainWorkerInventory, LogsOnlyHeaderWhenNoWorkersAreConfigured)
{
  LogWorkerInventoryOverrideState override_state;
  override_state.worker_count = 0;
  ScopedLogWorkerInventoryOverrides overrides(override_state);

  starpu_server::RuntimeConfig opts;
  opts.verbosity = starpu_server::VerbosityLevel::Info;

  OStreamCapture capture_out(std::cout);
  log_worker_inventory(opts);

  EXPECT_EQ(
      capture_out.str(), expected_info_log("Configured 0 StarPU worker(s)."));
  EXPECT_EQ(override_state.worker_count_calls, 1);
  EXPECT_EQ(override_state.worker_type_calls, 0);
  EXPECT_EQ(override_state.worker_device_id_calls, 0);
  EXPECT_EQ(override_state.describe_cpu_affinity_calls, 0);
}

TEST(
    ServerMainWorkerInventory,
    LogsPerWorkerTypeDeviceAndCpuAffinityForMixedWorkers)
{
  LogWorkerInventoryOverrideState override_state;
  override_state.worker_count = 3;
  override_state.worker_types = {
      STARPU_CPU_WORKER, STARPU_CPU_WORKER, STARPU_CUDA_WORKER};
  override_state.device_ids = {0, -1, 2};
  override_state.cpu_affinities = {"0-3", "", "should_not_be_used"};
  ScopedLogWorkerInventoryOverrides overrides(override_state);

  starpu_server::RuntimeConfig opts;
  opts.verbosity = starpu_server::VerbosityLevel::Info;

  OStreamCapture capture_out(std::cout);
  log_worker_inventory(opts);

  const auto expected =
      expected_info_log("Configured 3 StarPU worker(s).") +
      expected_info_log("Worker  0: type=CPU, device id=0, cores=0-3") +
      expected_info_log("Worker  1: type=CPU, device id=N/A") +
      expected_info_log("Worker  2: type=CUDA, device id=2");
  EXPECT_EQ(capture_out.str(), expected);
  EXPECT_EQ(override_state.worker_count_calls, 1);
  EXPECT_EQ(override_state.worker_type_calls, 3);
  EXPECT_EQ(override_state.worker_device_id_calls, 3);
  EXPECT_EQ(override_state.describe_cpu_affinity_calls, 2);
  EXPECT_EQ(
      override_state.requested_worker_ids_for_type,
      std::vector<int>({0, 1, 2}));
  EXPECT_EQ(
      override_state.requested_worker_ids_for_device,
      std::vector<int>({0, 1, 2}));
  EXPECT_EQ(
      override_state.requested_worker_ids_for_affinity,
      std::vector<int>({0, 1}));
}

TEST(
    ServerMainWorkerInventory,
    ReportsGpuReplicationStartupSummaryAndMetricsForPerWorkerPolicy)
{
  starpu_server::shutdown_metrics();
  ASSERT_TRUE(starpu_server::init_metrics(0));
  struct MetricsGuard {
    ~MetricsGuard() { starpu_server::shutdown_metrics(); }
  } metrics_guard;

  ScopedWorkerStreamQueryOverride worker_query_guard;

  starpu_server::RuntimeConfig opts;
  opts.verbosity = starpu_server::VerbosityLevel::Info;
  opts.devices.use_cuda = true;
  opts.devices.ids = {0, 1};
  opts.devices.gpu_model_replication =
      starpu_server::GpuModelReplicationPolicy::PerWorker;
  opts.model = starpu_server::ModelConfig{};
  opts.model->name = "resnet152";

  OStreamCapture capture_out(std::cout);
  report_gpu_replication_startup(opts, 3);

  const auto expected =
      expected_info_log(
          "GPU model replication summary: policy=per_worker, "
          "total_replicas=3, configured_cuda_devices=2.") +
      expected_info_log("CUDA device 0 -> workers=[7,9], model replicas=2") +
      expected_info_log("CUDA device 1 -> workers=[11], model replicas=1");
  EXPECT_EQ(capture_out.str(), expected);

  const auto metrics = starpu_server::get_metrics();
  ASSERT_NE(metrics, nullptr);
  const auto families = metrics->registry()->Collect();

  const auto policy_value = FindGaugeValue(
      families, "gpu_model_replication_policy_info",
      {{"model", "resnet152"}, {"policy", "per_worker"}});
  ASSERT_TRUE(policy_value.has_value());
  EXPECT_DOUBLE_EQ(*policy_value, 1.0);

  const auto replica_total = FindGaugeValue(
      families, "gpu_model_replicas_total", {{"model", "resnet152"}});
  ASSERT_TRUE(replica_total.has_value());
  EXPECT_DOUBLE_EQ(*replica_total, 3.0);

  const auto worker_7 = FindGaugeValue(
      families, "starpu_cuda_worker_info",
      {{"device", "0"}, {"worker_id", "7"}});
  ASSERT_TRUE(worker_7.has_value());
  EXPECT_DOUBLE_EQ(*worker_7, 1.0);

  const auto worker_9 = FindGaugeValue(
      families, "starpu_cuda_worker_info",
      {{"device", "0"}, {"worker_id", "9"}});
  ASSERT_TRUE(worker_9.has_value());
  EXPECT_DOUBLE_EQ(*worker_9, 1.0);

  const auto worker_11 = FindGaugeValue(
      families, "starpu_cuda_worker_info",
      {{"device", "1"}, {"worker_id", "11"}});
  ASSERT_TRUE(worker_11.has_value());
  EXPECT_DOUBLE_EQ(*worker_11, 1.0);
}

TEST(
    ServerMainWorkerInventory,
    ReportsGpuReplicationStartupReturnsWhenCudaDisabled)
{
  starpu_server::RuntimeConfig opts;
  opts.verbosity = starpu_server::VerbosityLevel::Silent;
  opts.devices.use_cuda = false;
  opts.devices.ids = {0, 1};

  OStreamCapture capture_out(std::cout);
  report_gpu_replication_startup(opts, 3);

  EXPECT_TRUE(capture_out.str().empty());
}

TEST(
    ServerMainWorkerInventory,
    ReportsGpuReplicationStartupUsesObservabilityMetrics)
{
  ScopedWorkerStreamQueryOverride worker_query_guard;

  auto metrics = starpu_server::create_metrics_recorder(0);
  ASSERT_NE(metrics, nullptr);
  ASSERT_TRUE(metrics->enabled());

  auto observability = std::make_shared<starpu_server::RuntimeObservability>();
  observability->metrics = metrics;

  starpu_server::RuntimeConfig opts;
  opts.verbosity = starpu_server::VerbosityLevel::Info;
  opts.devices.use_cuda = true;
  opts.devices.ids = {0, 1};
  opts.devices.gpu_model_replication =
      starpu_server::GpuModelReplicationPolicy::PerWorker;
  opts.model = starpu_server::ModelConfig{};
  opts.model->path = "/tmp/resnet50_observability.ts";

  OStreamCapture capture_out(std::cout);
  report_gpu_replication_startup(opts, 3, observability);

  const auto expected =
      expected_info_log(
          "GPU model replication summary: policy=per_worker, "
          "total_replicas=3, configured_cuda_devices=2.") +
      expected_info_log("CUDA device 0 -> workers=[7,9], model replicas=2") +
      expected_info_log("CUDA device 1 -> workers=[11], model replicas=1");
  EXPECT_EQ(capture_out.str(), expected);

  const auto families = metrics->registry()->registry()->Collect();

  const auto policy_value = FindGaugeValue(
      families, "gpu_model_replication_policy_info",
      {{"model", "/tmp/resnet50_observability.ts"}, {"policy", "per_worker"}});
  ASSERT_TRUE(policy_value.has_value());
  EXPECT_DOUBLE_EQ(*policy_value, 1.0);

  const auto replica_total = FindGaugeValue(
      families, "gpu_model_replicas_total",
      {{"model", "/tmp/resnet50_observability.ts"}});
  ASSERT_TRUE(replica_total.has_value());
  EXPECT_DOUBLE_EQ(*replica_total, 3.0);

  const auto worker_7 = FindGaugeValue(
      families, "starpu_cuda_worker_info",
      {{"device", "0"}, {"worker_id", "7"}});
  ASSERT_TRUE(worker_7.has_value());
  EXPECT_DOUBLE_EQ(*worker_7, 1.0);

  const auto worker_9 = FindGaugeValue(
      families, "starpu_cuda_worker_info",
      {{"device", "0"}, {"worker_id", "9"}});
  ASSERT_TRUE(worker_9.has_value());
  EXPECT_DOUBLE_EQ(*worker_9, 1.0);

  const auto worker_11 = FindGaugeValue(
      families, "starpu_cuda_worker_info",
      {{"device", "1"}, {"worker_id", "11"}});
  ASSERT_TRUE(worker_11.has_value());
  EXPECT_DOUBLE_EQ(*worker_11, 1.0);
}

TEST(
    ServerMainWorkerInventory,
    ReportsGpuReplicationStartupWarnsWhenAssignmentsThrow)
{
  ScopedWorkerStreamQueryOverride worker_query_guard;

  starpu_server::RuntimeConfig opts;
  opts.verbosity = starpu_server::VerbosityLevel::Info;
  opts.devices.use_cuda = true;
  opts.devices.ids = {0, -1, 1};
  opts.devices.gpu_model_replication =
      starpu_server::GpuModelReplicationPolicy::PerWorker;

  OStreamCapture capture_err(std::cerr);
  report_gpu_replication_startup(opts, 3);

  EXPECT_NE(
      capture_err.str().find(
          "Failed to summarize GPU model replication startup state"),
      std::string::npos);
}

TEST(
    ServerMainShutdownRuntime,
    SetupRuntimeStateCreatesObservabilityMonitorWhenProvided)
{
  starpu_server::RuntimeConfig opts;
  opts.congestion.enabled = false;

  ServerContext ctx;
  ctx.stop_requested.store(true, std::memory_order_relaxed);
  mark_server_stopped(ctx);
  starpu_server::InferenceQueue queue(4);
  auto observability = std::make_shared<starpu_server::RuntimeObservability>();

  setup_runtime_state(opts, ctx, queue, observability);

  ASSERT_NE(observability->congestion_monitor, nullptr);
  EXPECT_FALSE(ctx.stop_requested.load(std::memory_order_relaxed));
  EXPECT_EQ(signal_stop_requested_flag(), 0);

  observability->congestion_monitor->shutdown();
  observability->congestion_monitor.reset();
}

TEST(ServerMainShutdownRuntime, RunShutdownSequenceRejectsIncompleteContext)
{
  starpu_server::RuntimeConfig opts;
  ServerContext ctx;
  starpu_server::InferenceQueue queue(4);

  EXPECT_THROW(
      run_shutdown_sequence(opts, ctx, queue, ShutdownRuntimeContext{}, {}),
      std::invalid_argument);
}

TEST(
    ServerMainShutdownRuntime,
    RunShutdownSequenceShutsDownInjectedCongestionMonitor)
{
  starpu_server::RuntimeConfig opts;
  ServerContext ctx;
  ctx.stop_requested.store(true, std::memory_order_relaxed);
  mark_server_stopped(ctx);

  starpu_server::InferenceQueue queue(4);
  std::atomic<std::size_t> completed_jobs{0};
  std::condition_variable all_done_cv;
  std::mutex all_done_mutex;
  ThreadExceptionState thread_exception_state;
  const ShutdownRuntimeContext runtime_context{
      .thread_exception_state = &thread_exception_state,
      .completed_jobs = &completed_jobs,
      .all_done_cv = &all_done_cv,
      .all_done_mutex = &all_done_mutex,
  };

  auto observability = std::make_shared<starpu_server::RuntimeObservability>();
  observability->congestion_monitor =
      std::make_shared<starpu_server::congestion::Monitor>(nullptr);

  EXPECT_NO_THROW(
      run_shutdown_sequence(opts, ctx, queue, runtime_context, observability));
  EXPECT_EQ(observability->congestion_monitor, nullptr);
  EXPECT_TRUE(queue.is_shutdown());
  EXPECT_FALSE(queue.is_accepting());
}

static_assert(std::is_nothrow_destructible_v<RuntimeCleanupGuard>);
static_assert(std::is_nothrow_invocable_v<
              decltype(&RuntimeCleanupGuard::Dismiss), RuntimeCleanupGuard&>);
static_assert(std::is_nothrow_invocable_v<
              decltype(&RuntimeCleanupGuard::ResetTraceLoggerNoexceptForTest)>);
static_assert(std::is_nothrow_invocable_v<
              decltype(&RuntimeCleanupGuard::ShutdownMetricsNoexceptForTest)>);
static_assert(std::is_nothrow_invocable_v<
              decltype(&SignalNotificationPipe::SetPipeFailureForTest), bool>);
static_assert(
    std::is_nothrow_invocable_v<
        decltype(&SignalNotificationPipe::SetSetNonBlockingFailureForTest),
        bool>);
static_assert(std::is_nothrow_invocable_r_v<
              WaitForSignalNotificationReadOverrideForTestFn&,
              decltype(wait_for_signal_notification_read_override_for_test)>);
static_assert(std::is_same_v<
              decltype(TracePlotRuntimeHooks{}.waitpid_nohang),
              WaitPidNoHangOverrideForTestFn>);
static_assert(std::is_same_v<
              decltype(TracePlotRuntimeHooks{}.waitpid_blocking),
              WaitPidBlockingOverrideForTestFn>);
static_assert(std::is_same_v<
              decltype(TracePlotRuntimeHooks{}.wait_for_plot_process_wait),
              WaitForPlotProcessWaitOverrideForTestFn>);
static_assert(std::is_same_v<
              decltype(TracePlotRuntimeHooks{}.terminate_and_wait),
              TerminateAndWaitOverrideForTestFn>);
static_assert(std::is_same_v<
              decltype(TracePlotRuntimeHooks{}.resolve_python_candidates),
              ResolvePythonCandidatesOverrideForTestFn>);
static_assert(std::is_same_v<
              decltype(TracePlotRuntimeHooks{}.resolve_python_is_regular_file),
              ResolvePythonIsRegularFileOverrideForTestFn>);
static_assert(
    std::is_same_v<
        decltype(TracePlotRuntimeHooks{}.candidate_plot_scripts_read_symlink),
        CandidatePlotScriptsReadSymlinkOverrideForTestFn>);
static_assert(std::is_same_v<
              decltype(TracePlotRuntimeHooks{}.locate_plot_script_candidates),
              LocatePlotScriptCandidatesOverrideForTestFn>);
static_assert(std::is_same_v<
              decltype(TracePlotRuntimeHooks{}.run_plot_script),
              RunPlotScriptOverrideForTestFn>);
static_assert(std::is_same_v<
              decltype(TracePlotRuntimeHooks{}.run_plot_script_fork),
              RunPlotScriptForkOverrideForTestFn>);
static_assert(std::is_same_v<
              decltype(TracePlotRuntimeHooks{}.locate_plot_script),
              LocatePlotScriptOverrideForTestFn>);
static_assert(std::is_same_v<
              decltype(TracePlotRuntimeHooks{}.trace_summary_file_path),
              TraceSummaryFilePathOverrideForTestFn>);
static_assert(
    std::is_nothrow_invocable_r_v<
        std::remove_reference_t<
            decltype(load_model_and_reference_output_override_for_test())>&,
        decltype(load_model_and_reference_output_override_for_test)>);
static_assert(
    std::is_nothrow_invocable_r_v<
        std::remove_reference_t<decltype(run_warmup_override_for_test())>&,
        decltype(run_warmup_override_for_test)>);
static_assert(std::is_nothrow_invocable_r_v<
              WorkerCpusetProviderOverrideForTestFn&,
              decltype(worker_cpuset_provider_override_for_test)>);
static_assert(std::is_nothrow_invocable_r_v<
              HwlocBitmapFirstOverrideForTestFn&,
              decltype(hwloc_bitmap_first_override_for_test)>);
static_assert(std::is_nothrow_invocable_r_v<
              HwlocBitmapNextOverrideForTestFn&,
              decltype(hwloc_bitmap_next_override_for_test)>);
static_assert(std::is_nothrow_invocable_r_v<
              HwlocBitmapFreeOverrideForTestFn&,
              decltype(hwloc_bitmap_free_override_for_test)>);
static_assert(std::is_nothrow_invocable_r_v<
              WorkerCountOverrideForTestFn&,
              decltype(worker_count_override_for_test)>);
static_assert(
    std::is_nothrow_invocable_r_v<
        WorkerTypeOverrideForTestFn&, decltype(worker_type_override_for_test)>);
static_assert(std::is_nothrow_invocable_r_v<
              WorkerDeviceIdOverrideForTestFn&,
              decltype(worker_device_id_override_for_test)>);
static_assert(std::is_nothrow_invocable_r_v<
              DescribeCpuAffinityOverrideForTestFn&,
              decltype(describe_cpu_affinity_override_for_test)>);
static_assert(std::is_nothrow_invocable_r_v<
              ShutdownDrainTimeoutOverrideForTestFn&,
              decltype(shutdown_drain_timeout_override_for_test)>);
static_assert(std::is_nothrow_invocable_r_v<
              ShutdownDrainWaitStepOverrideForTestFn&,
              decltype(shutdown_drain_wait_step_override_for_test)>);
static_assert(std::is_nothrow_invocable_r_v<
              ShutdownDrainObserverForTestFn&,
              decltype(shutdown_drain_observer_for_test)>);

TEST(ServerMainRuntimeCleanupGuard, DestructorPerformsCleanupWhenActive)
{
  auto& tracer = starpu_server::BatchingTraceLogger::instance();
  tracer.configure(false, "");
  starpu_server::shutdown_metrics();

  tracer.configure(true, "");
  ASSERT_TRUE(tracer.enabled());
  ASSERT_TRUE(starpu_server::init_metrics(0));
  ASSERT_NE(starpu_server::get_metrics(), nullptr);

  {
    RuntimeCleanupGuard guard;
  }

  EXPECT_FALSE(tracer.enabled());
  EXPECT_EQ(starpu_server::get_metrics(), nullptr);
}

TEST(ServerMainRuntimeCleanupGuard, DestructorSkipsCleanupAfterDismiss)
{
  auto& tracer = starpu_server::BatchingTraceLogger::instance();
  tracer.configure(false, "");
  starpu_server::shutdown_metrics();

  tracer.configure(true, "");
  ASSERT_TRUE(tracer.enabled());
  ASSERT_TRUE(starpu_server::init_metrics(0));
  ASSERT_NE(starpu_server::get_metrics(), nullptr);

  {
    RuntimeCleanupGuard guard;
    EXPECT_NO_THROW(guard.Dismiss());
    EXPECT_NO_THROW(guard.Dismiss());
  }

  EXPECT_TRUE(tracer.enabled());
  EXPECT_NE(starpu_server::get_metrics(), nullptr);

  tracer.configure(false, "");
  starpu_server::shutdown_metrics();
}

TEST(
    ServerMainRuntimeCleanupGuard,
    DestructorResetsInjectedObservabilityMembersWhenActive)
{
  auto temp_dir = make_temp_test_directory("runtime_cleanup_guard_injected");
  auto tracer = std::make_shared<starpu_server::BatchingTraceLogger>();
  tracer->configure(true, (temp_dir.path / "trace.json").string());
  ASSERT_TRUE(tracer->enabled());

  auto metrics = starpu_server::create_metrics_recorder(0);
  ASSERT_NE(metrics, nullptr);
  ASSERT_TRUE(metrics->enabled());

  auto observability = std::make_shared<starpu_server::RuntimeObservability>();
  observability->tracer = tracer;
  observability->metrics = metrics;
  observability->congestion_monitor =
      std::make_shared<starpu_server::congestion::Monitor>(nullptr);

  {
    RuntimeCleanupGuard guard(observability);
  }

  ASSERT_NE(observability->tracer, nullptr);
  EXPECT_FALSE(observability->tracer->enabled());
  EXPECT_EQ(observability->metrics, nullptr);
  EXPECT_EQ(observability->congestion_monitor, nullptr);
}

TEST(
    ServerMainRuntimeCleanupGuard,
    DestructorSwallowsInjectedObservabilityTracerExceptions)
{
  auto temp_dir = make_temp_test_directory("runtime_cleanup_guard_throw");
  auto tracer = std::make_shared<starpu_server::BatchingTraceLogger>();
  tracer->configure(true, (temp_dir.path / "trace.json").string());
  ASSERT_TRUE(tracer->enabled());

  auto& trace_writer =
      starpu_server::testing::BatchingTraceLoggerTestAccessor::trace_writer(
          *tracer);
  auto& trace_stream =
      starpu_server::testing::TraceFileWriterTestAccessor::stream(trace_writer);
  trace_stream.setstate(std::ios::badbit);
  try {
    trace_stream.exceptions(std::ios::badbit);
  }
  catch (const std::ios_base::failure&) {
  }

  auto metrics = starpu_server::create_metrics_recorder(0);
  ASSERT_NE(metrics, nullptr);

  auto observability = std::make_shared<starpu_server::RuntimeObservability>();
  observability->tracer = tracer;
  observability->metrics = metrics;

  EXPECT_NO_THROW({ RuntimeCleanupGuard guard(observability); });

  EXPECT_NE(observability->metrics, nullptr);

  trace_stream.exceptions(std::ios::goodbit);
  trace_stream.clear();
  EXPECT_NO_THROW(tracer->configure(false, ""));
}

TEST(ServerMainRuntimeCleanupGuard, ResetTraceLoggerNoexceptDisablesTracer)
{
  auto& tracer = starpu_server::BatchingTraceLogger::instance();
  tracer.configure(false, "");
  tracer.configure(true, "");
  ASSERT_TRUE(tracer.enabled());

  EXPECT_NO_THROW(RuntimeCleanupGuard::ResetTraceLoggerNoexceptForTest());
  EXPECT_FALSE(tracer.enabled());
}

TEST(
    ServerMainRuntimeCleanupGuard,
    ResetTraceLoggerNoexceptSwallowsInternalExceptions)
{
  auto& tracer = starpu_server::BatchingTraceLogger::instance();
  tracer.configure(false, "");
  tracer.configure(true, "");
  ASSERT_TRUE(tracer.enabled());

  ScopedResetTraceLoggerForcedThrow forced_throw;
  EXPECT_NO_THROW(RuntimeCleanupGuard::ResetTraceLoggerNoexceptForTest());

  EXPECT_TRUE(tracer.enabled());
  tracer.configure(false, "");
}

TEST(ServerMainRuntimeCleanupGuard, ShutdownMetricsNoexceptShutsDownRegistry)
{
  starpu_server::shutdown_metrics();
  ASSERT_TRUE(starpu_server::init_metrics(0));
  ASSERT_NE(starpu_server::get_metrics(), nullptr);

  EXPECT_NO_THROW(RuntimeCleanupGuard::ShutdownMetricsNoexceptForTest());
  EXPECT_EQ(starpu_server::get_metrics(), nullptr);
}

TEST(
    ServerMainRuntimeCleanupGuard,
    ShutdownMetricsNoexceptSwallowsInternalExceptions)
{
  starpu_server::shutdown_metrics();
  ASSERT_TRUE(starpu_server::init_metrics(0));
  ASSERT_NE(starpu_server::get_metrics(), nullptr);

  ScopedShutdownMetricsForcedThrow forced_throw;
  EXPECT_NO_THROW(RuntimeCleanupGuard::ShutdownMetricsNoexceptForTest());

  EXPECT_NE(starpu_server::get_metrics(), nullptr);
  starpu_server::shutdown_metrics();
}

TEST(
    ServerMainRuntimeCleanupGuard,
    ResetInjectedObservabilityNoexceptReturnsWhenObservabilityNull)
{
  const auto reset_injected_observability =
      get(ResetInjectedObservabilityNoexceptTag{});

  EXPECT_NO_THROW(reset_injected_observability(nullptr));
}

TEST(
    ServerMainSignalNotificationPipe, ConstructorFallsBackWhenPipeCreationFails)
{
  signal_stop_notify_fd() = -1;
  ScopedSignalPipeForcedPipeFailure forced_failure;
  OStreamCapture capture_err(std::cerr);
  SignalNotificationPipe signal_pipe;

  EXPECT_FALSE(signal_pipe.active());
  EXPECT_EQ(signal_pipe.read_fd(), -1);
  EXPECT_EQ(signal_stop_notify_fd(), -1);
  EXPECT_NE(
      capture_err.str().find("Failed to create stop-notification pipe"),
      std::string::npos);
}

TEST(
    ServerMainSignalNotificationPipe,
    ConstructorFallsBackWhenSetNonBlockingFails)
{
  signal_stop_notify_fd() = -1;
  ScopedSignalPipeForcedSetNonBlockingFailure forced_failure;
  OStreamCapture capture_err(std::cerr);
  SignalNotificationPipe signal_pipe;

  EXPECT_FALSE(signal_pipe.active());
  EXPECT_EQ(signal_pipe.read_fd(), -1);
  EXPECT_EQ(signal_stop_notify_fd(), -1);
  EXPECT_NE(
      capture_err.str().find("Failed to configure stop-notification pipe write "
                             "end as non-blocking"),
      std::string::npos);
}

TEST(ServerMainSignalNotificationPipe, SetNonBlockingReturnsFalseForNegativeFd)
{
  EXPECT_FALSE(SignalNotificationPipe::SetNonBlockingForTest(-1));
}

TEST(
    ServerMainSignalNotificationPipe,
    SetNonBlockingReturnsFalseWhenFcntlGetFlFails)
{
  std::array<int, 2> file_descriptors{-1, -1};
  ASSERT_EQ(::pipe(file_descriptors.data()), 0);

  const int closed_fd = file_descriptors[1];
  ASSERT_EQ(::close(file_descriptors[1]), 0);
  file_descriptors[1] = -1;

  EXPECT_FALSE(SignalNotificationPipe::SetNonBlockingForTest(closed_fd));

  ASSERT_EQ(::close(file_descriptors[0]), 0);
}

TEST(ServerMainSignalNotificationWait, ReturnsImmediatelyWhenReadFdIsNegative)
{
  WaitForSignalNotificationReadOverrideState override_state;
  override_state.read_results = {-1};
  override_state.errnos = {EIO};
  ScopedWaitForSignalNotificationReadOverride override_guard(override_state);

  OStreamCapture capture_err(std::cerr);
  wait_for_signal_notification(-1);

  EXPECT_EQ(override_state.call_count, 0);
  EXPECT_TRUE(capture_err.str().empty());
}

TEST(ServerMainSignalNotificationWait, RetriesReadWhenInterruptedBySignal)
{
  WaitForSignalNotificationReadOverrideState override_state;
  override_state.read_results = {-1, 0};
  override_state.errnos = {EINTR, 0};
  ScopedWaitForSignalNotificationReadOverride override_guard(override_state);

  OStreamCapture capture_err(std::cerr);
  wait_for_signal_notification(42);

  EXPECT_EQ(override_state.call_count, 2);
  EXPECT_EQ(override_state.last_read_fd, 42);
  EXPECT_EQ(override_state.last_buffer_size, static_cast<std::size_t>(16));
  EXPECT_TRUE(capture_err.str().empty());
}

TEST(ServerMainSignalNotificationWait, LogsWarningWhenReadFailsWithNonEintr)
{
  WaitForSignalNotificationReadOverrideState override_state;
  override_state.read_results = {-1};
  override_state.errnos = {EIO};
  ScopedWaitForSignalNotificationReadOverride override_guard(override_state);

  OStreamCapture capture_err(std::cerr);
  wait_for_signal_notification(7);

  EXPECT_EQ(override_state.call_count, 1);
  EXPECT_EQ(
      capture_err.str(),
      expected_warning_log(
          std::string("Failed while waiting for stop signal notification: ") +
          std::strerror(EIO)));
}

TEST(ServerMainOrchestration, LaunchThreadsStopsAfterSignal)
{
  if (running_under_tsan()) {
    GTEST_SKIP()
        << "Skipped under TSAN: gRPC event-engine triggers known external race "
           "in absl::raw_hash_set";
  }

  constexpr auto kLaunchTimeout = std::chrono::seconds(10);
  constexpr auto kServerStartTimeout = std::chrono::seconds(5);

  const int port = pick_unused_port();
  ASSERT_GT(port, 0);
  const std::string address = "127.0.0.1:" + std::to_string(port);

  starpu_server::RuntimeConfig opts;
  opts.server_address = address;
  opts.congestion.enabled = false;
  opts.batching.max_queue_size = 8;
  opts.devices.gpu_model_replication =
      starpu_server::GpuModelReplicationPolicy::PerWorker;

  signal_stop_requested_flag() = 0;
  auto& ctx = server_context();
  ctx.stop_requested.store(false, std::memory_order_relaxed);

  starpu_server::StarPUSetup starpu(opts);
  torch::jit::script::Module model_cpu("m");
  std::vector<torch::jit::script::Module> models_gpu;
  std::vector<torch::Tensor> reference_outputs;
  starpu_server::InferenceQueue queue(opts.batching.max_queue_size);

  OStreamCapture capture_err(std::cerr);
  std::promise<void> done_promise;
  auto done_future = done_promise.get_future();

  std::jthread launch_thread([&]() {
    try {
      launch_threads(
          opts, starpu, model_cpu, models_gpu, reference_outputs, queue);
      done_promise.set_value();
    }
    catch (...) {
      done_promise.set_exception(std::current_exception());
    }
  });

  ASSERT_TRUE(wait_for_channel_ready(address, kServerStartTimeout))
      << "Timed out waiting for gRPC server startup";
  signal_handler(SIGTERM);

  if (done_future.wait_for(kLaunchTimeout) != std::future_status::ready) {
    signal_stop_requested_flag() = 1;
    ctx.stop_requested.store(true, std::memory_order_relaxed);
    ctx.stop_cv.notify_one();
  }

  ASSERT_EQ(done_future.wait_for(kLaunchTimeout), std::future_status::ready)
      << "launch_threads did not stop within timeout";
  EXPECT_NO_THROW(done_future.get());

  signal_stop_requested_flag() = 0;
}

TEST(
    ServerMainOrchestration,
    LaunchThreadsStopsAfterSignalWhenSignalPipeFallsBackToPolling)
{
  if (running_under_tsan()) {
    GTEST_SKIP()
        << "Skipped under TSAN: gRPC event-engine triggers known external race "
           "in absl::raw_hash_set";
  }

  constexpr auto kLaunchTimeout = std::chrono::seconds(10);
  constexpr auto kServerStartTimeout = std::chrono::seconds(5);

  const int port = pick_unused_port();
  ASSERT_GT(port, 0);
  const std::string address = "127.0.0.1:" + std::to_string(port);

  starpu_server::RuntimeConfig opts;
  opts.server_address = address;
  opts.congestion.enabled = false;
  opts.batching.max_queue_size = 8;

  signal_stop_requested_flag() = 0;
  auto& ctx = server_context();
  ctx.stop_requested.store(false, std::memory_order_relaxed);

  ScopedSignalPipeForcedPipeFailure forced_pipe_failure;
  WaitForSignalNotificationReadOverrideState read_override_state;
  ScopedWaitForSignalNotificationReadOverride read_override_guard(
      read_override_state);

  starpu_server::StarPUSetup starpu(opts);
  torch::jit::script::Module model_cpu("m");
  std::vector<torch::jit::script::Module> models_gpu;
  std::vector<torch::Tensor> reference_outputs;
  starpu_server::InferenceQueue queue(opts.batching.max_queue_size);

  OStreamCapture capture_err(std::cerr);
  std::promise<void> done_promise;
  auto done_future = done_promise.get_future();

  std::jthread launch_thread([&]() {
    try {
      launch_threads(
          opts, starpu, model_cpu, models_gpu, reference_outputs, queue);
      done_promise.set_value();
    }
    catch (...) {
      done_promise.set_exception(std::current_exception());
    }
  });

  ASSERT_TRUE(wait_for_channel_ready(address, kServerStartTimeout))
      << "Timed out waiting for gRPC server startup";
  EXPECT_EQ(signal_stop_notify_fd(), -1);

  std::this_thread::sleep_for(std::chrono::milliseconds(40));
  EXPECT_EQ(read_override_state.call_count, 0);

  signal_handler(SIGTERM);

  if (done_future.wait_for(kLaunchTimeout) != std::future_status::ready) {
    signal_stop_requested_flag() = 1;
    ctx.stop_requested.store(true, std::memory_order_relaxed);
    ctx.stop_cv.notify_one();
  }

  ASSERT_EQ(done_future.wait_for(kLaunchTimeout), std::future_status::ready)
      << "launch_threads did not stop within timeout in polling fallback mode";
  EXPECT_NO_THROW(done_future.get());
  EXPECT_EQ(read_override_state.call_count, 0);
  EXPECT_EQ(
      capture_err.str(),
      expected_warning_log(
          "Failed to create stop-notification pipe; falling back to polling "
          "signal flag: Too many open files"));

  signal_stop_requested_flag() = 0;
}

TEST(
    ServerMainOrchestration,
    LaunchThreadsPropagatesConfiguredModelInputsAndOutputsToGrpcMetadata)
{
  if (running_under_tsan()) {
    GTEST_SKIP()
        << "Skipped under TSAN: gRPC event-engine triggers known external race "
           "in absl::raw_hash_set";
  }

  constexpr auto kLaunchTimeout = std::chrono::seconds(10);
  constexpr auto kServerStartTimeout = std::chrono::seconds(5);

  const int port = pick_unused_port();
  ASSERT_GT(port, 0);
  const std::string address = "127.0.0.1:" + std::to_string(port);

  starpu_server::RuntimeConfig opts;
  opts.server_address = address;
  opts.congestion.enabled = false;
  opts.batching.max_queue_size = 8;
  starpu_server::testing::set_effective_batch_capacity_for_tests(opts, 16);
  opts.name = "fallback_model_name";
  opts.model = starpu_server::ModelConfig{};
  opts.model->name = "configured_model_name";
  opts.model->inputs = {
      starpu_server::TensorConfig{
          .name = "pixels", .dims = {1, 3, 224, 224}, .type = at::kFloat},
      starpu_server::TensorConfig{
          .name = "tokens", .dims = {1, 128}, .type = at::kFloat},
  };
  opts.model->outputs = {
      starpu_server::TensorConfig{
          .name = "logits", .dims = {1, 1000}, .type = at::kFloat},
      starpu_server::TensorConfig{
          .name = "labels", .dims = {1}, .type = at::kFloat},
  };

  signal_stop_requested_flag() = 0;
  auto& ctx = server_context();
  ctx.stop_requested.store(false, std::memory_order_relaxed);

  starpu_server::StarPUSetup starpu(opts);
  torch::jit::script::Module model_cpu("m");
  std::vector<torch::jit::script::Module> models_gpu;
  std::vector<torch::Tensor> reference_outputs = {
      torch::zeros({1, 1000}, torch::dtype(torch::kFloat32)),
      torch::zeros({1}, torch::dtype(torch::kFloat32)),
  };
  starpu_server::InferenceQueue queue(opts.batching.max_queue_size);

  std::promise<void> done_promise;
  auto done_future = done_promise.get_future();

  std::jthread launch_thread([&]() {
    try {
      launch_threads(
          opts, starpu, model_cpu, models_gpu, reference_outputs, queue);
      done_promise.set_value();
    }
    catch (...) {
      done_promise.set_exception(std::current_exception());
    }
  });

  ASSERT_TRUE(wait_for_channel_ready(address, kServerStartTimeout))
      << "Timed out waiting for gRPC server startup";

  auto channel =
      grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
  auto stub = inference::GRPCInferenceService::NewStub(channel);
  grpc::ClientContext rpc_ctx;
  rpc_ctx.set_deadline(
      std::chrono::system_clock::now() + std::chrono::seconds(2));
  inference::ModelMetadataRequest request;
  request.set_name("configured_model_name");
  inference::ModelMetadataResponse response;

  const auto status = stub->ModelMetadata(&rpc_ctx, request, &response);
  ASSERT_TRUE(status.ok()) << status.error_message();

  EXPECT_EQ(response.name(), "configured_model_name");
  ASSERT_EQ(response.inputs_size(), 2);
  EXPECT_EQ(response.inputs(0).name(), "pixels");
  EXPECT_EQ(response.inputs(0).datatype(), "FP32");
  ASSERT_EQ(response.inputs(0).shape_size(), 4);
  EXPECT_EQ(response.inputs(0).shape(0), 1);
  EXPECT_EQ(response.inputs(0).shape(1), 3);
  EXPECT_EQ(response.inputs(0).shape(2), 224);
  EXPECT_EQ(response.inputs(0).shape(3), 224);
  EXPECT_EQ(response.inputs(1).name(), "tokens");
  EXPECT_EQ(response.inputs(1).datatype(), "FP32");
  ASSERT_EQ(response.inputs(1).shape_size(), 2);
  EXPECT_EQ(response.inputs(1).shape(0), 1);
  EXPECT_EQ(response.inputs(1).shape(1), 128);

  ASSERT_EQ(response.outputs_size(), 2);
  EXPECT_EQ(response.outputs(0).name(), "logits");
  EXPECT_EQ(response.outputs(0).datatype(), "FP32");
  ASSERT_EQ(response.outputs(0).shape_size(), 2);
  EXPECT_EQ(response.outputs(0).shape(0), 1);
  EXPECT_EQ(response.outputs(0).shape(1), 1000);
  EXPECT_EQ(response.outputs(1).name(), "labels");
  EXPECT_EQ(response.outputs(1).datatype(), "FP32");
  ASSERT_EQ(response.outputs(1).shape_size(), 1);
  EXPECT_EQ(response.outputs(1).shape(0), 1);

  signal_handler(SIGTERM);

  if (done_future.wait_for(kLaunchTimeout) != std::future_status::ready) {
    signal_stop_requested_flag() = 1;
    ctx.stop_requested.store(true, std::memory_order_relaxed);
    ctx.stop_cv.notify_one();
  }

  ASSERT_EQ(done_future.wait_for(kLaunchTimeout), std::future_status::ready)
      << "launch_threads did not stop within timeout after metadata RPC";
  EXPECT_NO_THROW(done_future.get());

  signal_stop_requested_flag() = 0;
}

TEST(
    ServerMainOrchestration,
    LaunchThreadsLogsShutdownDrainWhenJobsRemainIncompleteAtShutdown)
{
  if (running_under_tsan()) {
    GTEST_SKIP()
        << "Skipped under TSAN: gRPC event-engine triggers known external race "
           "in absl::raw_hash_set";
  }

  constexpr auto kLaunchTimeout = std::chrono::seconds(10);
  constexpr auto kServerStartTimeout = std::chrono::seconds(5);
  constexpr auto kHookEnteredTimeout = std::chrono::seconds(5);
  constexpr auto kQueueShutdownTimeout = std::chrono::seconds(5);

  const int port = pick_unused_port();
  ASSERT_GT(port, 0);
  const std::string address = "127.0.0.1:" + std::to_string(port);

  starpu_server::RuntimeConfig opts;
  opts.server_address = address;
  opts.verbosity = starpu_server::VerbosityLevel::Info;
  opts.congestion.enabled = false;
  opts.batching.max_queue_size = 8;
  starpu_server::testing::set_effective_batch_capacity_for_tests(opts, 4);
  opts.model = starpu_server::ModelConfig{};
  opts.model->name = "shutdown_drain_model";
  opts.model->inputs = {
      starpu_server::TensorConfig{
          .name = "input0", .dims = {2}, .type = at::kFloat},
  };
  opts.model->outputs = {
      starpu_server::TensorConfig{
          .name = "output0", .dims = {2}, .type = at::kFloat},
  };

  signal_stop_requested_flag() = 0;
  auto& ctx = server_context();
  ctx.stop_requested.store(false, std::memory_order_relaxed);

  std::mutex hook_mutex;
  std::condition_variable hook_cv;
  bool hook_entered = false;
  bool release_hook = false;
  ScopedRunBeforeSubmitHook run_before_submit_hook([&]() {
    std::unique_lock<std::mutex> lock(hook_mutex);
    hook_entered = true;
    hook_cv.notify_one();
    hook_cv.wait(lock, [&release_hook]() { return release_hook; });
  });
  ScopedSubmitInferenceTaskHook submit_hook([]() {
    throw std::runtime_error("forced submit failure for shutdown drain test");
  });

  starpu_server::StarPUSetup starpu(opts);
  torch::jit::script::Module model_cpu("m");
  std::vector<torch::jit::script::Module> models_gpu;
  std::vector<torch::Tensor> reference_outputs = {
      torch::zeros({1, 2}, torch::dtype(torch::kFloat32))};
  starpu_server::InferenceQueue queue(opts.batching.max_queue_size);

  OStreamCapture capture_err(std::cerr);
  OStreamCapture capture_out(std::cout);
  std::promise<void> done_promise;
  auto done_future = done_promise.get_future();

  std::jthread launch_thread([&]() {
    try {
      launch_threads(
          opts, starpu, model_cpu, models_gpu, reference_outputs, queue);
      done_promise.set_value();
    }
    catch (...) {
      done_promise.set_exception(std::current_exception());
    }
  });

  ASSERT_TRUE(wait_for_channel_ready(address, kServerStartTimeout))
      << "Timed out waiting for gRPC server startup";

  auto queued_job = std::make_shared<starpu_server::InferenceJob>(
      std::vector<torch::Tensor>{
          torch::ones({1, 2}, torch::dtype(torch::kFloat32))},
      std::vector<at::ScalarType>{at::kFloat}, 1337);
  ASSERT_TRUE(queue.push(queued_job));

  {
    std::unique_lock<std::mutex> lock(hook_mutex);
    ASSERT_TRUE(hook_cv.wait_for(lock, kHookEnteredTimeout, [&hook_entered]() {
      return hook_entered;
    })) << "Timed out waiting for worker to block before submit";
  }

  signal_handler(SIGTERM);

  const auto queue_shutdown_observed = [&queue, kQueueShutdownTimeout]() {
    const auto deadline =
        std::chrono::steady_clock::now() + kQueueShutdownTimeout;
    while (std::chrono::steady_clock::now() < deadline) {
      if (queue.is_shutdown()) {
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return queue.is_shutdown();
  }();
  ASSERT_TRUE(queue_shutdown_observed)
      << "Timed out waiting for launch_threads to start shutdown drain";

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  {
    std::lock_guard<std::mutex> lock(hook_mutex);
    release_hook = true;
  }
  hook_cv.notify_all();

  if (done_future.wait_for(kLaunchTimeout) != std::future_status::ready) {
    {
      std::lock_guard<std::mutex> lock(hook_mutex);
      release_hook = true;
    }
    hook_cv.notify_all();
    signal_stop_requested_flag() = 1;
    ctx.stop_requested.store(true, std::memory_order_relaxed);
    ctx.stop_cv.notify_one();
  }

  ASSERT_EQ(done_future.wait_for(kLaunchTimeout), std::future_status::ready)
      << "launch_threads did not stop within timeout in shutdown drain path";
  EXPECT_NO_THROW(done_future.get());
  EXPECT_NE(
      capture_err.str().find("forced submit failure for shutdown drain test"),
      std::string::npos);

  EXPECT_NE(
      capture_out.str().find("Shutdown drain started: completed="),
      std::string::npos);

  signal_stop_requested_flag() = 0;
}

TEST(
    ServerMainOrchestration,
    LaunchThreadsDrainLoopBreaksWhenCompletedReachesTotalJobs)
{
  if (running_under_tsan()) {
    GTEST_SKIP()
        << "Skipped under TSAN: gRPC event-engine triggers known external race "
           "in absl::raw_hash_set";
  }

  constexpr auto kLaunchTimeout = std::chrono::seconds(10);
  constexpr auto kServerStartTimeout = std::chrono::seconds(5);
  constexpr auto kCompletionObservedTimeout = std::chrono::seconds(5);

  const int port = pick_unused_port();
  ASSERT_GT(port, 0);
  const std::string address = "127.0.0.1:" + std::to_string(port);

  starpu_server::RuntimeConfig opts;
  opts.server_address = address;
  opts.verbosity = starpu_server::VerbosityLevel::Silent;
  opts.congestion.enabled = false;
  opts.batching.max_queue_size = 8;
  starpu_server::testing::set_effective_batch_capacity_for_tests(opts, 4);
  opts.model = starpu_server::ModelConfig{};
  opts.model->name = "drain_completed_model";
  opts.model->inputs = {
      starpu_server::TensorConfig{
          .name = "input0", .dims = {2}, .type = at::kFloat},
  };
  opts.model->outputs = {
      starpu_server::TensorConfig{
          .name = "output0", .dims = {2}, .type = at::kFloat},
  };

  signal_stop_requested_flag() = 0;
  auto& ctx = server_context();
  ctx.stop_requested.store(false, std::memory_order_relaxed);

  ShutdownDrainOverrideState drain_state;
  ScopedShutdownDrainOverrides shutdown_drain_overrides(drain_state);

  std::mutex completion_mutex;
  std::condition_variable completion_cv;
  bool completion_observed = false;
  starpu_server::ResultDispatcher::PrepareJobCompletionCallbackTestHooks
      completion_hooks;
  completion_hooks.before_dispatch = [&]() {
    std::lock_guard<std::mutex> lock(completion_mutex);
    completion_observed = true;
    completion_cv.notify_one();
  };
  ScopedPrepareJobCompletionCallbackHooks completion_hooks_guard(
      std::move(completion_hooks));
  ScopedSubmitInferenceTaskHook submit_hook([]() {
    throw std::runtime_error(
        "forced submit failure for completed>=total drain test");
  });

  starpu_server::StarPUSetup starpu(opts);
  torch::jit::script::Module model_cpu("m");
  std::vector<torch::jit::script::Module> models_gpu;
  std::vector<torch::Tensor> reference_outputs = {
      torch::zeros({1, 2}, torch::dtype(torch::kFloat32))};
  starpu_server::InferenceQueue queue(opts.batching.max_queue_size);

  OStreamCapture capture_err(std::cerr);
  std::promise<void> done_promise;
  auto done_future = done_promise.get_future();

  std::jthread launch_thread([&]() {
    try {
      launch_threads(
          opts, starpu, model_cpu, models_gpu, reference_outputs, queue);
      done_promise.set_value();
    }
    catch (...) {
      done_promise.set_exception(std::current_exception());
    }
  });

  ASSERT_TRUE(wait_for_channel_ready(address, kServerStartTimeout))
      << "Timed out waiting for gRPC server startup";

  auto queued_job = std::make_shared<starpu_server::InferenceJob>(
      std::vector<torch::Tensor>{
          torch::ones({1, 2}, torch::dtype(torch::kFloat32))},
      std::vector<at::ScalarType>{at::kFloat}, 2001);
  ASSERT_TRUE(queue.push(queued_job));

  {
    std::unique_lock<std::mutex> lock(completion_mutex);
    ASSERT_TRUE(completion_cv.wait_for(
        lock, kCompletionObservedTimeout,
        [&completion_observed]() { return completion_observed; }))
        << "Timed out waiting for completion callback to be reached";
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  signal_handler(SIGTERM);

  if (done_future.wait_for(kLaunchTimeout) != std::future_status::ready) {
    signal_stop_requested_flag() = 1;
    ctx.stop_requested.store(true, std::memory_order_relaxed);
    ctx.stop_cv.notify_one();
  }

  ASSERT_EQ(done_future.wait_for(kLaunchTimeout), std::future_status::ready)
      << "launch_threads did not stop within timeout in completed drain path";
  EXPECT_NO_THROW(done_future.get());
  EXPECT_NE(
      capture_err.str().find("forced submit failure for completed>=total drain "
                             "test"),
      std::string::npos);

  EXPECT_GT(drain_state.entered_calls, 0);
  EXPECT_GT(drain_state.completed_reached_total_calls, 0);
  EXPECT_EQ(drain_state.deadline_reached_calls, 0);

  signal_stop_requested_flag() = 0;
}

TEST(
    ServerMainOrchestration,
    LaunchThreadsDrainLoopWaitsAndTimesOutWhenJobsRemainIncomplete)
{
  if (running_under_tsan()) {
    GTEST_SKIP()
        << "Skipped under TSAN: gRPC event-engine triggers known external race "
           "in absl::raw_hash_set";
  }

  constexpr auto kLaunchTimeout = std::chrono::seconds(10);
  constexpr auto kServerStartTimeout = std::chrono::seconds(5);
  constexpr auto kHookEnteredTimeout = std::chrono::seconds(5);
  constexpr auto kDeadlineObservedTimeout = std::chrono::seconds(5);

  const int port = pick_unused_port();
  ASSERT_GT(port, 0);
  const std::string address = "127.0.0.1:" + std::to_string(port);

  starpu_server::RuntimeConfig opts;
  opts.server_address = address;
  opts.verbosity = starpu_server::VerbosityLevel::Silent;
  opts.congestion.enabled = false;
  opts.batching.max_queue_size = 8;
  starpu_server::testing::set_effective_batch_capacity_for_tests(opts, 4);
  opts.model = starpu_server::ModelConfig{};
  opts.model->name = "drain_timeout_model";
  opts.model->inputs = {
      starpu_server::TensorConfig{
          .name = "input0", .dims = {2}, .type = at::kFloat},
  };
  opts.model->outputs = {
      starpu_server::TensorConfig{
          .name = "output0", .dims = {2}, .type = at::kFloat},
  };

  signal_stop_requested_flag() = 0;
  auto& ctx = server_context();
  ctx.stop_requested.store(false, std::memory_order_relaxed);

  ShutdownDrainOverrideState drain_state;
  drain_state.timeout = std::chrono::milliseconds(200);
  drain_state.wait_step = std::chrono::milliseconds(5);
  ScopedShutdownDrainOverrides shutdown_drain_overrides(drain_state);

  std::mutex hook_mutex;
  std::condition_variable hook_cv;
  bool hook_entered = false;
  bool release_hook = false;
  ScopedRunBeforeSubmitHook run_before_submit_hook([&]() {
    std::unique_lock<std::mutex> lock(hook_mutex);
    hook_entered = true;
    hook_cv.notify_one();
    hook_cv.wait(lock, [&release_hook]() { return release_hook; });
  });
  ScopedSubmitInferenceTaskHook submit_hook([]() {
    throw std::runtime_error(
        "forced submit failure for timeout drain test after unblock");
  });

  starpu_server::StarPUSetup starpu(opts);
  torch::jit::script::Module model_cpu("m");
  std::vector<torch::jit::script::Module> models_gpu;
  std::vector<torch::Tensor> reference_outputs = {
      torch::zeros({1, 2}, torch::dtype(torch::kFloat32))};
  starpu_server::InferenceQueue queue(opts.batching.max_queue_size);

  OStreamCapture capture_err(std::cerr);
  std::promise<void> done_promise;
  auto done_future = done_promise.get_future();

  std::jthread launch_thread([&]() {
    try {
      launch_threads(
          opts, starpu, model_cpu, models_gpu, reference_outputs, queue);
      done_promise.set_value();
    }
    catch (...) {
      done_promise.set_exception(std::current_exception());
    }
  });

  ASSERT_TRUE(wait_for_channel_ready(address, kServerStartTimeout))
      << "Timed out waiting for gRPC server startup";

  auto queued_job = std::make_shared<starpu_server::InferenceJob>(
      std::vector<torch::Tensor>{
          torch::ones({1, 2}, torch::dtype(torch::kFloat32))},
      std::vector<at::ScalarType>{at::kFloat}, 2002);
  ASSERT_TRUE(queue.push(queued_job));

  {
    std::unique_lock<std::mutex> lock(hook_mutex);
    ASSERT_TRUE(hook_cv.wait_for(lock, kHookEnteredTimeout, [&hook_entered]() {
      return hook_entered;
    })) << "Timed out waiting for worker to block before submit";
  }

  signal_handler(SIGTERM);

  {
    std::unique_lock<std::mutex> lock(drain_state.mutex);
    ASSERT_TRUE(drain_state.cv.wait_for(
        lock, kDeadlineObservedTimeout,
        [&drain_state]() { return drain_state.deadline_reached_calls > 0; }))
        << "Timed out waiting for shutdown drain deadline branch";
  }

  {
    std::lock_guard<std::mutex> lock(hook_mutex);
    release_hook = true;
  }
  hook_cv.notify_all();

  if (done_future.wait_for(kLaunchTimeout) != std::future_status::ready) {
    {
      std::lock_guard<std::mutex> lock(hook_mutex);
      release_hook = true;
    }
    hook_cv.notify_all();
    signal_stop_requested_flag() = 1;
    ctx.stop_requested.store(true, std::memory_order_relaxed);
    ctx.stop_cv.notify_one();
  }

  ASSERT_EQ(done_future.wait_for(kLaunchTimeout), std::future_status::ready)
      << "launch_threads did not stop within timeout in timeout drain path";
  EXPECT_NO_THROW(done_future.get());

  EXPECT_GT(drain_state.entered_calls, 0);
  EXPECT_GT(drain_state.before_wait_calls, 0);
  EXPECT_GT(drain_state.deadline_reached_calls, 0);
  EXPECT_NE(
      capture_err.str().find("Shutdown drain timeout after"),
      std::string::npos);

  signal_stop_requested_flag() = 0;
}

TEST(
    ServerMainOrchestration,
    LaunchThreadsStopsWhenGrpcStartupFailsOnOccupiedPort)
{
  if (running_under_tsan()) {
    GTEST_SKIP()
        << "Skipped under TSAN: gRPC event-engine triggers known external race "
           "in absl::raw_hash_set";
  }

  constexpr auto kLaunchTimeout = std::chrono::seconds(10);

  auto occupied_port = occupy_loopback_port();
  ASSERT_GE(occupied_port.socket_fd, 0);
  ASSERT_GT(occupied_port.port, 0);
  const std::string address = "127.0.0.1:" + std::to_string(occupied_port.port);

  starpu_server::RuntimeConfig opts;
  opts.server_address = address;
  opts.congestion.enabled = false;
  opts.batching.max_queue_size = 8;

  signal_stop_requested_flag() = 0;
  auto& ctx = server_context();
  ctx.stop_requested.store(false, std::memory_order_relaxed);

  starpu_server::StarPUSetup starpu(opts);
  torch::jit::script::Module model_cpu("m");
  std::vector<torch::jit::script::Module> models_gpu;
  std::vector<torch::Tensor> reference_outputs;
  starpu_server::InferenceQueue queue(opts.batching.max_queue_size);

  testing::internal::CaptureStderr();
  std::promise<void> done_promise;
  auto done_future = done_promise.get_future();
  std::jthread launch_thread([&]() {
    try {
      launch_threads(
          opts, starpu, model_cpu, models_gpu, reference_outputs, queue);
      done_promise.set_value();
    }
    catch (...) {
      done_promise.set_exception(std::current_exception());
    }
  });

  ASSERT_EQ(done_future.wait_for(kLaunchTimeout), std::future_status::ready)
      << "launch_threads did not stop after gRPC startup failure";
  EXPECT_NO_THROW(done_future.get());
  const std::string err = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(ctx.stop_requested.load(std::memory_order_relaxed));
  EXPECT_EQ(signal_stop_requested_flag(), 0);
  EXPECT_NE(
      err.find("Failed to start gRPC server on " + address), std::string::npos);

  signal_stop_requested_flag() = 0;
}

TEST(ServerMainOrchestration, LaunchThreadsStopsOnBrutalSignal)
{
  if (running_under_tsan()) {
    GTEST_SKIP()
        << "Skipped under TSAN: gRPC event-engine triggers known external race "
           "in absl::raw_hash_set";
  }

  constexpr auto kLaunchTimeout = std::chrono::seconds(10);
  constexpr auto kServerStartTimeout = std::chrono::seconds(5);

  const int port = pick_unused_port();
  ASSERT_GT(port, 0);
  const std::string address = "127.0.0.1:" + std::to_string(port);

  starpu_server::RuntimeConfig opts;
  opts.server_address = address;
  opts.congestion.enabled = false;
  opts.batching.max_queue_size = 8;

  signal_stop_requested_flag() = 0;
  auto& ctx = server_context();
  ctx.stop_requested.store(false, std::memory_order_relaxed);

  starpu_server::StarPUSetup starpu(opts);
  torch::jit::script::Module model_cpu("m");
  std::vector<torch::jit::script::Module> models_gpu;
  std::vector<torch::Tensor> reference_outputs;
  starpu_server::InferenceQueue queue(opts.batching.max_queue_size);

  std::promise<void> done_promise;
  auto done_future = done_promise.get_future();

  std::jthread launch_thread([&]() {
    try {
      launch_threads(
          opts, starpu, model_cpu, models_gpu, reference_outputs, queue);
      done_promise.set_value();
    }
    catch (...) {
      done_promise.set_exception(std::current_exception());
    }
  });

  ASSERT_TRUE(wait_for_channel_ready(address, kServerStartTimeout))
      << "Timed out waiting for gRPC server startup";

  std::latch storm_start{1};
  std::jthread signal_storm([&storm_start]() {
    storm_start.wait();
    for (int i = 0; i < 50; ++i) {
      signal_handler(SIGINT);
    }
  });
  storm_start.count_down();

  if (done_future.wait_for(kLaunchTimeout) != std::future_status::ready) {
    signal_stop_requested_flag() = 1;
    ctx.stop_requested.store(true, std::memory_order_relaxed);
    ctx.stop_cv.notify_one();
  }

  ASSERT_EQ(done_future.wait_for(kLaunchTimeout), std::future_status::ready)
      << "launch_threads did not stop within timeout on brutal signal";
  EXPECT_NO_THROW(done_future.get());

  signal_stop_requested_flag() = 0;
}

TEST(ServerMainOrchestration, LaunchThreadsStopsUnderConcurrentRpcLoad)
{
  if (running_under_tsan()) {
    GTEST_SKIP()
        << "Skipped under TSAN: gRPC event-engine triggers known external race "
           "in absl::raw_hash_set";
  }

  constexpr auto kLaunchTimeout = std::chrono::seconds(10);
  constexpr auto kClientActivityTimeout = std::chrono::seconds(2);
  constexpr int kClientThreads = 6;

  const int port = pick_unused_port();
  ASSERT_GT(port, 0);
  const std::string address = "127.0.0.1:" + std::to_string(port);

  starpu_server::RuntimeConfig opts;
  opts.server_address = address;
  opts.congestion.enabled = false;
  opts.batching.max_queue_size = 16;

  signal_stop_requested_flag() = 0;
  auto& ctx = server_context();
  ctx.stop_requested.store(false, std::memory_order_relaxed);

  starpu_server::StarPUSetup starpu(opts);
  torch::jit::script::Module model_cpu("m");
  std::vector<torch::jit::script::Module> models_gpu;
  std::vector<torch::Tensor> reference_outputs;
  starpu_server::InferenceQueue queue(opts.batching.max_queue_size);

  std::promise<void> done_promise;
  auto done_future = done_promise.get_future();
  std::jthread launch_thread([&]() {
    try {
      launch_threads(
          opts, starpu, model_cpu, models_gpu, reference_outputs, queue);
      done_promise.set_value();
    }
    catch (...) {
      done_promise.set_exception(std::current_exception());
    }
  });

  auto channel =
      grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
  ASSERT_TRUE(channel->WaitForConnected(
      std::chrono::system_clock::now() + std::chrono::seconds(5)));

  std::atomic<bool> stop_clients{false};
  std::atomic<int> started_requests{0};
  std::mutex started_requests_mutex;
  std::condition_variable started_requests_cv;
  std::vector<std::jthread> clients;
  clients.reserve(kClientThreads);
  for (int i = 0; i < kClientThreads; ++i) {
    clients.emplace_back([&, channel]() {
      auto stub = inference::GRPCInferenceService::NewStub(channel);
      while (!stop_clients.load(std::memory_order_acquire)) {
        grpc::ClientContext rpc_ctx;
        rpc_ctx.set_deadline(
            std::chrono::system_clock::now() + std::chrono::milliseconds(200));
        inference::ServerLiveRequest request;
        inference::ServerLiveResponse response;
        const int previous =
            started_requests.fetch_add(1, std::memory_order_relaxed);
        if (previous == 0) {
          std::lock_guard<std::mutex> lock(started_requests_mutex);
          started_requests_cv.notify_one();
        }
        (void)stub->ServerLive(&rpc_ctx, request, &response);
      }
    });
  }

  {
    std::unique_lock<std::mutex> lock(started_requests_mutex);
    ASSERT_TRUE(started_requests_cv.wait_for(
        lock, kClientActivityTimeout,
        [&started_requests]() {
          return started_requests.load(std::memory_order_relaxed) > 0;
        }))
        << "No RPC request started before stop signal";
  }
  signal_handler(SIGTERM);
  stop_clients.store(true, std::memory_order_release);
  clients.clear();

  if (done_future.wait_for(kLaunchTimeout) != std::future_status::ready) {
    signal_stop_requested_flag() = 1;
    ctx.stop_requested.store(true, std::memory_order_relaxed);
    ctx.stop_cv.notify_one();
  }

  ASSERT_EQ(done_future.wait_for(kLaunchTimeout), std::future_status::ready)
      << "launch_threads did not stop within timeout under RPC load";
  EXPECT_NO_THROW(done_future.get());
  EXPECT_GT(started_requests.load(std::memory_order_relaxed), 0);

  signal_stop_requested_flag() = 0;
}

TEST(
    ServerMainOrchestration,
    LaunchThreadsStopsOnSignalStormUnderConcurrentRpcLoad)
{
  if (running_under_tsan()) {
    GTEST_SKIP()
        << "Skipped under TSAN: gRPC event-engine triggers known external race "
           "in absl::raw_hash_set";
  }

  constexpr auto kLaunchTimeout = std::chrono::seconds(10);
  constexpr auto kClientActivityTimeout = std::chrono::seconds(2);
  constexpr int kClientThreads = 8;

  const int port = pick_unused_port();
  ASSERT_GT(port, 0);
  const std::string address = "127.0.0.1:" + std::to_string(port);

  starpu_server::RuntimeConfig opts;
  opts.server_address = address;
  opts.congestion.enabled = false;
  opts.batching.max_queue_size = 16;

  signal_stop_requested_flag() = 0;
  auto& ctx = server_context();
  ctx.stop_requested.store(false, std::memory_order_relaxed);

  starpu_server::StarPUSetup starpu(opts);
  torch::jit::script::Module model_cpu("m");
  std::vector<torch::jit::script::Module> models_gpu;
  std::vector<torch::Tensor> reference_outputs;
  starpu_server::InferenceQueue queue(opts.batching.max_queue_size);

  std::promise<void> done_promise;
  auto done_future = done_promise.get_future();
  std::jthread launch_thread([&]() {
    try {
      launch_threads(
          opts, starpu, model_cpu, models_gpu, reference_outputs, queue);
      done_promise.set_value();
    }
    catch (...) {
      done_promise.set_exception(std::current_exception());
    }
  });

  auto channel =
      grpc::CreateChannel(address, grpc::InsecureChannelCredentials());
  ASSERT_TRUE(channel->WaitForConnected(
      std::chrono::system_clock::now() + std::chrono::seconds(5)));

  std::atomic<bool> stop_clients{false};
  std::atomic<int> started_requests{0};
  std::mutex started_requests_mutex;
  std::condition_variable started_requests_cv;
  std::vector<std::jthread> clients;
  clients.reserve(kClientThreads);
  for (int i = 0; i < kClientThreads; ++i) {
    clients.emplace_back([&, channel]() {
      auto stub = inference::GRPCInferenceService::NewStub(channel);
      while (!stop_clients.load(std::memory_order_acquire)) {
        grpc::ClientContext rpc_ctx;
        rpc_ctx.set_deadline(
            std::chrono::system_clock::now() + std::chrono::milliseconds(200));
        inference::ServerLiveRequest request;
        inference::ServerLiveResponse response;
        const int previous =
            started_requests.fetch_add(1, std::memory_order_relaxed);
        if (previous == 0) {
          std::lock_guard<std::mutex> lock(started_requests_mutex);
          started_requests_cv.notify_one();
        }
        (void)stub->ServerLive(&rpc_ctx, request, &response);
      }
    });
  }

  {
    std::unique_lock<std::mutex> lock(started_requests_mutex);
    ASSERT_TRUE(started_requests_cv.wait_for(
        lock, kClientActivityTimeout,
        [&started_requests]() {
          return started_requests.load(std::memory_order_relaxed) > 0;
        }))
        << "No RPC request started before signal storm";
  }

  std::latch storm_start{1};
  std::jthread signal_storm([&storm_start]() {
    storm_start.wait();
    for (int i = 0; i < 64; ++i) {
      signal_handler((i % 2 == 0) ? SIGINT : SIGTERM);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  });
  storm_start.count_down();

  if (done_future.wait_for(kLaunchTimeout) != std::future_status::ready) {
    signal_stop_requested_flag() = 1;
    ctx.stop_requested.store(true, std::memory_order_relaxed);
    ctx.stop_cv.notify_one();
  }

  stop_clients.store(true, std::memory_order_release);
  clients.clear();

  ASSERT_EQ(done_future.wait_for(kLaunchTimeout), std::future_status::ready)
      << "launch_threads did not stop within timeout on signal storm under "
         "RPC load";
  EXPECT_NO_THROW(done_future.get());
  EXPECT_GT(started_requests.load(std::memory_order_relaxed), 0);

  signal_stop_requested_flag() = 0;
}

TEST(
    ServerMainPlotScript,
    ResolvePythonExecutableSkipsCandidatesThatAreNotRegularFiles)
{
  ResolvePythonExecutableOverrideState override_state;
  override_state.candidates = {
      "/tmp/python_candidate_not_regular_0",
      "/tmp/python_candidate_not_regular_1"};
  override_state.is_regular_file_result = false;
  override_state.force_status_error = false;
  ScopedResolvePythonExecutableOverrides overrides(override_state);

  const auto python_path = resolve_python_executable(&overrides.hooks());
  EXPECT_FALSE(python_path.has_value());
  EXPECT_EQ(override_state.candidates_calls, 1);
  EXPECT_EQ(override_state.is_regular_file_calls, 2);
  EXPECT_EQ(override_state.observed_candidates, override_state.candidates);
}

TEST(
    ServerMainPlotScript,
    ResolvePythonExecutableReturnsNulloptWhenFileStatusReportsError)
{
  ResolvePythonExecutableOverrideState override_state;
  override_state.candidates = {"/tmp/python_candidate_status_error"};
  override_state.is_regular_file_result = true;
  override_state.force_status_error = true;
  ScopedResolvePythonExecutableOverrides overrides(override_state);

  const auto python_path = resolve_python_executable(&overrides.hooks());
  EXPECT_FALSE(python_path.has_value());
  EXPECT_EQ(override_state.candidates_calls, 1);
  EXPECT_EQ(override_state.is_regular_file_calls, 1);
  ASSERT_EQ(override_state.observed_candidates.size(), 1U);
  EXPECT_EQ(
      override_state.observed_candidates[0],
      std::filesystem::path("/tmp/python_candidate_status_error"));
}

TEST(
    ServerMainPlotScript,
    CandidatePlotScriptsReturnsEmptyWhenExecutableSymlinkReadFails)
{
  CandidatePlotScriptsReadSymlinkOverrideState override_state;
  override_state.error = std::make_error_code(std::errc::io_error);
  override_state.resolved_executable_path = "/tmp/unused";
  ScopedCandidatePlotScriptsReadSymlinkOverride override_guard(override_state);

  const auto candidates = candidate_plot_scripts(&override_guard.hooks());

  EXPECT_TRUE(candidates.empty());
  EXPECT_EQ(override_state.call_count, 1);
  EXPECT_EQ(override_state.observed_link_path, "/proc/self/exe");
}

TEST(
    ServerMainPlotScript,
    CandidatePlotScriptsStopsWhenBaseDirectoryHasNoParentPath)
{
  CandidatePlotScriptsReadSymlinkOverrideState override_state;
  override_state.error.clear();
  override_state.resolved_executable_path = "starpu_server";
  ScopedCandidatePlotScriptsReadSymlinkOverride override_guard(override_state);

  const auto candidates = candidate_plot_scripts(&override_guard.hooks());

  ASSERT_EQ(candidates.size(), 1U);
  EXPECT_EQ(candidates.front(), "scripts/plot_batch_summary.py");
  EXPECT_EQ(override_state.call_count, 1);
  EXPECT_EQ(override_state.observed_link_path, "/proc/self/exe");
}

TEST(ServerMainPlotScript, LocatePlotScriptSkipsEmptyCandidates)
{
  auto temp_directory =
      make_temp_test_directory("locate_plot_script_empty_candidate");
  const auto script_path = temp_directory.path / "plot_batch_summary.py";
  {
    std::ofstream script_file(script_path);
    ASSERT_TRUE(script_file.is_open());
    script_file << "#!/usr/bin/env python3\n";
  }
  ASSERT_TRUE(std::filesystem::is_regular_file(script_path));

  LocatePlotScriptCandidatesOverrideState override_state;
  override_state.candidates = {std::filesystem::path{}, script_path};
  ScopedLocatePlotScriptCandidatesOverride override_guard(override_state);

  starpu_server::RuntimeConfig opts;
  const auto located = locate_plot_script(opts, &override_guard.hooks());

  ASSERT_TRUE(located.has_value());
  EXPECT_EQ(*located, script_path);
  EXPECT_EQ(override_state.call_count, 1);
}

TEST(ServerMainPlotScript, LocatePlotScriptResolvesRelativeCandidateToAbsolute)
{
  auto temp_directory = make_temp_test_directory("locate_plot_script_relative");
  const auto script_path = temp_directory.path / "plot_batch_summary.py";
  {
    std::ofstream script_file(script_path);
    ASSERT_TRUE(script_file.is_open());
    script_file << "#!/usr/bin/env python3\n";
  }
  ASSERT_TRUE(std::filesystem::is_regular_file(script_path));

  std::error_code rel_ec;
  const auto relative_candidate = std::filesystem::relative(
      script_path, std::filesystem::current_path(), rel_ec);
  ASSERT_FALSE(rel_ec);
  ASSERT_FALSE(relative_candidate.empty());
  ASSERT_FALSE(relative_candidate.is_absolute());

  LocatePlotScriptCandidatesOverrideState override_state;
  override_state.candidates = {relative_candidate};
  ScopedLocatePlotScriptCandidatesOverride override_guard(override_state);

  starpu_server::RuntimeConfig opts;
  const auto located = locate_plot_script(opts, &override_guard.hooks());

  ASSERT_TRUE(located.has_value());
  EXPECT_EQ(*located, std::filesystem::absolute(relative_candidate));
  EXPECT_TRUE(located->is_absolute());
  EXPECT_EQ(override_state.call_count, 1);
}

TEST(
    ServerMainPlotScript,
    LocatePlotScriptSkipsNonRegularFileCandidatesBeforeReturningAFile)
{
  auto temp_directory =
      make_temp_test_directory("locate_plot_script_type_check");
  const auto directory_candidate = temp_directory.path / "plot_dir";
  ASSERT_TRUE(std::filesystem::create_directories(directory_candidate));
  const auto script_path = temp_directory.path / "plot_batch_summary.py";
  {
    std::ofstream script_file(script_path);
    ASSERT_TRUE(script_file.is_open());
    script_file << "#!/usr/bin/env python3\n";
  }
  ASSERT_TRUE(std::filesystem::is_directory(directory_candidate));
  ASSERT_TRUE(std::filesystem::is_regular_file(script_path));

  LocatePlotScriptCandidatesOverrideState override_state;
  override_state.candidates = {directory_candidate, script_path};
  ScopedLocatePlotScriptCandidatesOverride override_guard(override_state);

  starpu_server::RuntimeConfig opts;
  const auto located = locate_plot_script(opts, &override_guard.hooks());

  ASSERT_TRUE(located.has_value());
  EXPECT_EQ(*located, script_path);
  EXPECT_EQ(override_state.call_count, 1);
}

TEST(
    ServerMainPlotScript,
    LocatePlotScriptReturnsNulloptWhenNoCandidateCanBeUsed)
{
  auto temp_directory = make_temp_test_directory("locate_plot_script_nullopt");
  const auto missing_candidate = temp_directory.path / "missing_plot.py";
  const auto directory_candidate =
      temp_directory.path / "non_regular_candidate";
  ASSERT_TRUE(std::filesystem::create_directories(directory_candidate));

  LocatePlotScriptCandidatesOverrideState override_state;
  override_state.candidates = {
      std::filesystem::path{}, missing_candidate, directory_candidate};
  ScopedLocatePlotScriptCandidatesOverride override_guard(override_state);

  starpu_server::RuntimeConfig opts;
  const auto located = locate_plot_script(opts, &override_guard.hooks());

  EXPECT_FALSE(located.has_value());
  EXPECT_EQ(override_state.call_count, 1);
}

TEST(ServerMainPlotScript, LocatePlotScriptFindsRepositoryScript)
{
  starpu_server::RuntimeConfig opts;
  const auto script_path = locate_plot_script(opts);
  ASSERT_TRUE(script_path.has_value());
  EXPECT_TRUE(script_path->is_absolute());
  EXPECT_EQ(script_path->filename(), "plot_batch_summary.py");
  EXPECT_TRUE(std::filesystem::is_regular_file(*script_path));
}

TEST(ServerMainPlotPath, PlotsOutputPathRemovesSummarySuffixWhenPresent)
{
  const std::filesystem::path summary_path{"/tmp/batch_summary.csv"};
  const auto output_path = plots_output_path(summary_path);
  EXPECT_EQ(output_path, std::filesystem::path("/tmp/batch_plots.png"));
}

TEST(ServerMainPlotPath, PlotsOutputPathKeepsStemWhenSummarySuffixAbsent)
{
  const std::filesystem::path summary_path{"/tmp/trace.csv"};
  const auto output_path = plots_output_path(summary_path);
  EXPECT_EQ(output_path, std::filesystem::path("/tmp/trace_plots.png"));
}

TEST(ServerMainPlotPath, PlotsOutputPathUsesLastSummaryOccurrence)
{
  const std::filesystem::path summary_path{"/tmp/a_summary_v2_summary.csv"};
  const auto output_path = plots_output_path(summary_path);
  EXPECT_EQ(output_path, std::filesystem::path("/tmp/a_summary_v2_plots.png"));
}

TEST(ServerMainPlotPath, RunTracePlotsReturnsImmediatelyWhenTracingDisabled)
{
  ScopedTraceLoggerReset trace_logger_reset;

  PlotScriptOverrideState override_state;
  override_state.summary_path_result = std::filesystem::path("/tmp/unused.csv");
  override_state.locate_result = std::filesystem::path("/tmp/fake_plot.py");
  override_state.run_result = 0;
  ScopedPlotScriptOverrides overrides(override_state);

  starpu_server::RuntimeConfig opts;
  opts.batching.trace_enabled = false;

  OStreamCapture capture_err(std::cerr);
  OStreamCapture capture_out(std::cout);
  run_trace_plots_if_enabled(opts, &overrides.hooks());

  EXPECT_EQ(override_state.summary_path_calls, 0);
  EXPECT_EQ(override_state.locate_calls, 0);
  EXPECT_EQ(override_state.run_calls, 0);
  EXPECT_EQ(capture_err.str(), "");
  EXPECT_EQ(capture_out.str(), "");
}

TEST(
    ServerMainPlotPath,
    RunTracePlotsUsesTracerSummaryPathWhenSummaryOverrideIsUnset)
{
  ScopedTraceLoggerReset trace_logger_reset;
  auto temp_directory =
      make_temp_test_directory("run_trace_plots_tracer_summary_path");
  const auto trace_path = temp_directory.path / "perfetto_trace.json";

  auto& tracer = starpu_server::BatchingTraceLogger::instance();
  tracer.configure(true, trace_path.string());
  const auto summary_path_opt = tracer.summary_file_path();
  ASSERT_TRUE(summary_path_opt.has_value());
  const auto summary_path = *summary_path_opt;
  ASSERT_TRUE(std::filesystem::exists(summary_path));
  std::error_code remove_ec;
  std::filesystem::remove(summary_path, remove_ec);
  ASSERT_FALSE(remove_ec);
  ASSERT_FALSE(std::filesystem::exists(summary_path));

  starpu_server::RuntimeConfig opts;
  opts.batching.trace_enabled = true;

  OStreamCapture capture_err(std::cerr);
  OStreamCapture capture_out(std::cout);
  run_trace_plots_if_enabled(opts);

  EXPECT_EQ(
      capture_err.str(), expected_warning_log(
                             "Tracing summary file '" + summary_path.string() +
                             "' not found; skipping plot generation."));
  EXPECT_EQ(capture_out.str(), "");
}

TEST(ServerMainPlotPath, RunTracePlotsWarnsWhenNoSummaryFileIsAvailable)
{
  ScopedTraceLoggerReset trace_logger_reset;

  PlotScriptOverrideState override_state;
  override_state.summary_path_result = std::nullopt;
  override_state.locate_result = std::filesystem::path("/tmp/fake_plot.py");
  override_state.run_result = 0;
  ScopedPlotScriptOverrides overrides(override_state);

  starpu_server::RuntimeConfig opts;
  opts.batching.trace_enabled = true;

  OStreamCapture capture_err(std::cerr);
  OStreamCapture capture_out(std::cout);
  run_trace_plots_if_enabled(opts, &overrides.hooks());

  EXPECT_EQ(override_state.summary_path_calls, 1);
  EXPECT_EQ(override_state.locate_calls, 0);
  EXPECT_EQ(override_state.run_calls, 0);
  EXPECT_EQ(
      capture_err.str(),
      expected_warning_log(
          "Tracing was enabled but no trace.csv was produced; skipping plot "
          "generation."));
  EXPECT_EQ(capture_out.str(), "");
}

TEST(ServerMainPlotPath, RunTracePlotsWarnsWhenSummaryFileIsMissing)
{
  ScopedTraceLoggerReset trace_logger_reset;
  auto temp_directory = make_temp_test_directory("run_trace_plots_missing_csv");
  const auto summary_path = temp_directory.path / "trace_summary.csv";
  ASSERT_FALSE(std::filesystem::exists(summary_path));

  PlotScriptOverrideState override_state;
  override_state.summary_path_result = summary_path;
  override_state.locate_result = std::filesystem::path("/tmp/fake_plot.py");
  override_state.run_result = 0;
  ScopedPlotScriptOverrides overrides(override_state);

  starpu_server::RuntimeConfig opts;
  opts.batching.trace_enabled = true;

  OStreamCapture capture_err(std::cerr);
  OStreamCapture capture_out(std::cout);
  run_trace_plots_if_enabled(opts, &overrides.hooks());

  EXPECT_EQ(override_state.summary_path_calls, 1);
  EXPECT_EQ(override_state.locate_calls, 0);
  EXPECT_EQ(override_state.run_calls, 0);
  EXPECT_EQ(
      capture_err.str(), expected_warning_log(
                             "Tracing summary file '" + summary_path.string() +
                             "' not found; skipping plot generation."));
  EXPECT_EQ(capture_out.str(), "");
}

TEST(ServerMainPlotPath, RunTracePlotsWarnsWhenPlotScriptCannotBeLocated)
{
  ScopedTraceLoggerReset trace_logger_reset;
  auto temp_directory =
      make_temp_test_directory("run_trace_plots_missing_script");
  const auto summary_path = temp_directory.path / "trace_summary.csv";
  TempFileGuard summary_guard{summary_path};
  {
    std::ofstream summary_file(summary_path);
    ASSERT_TRUE(summary_file.is_open());
    summary_file << "batch_id\n";
  }
  ASSERT_TRUE(std::filesystem::exists(summary_path));

  PlotScriptOverrideState override_state;
  override_state.summary_path_result = summary_path;
  override_state.locate_result = std::nullopt;
  override_state.run_result = 0;
  ScopedPlotScriptOverrides overrides(override_state);

  starpu_server::RuntimeConfig opts;
  opts.batching.trace_enabled = true;

  OStreamCapture capture_err(std::cerr);
  OStreamCapture capture_out(std::cout);
  run_trace_plots_if_enabled(opts, &overrides.hooks());

  EXPECT_EQ(override_state.summary_path_calls, 1);
  EXPECT_EQ(override_state.locate_calls, 1);
  EXPECT_EQ(override_state.run_calls, 0);
  EXPECT_EQ(
      capture_err.str(),
      expected_warning_log(
          "Unable to locate scripts/plot_batch_summary.py; skipping plot "
          "generation."));
  EXPECT_EQ(capture_out.str(), "");
}

TEST(ServerMainPlotPath, RunTracePlotsWarnsWhenPlotScriptDidNotComplete)
{
  ScopedTraceLoggerReset trace_logger_reset;
  auto temp_directory =
      make_temp_test_directory("run_trace_plots_script_incomplete");
  const auto summary_path = temp_directory.path / "trace_summary.csv";
  TempFileGuard summary_guard{summary_path};
  {
    std::ofstream summary_file(summary_path);
    ASSERT_TRUE(summary_file.is_open());
    summary_file << "batch_id\n";
  }
  ASSERT_TRUE(std::filesystem::exists(summary_path));

  PlotScriptOverrideState override_state;
  override_state.summary_path_result = summary_path;
  override_state.locate_result = temp_directory.path / "plot_batch_summary.py";
  override_state.run_result = std::nullopt;
  ScopedPlotScriptOverrides overrides(override_state);

  starpu_server::RuntimeConfig opts;
  opts.batching.trace_enabled = true;

  OStreamCapture capture_err(std::cerr);
  OStreamCapture capture_out(std::cout);
  run_trace_plots_if_enabled(opts, &overrides.hooks());

  ASSERT_TRUE(override_state.locate_result.has_value());
  EXPECT_EQ(override_state.summary_path_calls, 1);
  EXPECT_EQ(override_state.locate_calls, 1);
  EXPECT_EQ(override_state.run_calls, 1);
  EXPECT_EQ(override_state.run_script_path, *override_state.locate_result);
  EXPECT_EQ(override_state.run_summary_path, summary_path);
  EXPECT_EQ(override_state.run_output_path, plots_output_path(summary_path));
  EXPECT_EQ(
      capture_err.str(),
      expected_warning_log(
          "Failed to generate batching latency plots; plot script did not "
          "complete."));
  EXPECT_EQ(capture_out.str(), "");
}

TEST(ServerMainPlotPath, RunTracePlotsWarnsWhenPlotScriptFails)
{
  ScopedTraceLoggerReset trace_logger_reset;
  auto temp_directory =
      make_temp_test_directory("run_trace_plots_script_failed");
  const auto summary_path = temp_directory.path / "trace_summary.csv";
  TempFileGuard summary_guard{summary_path};
  {
    std::ofstream summary_file(summary_path);
    ASSERT_TRUE(summary_file.is_open());
    summary_file << "batch_id\n";
  }
  ASSERT_TRUE(std::filesystem::exists(summary_path));

  PlotScriptOverrideState override_state;
  override_state.summary_path_result = summary_path;
  override_state.locate_result = temp_directory.path / "plot_batch_summary.py";
  override_state.run_result = 42;
  ScopedPlotScriptOverrides overrides(override_state);

  starpu_server::RuntimeConfig opts;
  opts.batching.trace_enabled = true;

  const auto expected_output_path = plots_output_path(summary_path);
  const auto expected_message = std::format(
      "Failed to generate batching latency plots; python3 {} {} --output {} "
      "exited with code {}.",
      override_state.locate_result->string(), summary_path.string(),
      expected_output_path.string(), *override_state.run_result);

  OStreamCapture capture_err(std::cerr);
  OStreamCapture capture_out(std::cout);
  run_trace_plots_if_enabled(opts, &overrides.hooks());

  EXPECT_EQ(override_state.summary_path_calls, 1);
  EXPECT_EQ(override_state.locate_calls, 1);
  EXPECT_EQ(override_state.run_calls, 1);
  EXPECT_EQ(override_state.run_output_path, expected_output_path);
  EXPECT_EQ(capture_err.str(), expected_warning_log(expected_message));
  EXPECT_EQ(capture_out.str(), "");
}

TEST(ServerMainPlotPath, RunTracePlotsLogsInfoWhenPlotScriptSucceeds)
{
  ScopedTraceLoggerReset trace_logger_reset;
  auto temp_directory = make_temp_test_directory("run_trace_plots_script_ok");
  const auto summary_path = temp_directory.path / "trace_summary.csv";
  TempFileGuard summary_guard{summary_path};
  {
    std::ofstream summary_file(summary_path);
    ASSERT_TRUE(summary_file.is_open());
    summary_file << "batch_id\n";
  }
  ASSERT_TRUE(std::filesystem::exists(summary_path));

  PlotScriptOverrideState override_state;
  override_state.summary_path_result = summary_path;
  override_state.locate_result = temp_directory.path / "plot_batch_summary.py";
  override_state.run_result = 0;
  ScopedPlotScriptOverrides overrides(override_state);

  starpu_server::RuntimeConfig opts;
  opts.batching.trace_enabled = true;
  opts.verbosity = starpu_server::VerbosityLevel::Info;

  const auto expected_output_path = plots_output_path(summary_path);
  const auto expected_message = std::format(
      "Batching latency plots written to '{}'.", expected_output_path.string());

  OStreamCapture capture_err(std::cerr);
  OStreamCapture capture_out(std::cout);
  run_trace_plots_if_enabled(opts, &overrides.hooks());

  EXPECT_EQ(override_state.summary_path_calls, 1);
  EXPECT_EQ(override_state.locate_calls, 1);
  EXPECT_EQ(override_state.run_calls, 1);
  EXPECT_EQ(override_state.run_output_path, expected_output_path);
  EXPECT_EQ(capture_err.str(), "");
  EXPECT_EQ(capture_out.str(), expected_info_log(expected_message));
}

TEST(ServerMainPlotProcess, WaitForPlotProcessReturnsChildExitCode)
{
  const pid_t pid = spawn_exiting_child(7);
  ASSERT_GT(pid, 0);
  const auto exit_code = wait_for_plot_process(pid);
  ASSERT_TRUE(exit_code.has_value());
  EXPECT_EQ(*exit_code, 7);
}

TEST(ServerMainPlotProcess, WaitForPlotProcessReturnsNulloptOnWaitError)
{
  WaitForPlotProcessOverrideState override_state;
  override_state.wait_result = {WaitOutcome::Error, std::nullopt};
  override_state.terminate_result = 99;
  ScopedWaitForPlotProcessOverrides override_guard(override_state);

  const auto exit_code = wait_for_plot_process(4444, &override_guard.hooks());

  EXPECT_EQ(override_state.wait_calls, 1);
  EXPECT_EQ(override_state.wait_pid, 4444);
  EXPECT_EQ(
      override_state.wait_timeout, std::chrono::steady_clock::duration::zero());
  EXPECT_EQ(override_state.terminate_calls, 0);
  EXPECT_FALSE(exit_code.has_value());
}

TEST(ServerMainPlotProcess, WaitForPlotProcessDelegatesToTerminateAndWait)
{
  WaitForPlotProcessOverrideState override_state;
  override_state.wait_result = {WaitOutcome::TimedOut, std::nullopt};
  override_state.terminate_result = 17;
  ScopedWaitForPlotProcessOverrides override_guard(override_state);

  const auto exit_code = wait_for_plot_process(5555, &override_guard.hooks());

  EXPECT_EQ(override_state.wait_calls, 1);
  EXPECT_EQ(override_state.wait_pid, 5555);
  EXPECT_EQ(override_state.terminate_calls, 1);
  EXPECT_EQ(override_state.terminate_pid, 5555);
  ASSERT_TRUE(exit_code.has_value());
  EXPECT_EQ(*exit_code, 17);
}

TEST(
    ServerMainPlotProcess,
    WaitForPlotProcessCallsRealTerminateAndWaitWhenTerminateOverrideMissing)
{
  const pid_t pid = spawn_sleeping_child(false);
  ASSERT_GT(pid, 0);

  WaitForPlotProcessOverrideState override_state;
  override_state.wait_result = {WaitOutcome::TimedOut, std::nullopt};
  ScopedWaitForPlotProcessOverrides override_guard(override_state);
  override_guard.hooks().terminate_and_wait = nullptr;

  OStreamCapture capture_err(std::cerr);
  const auto exit_code = wait_for_plot_process(pid, &override_guard.hooks());

  EXPECT_EQ(override_state.wait_calls, 1);
  EXPECT_EQ(override_state.wait_pid, pid);
  EXPECT_EQ(override_state.terminate_calls, 0);
  ASSERT_TRUE(exit_code.has_value());
  EXPECT_TRUE(*exit_code == 128 + SIGTERM || *exit_code == 128 + SIGKILL)
      << "Expected SIGTERM or SIGKILL exit code, got " << *exit_code;
  EXPECT_EQ(
      capture_err.str(),
      expected_warning_log("Plot generation timed out; terminating python3."));
}

TEST(ServerMainPlotProcess, WaitStatusToExitCodeReturnsNulloptForStoppedStatus)
{
  // Typical POSIX wait status encoding for a stopped process.
  constexpr int stopped_status = 0x7f;
  EXPECT_TRUE(WIFSTOPPED(stopped_status));
  EXPECT_FALSE(WIFEXITED(stopped_status));
  EXPECT_FALSE(WIFSIGNALED(stopped_status));

  const auto exit_code = wait_status_to_exit_code(stopped_status);
  EXPECT_FALSE(exit_code.has_value());
}

TEST(ServerMainPlotProcess, WaitpidNohangRetriesWhenInterruptedBySignal)
{
  WaitPidNoHangOverrideState override_state;
  override_state.waitpid_results = {-1, 0};
  override_state.errnos = {EINTR, 0};
  ScopedWaitPidNoHangOverride override_guard(override_state);

  int status = 123;
  const auto result = waitpid_nohang(4242, status, &override_guard.hooks());

  EXPECT_EQ(override_state.call_count, 2);
  EXPECT_EQ(override_state.last_pid, 4242);
  EXPECT_EQ(override_state.last_options, WNOHANG);
  EXPECT_EQ(result.state, WaitPidState::StillRunning);
  EXPECT_FALSE(result.exit_code.has_value());
}

TEST(
    ServerMainPlotProcess,
    WaitpidNohangReturnsErrorAndNulloptWhenWaitpidFailsWithNonEintr)
{
  WaitPidNoHangOverrideState override_state;
  override_state.waitpid_results = {-1};
  override_state.errnos = {ECHILD};
  ScopedWaitPidNoHangOverride override_guard(override_state);

  OStreamCapture capture_err(std::cerr);
  int status = 0;
  const auto result = waitpid_nohang(999999, status, &override_guard.hooks());

  EXPECT_EQ(override_state.call_count, 1);
  EXPECT_EQ(override_state.last_pid, 999999);
  EXPECT_EQ(override_state.last_options, WNOHANG);
  EXPECT_EQ(result.state, WaitPidState::Error);
  EXPECT_FALSE(result.exit_code.has_value());
  EXPECT_EQ(
      capture_err.str(),
      expected_warning_log(
          std::string("Failed to wait for plot generation process: ") +
          std::strerror(ECHILD)));
}

TEST(
    ServerMainPlotProcess,
    WaitForExitWithTimeoutReturnsErrorWhenWaitpidNohangFails)
{
  WaitPidNoHangOverrideState override_state;
  override_state.waitpid_results = {-1};
  override_state.errnos = {ECHILD};
  ScopedWaitPidNoHangOverride override_guard(override_state);

  OStreamCapture capture_err(std::cerr);
  const auto result = wait_for_exit_with_timeout(
      13579, std::chrono::milliseconds(20), &override_guard.hooks());

  EXPECT_EQ(override_state.call_count, 1);
  EXPECT_EQ(override_state.last_pid, 13579);
  EXPECT_EQ(override_state.last_options, WNOHANG);
  EXPECT_EQ(result.outcome, WaitOutcome::Error);
  EXPECT_FALSE(result.exit_code.has_value());
  EXPECT_EQ(
      capture_err.str(),
      expected_warning_log(
          std::string("Failed to wait for plot generation process: ") +
          std::strerror(ECHILD)));
}

TEST(ServerMainPlotProcess, WaitForExitBlockingRetriesWhenInterruptedBySignal)
{
  WaitPidBlockingOverrideState override_state;
  constexpr pid_t kPid = 24680;
  constexpr int exited_status = 5 << 8;
  ASSERT_TRUE(WIFEXITED(exited_status));
  override_state.waitpid_results = {-1, kPid};
  override_state.errnos = {EINTR, 0};
  override_state.statuses = {0, exited_status};
  ScopedWaitPidBlockingOverride override_guard(override_state);

  const auto exit_code = wait_for_exit_blocking(kPid, &override_guard.hooks());

  EXPECT_EQ(override_state.call_count, 2);
  EXPECT_EQ(override_state.last_pid, kPid);
  EXPECT_EQ(override_state.last_options, 0);
  ASSERT_TRUE(exit_code.has_value());
  EXPECT_EQ(*exit_code, 5);
}

TEST(
    ServerMainPlotProcess,
    WaitForExitBlockingReturnsNulloptWhenWaitpidFailsWithNonEintr)
{
  WaitPidBlockingOverrideState override_state;
  override_state.waitpid_results = {-1};
  override_state.errnos = {ECHILD};
  ScopedWaitPidBlockingOverride override_guard(override_state);

  OStreamCapture capture_err(std::cerr);
  const auto exit_code = wait_for_exit_blocking(86420, &override_guard.hooks());

  EXPECT_EQ(override_state.call_count, 1);
  EXPECT_EQ(override_state.last_pid, 86420);
  EXPECT_EQ(override_state.last_options, 0);
  EXPECT_FALSE(exit_code.has_value());
  EXPECT_EQ(
      capture_err.str(),
      expected_warning_log(
          std::string("Failed to wait for plot generation process: ") +
          std::strerror(ECHILD)));
}

TEST(ServerMainPlotProcess, LogWaitpidErrorLogsWarningWithErrnoMessage)
{
  const int saved_errno = errno;
  errno = ECHILD;

  OStreamCapture capture_err(std::cerr);
  log_waitpid_error();

  EXPECT_EQ(
      capture_err.str(),
      expected_warning_log(
          std::string("Failed to wait for plot generation process: ") +
          std::strerror(ECHILD)));

  errno = saved_errno;
}

TEST(ServerMainPlotProcess, WaitForExitWithTimeoutTimesOutThenReapsChild)
{
  const pid_t pid = spawn_sleeping_child(false);
  ASSERT_GT(pid, 0);

  const auto result =
      wait_for_exit_with_timeout(pid, std::chrono::milliseconds(10));
  EXPECT_EQ(result.outcome, WaitOutcome::TimedOut);
  EXPECT_FALSE(result.exit_code.has_value());

  ASSERT_EQ(::kill(pid, SIGKILL), 0);
  const auto exit_code = wait_for_exit_blocking(pid);
  ASSERT_TRUE(exit_code.has_value());
  EXPECT_EQ(*exit_code, 128 + SIGKILL);
}

TEST(
    ServerMainPlotProcess,
    TerminateAndWaitReturnsExitCodeWhenTermWaitReportsExited)
{
  WaitPidNoHangOverrideState override_state;
  constexpr pid_t kPid = 31415;
  constexpr int exited_status = 3 << 8;
  ASSERT_TRUE(WIFEXITED(exited_status));
  override_state.waitpid_results = {kPid};
  override_state.errnos = {0};
  override_state.statuses = {exited_status};
  ScopedWaitPidNoHangOverride override_guard(override_state);

  OStreamCapture capture_err(std::cerr);
  const auto exit_code = terminate_and_wait(kPid, &override_guard.hooks());

  EXPECT_EQ(override_state.call_count, 1);
  EXPECT_EQ(override_state.last_pid, kPid);
  EXPECT_EQ(override_state.last_options, WNOHANG);
  ASSERT_TRUE(exit_code.has_value());
  EXPECT_EQ(*exit_code, 3);
  EXPECT_EQ(
      capture_err.str(),
      expected_warning_log("Plot generation timed out; terminating python3."));
}

TEST(
    ServerMainPlotProcess,
    TerminateAndWaitReturnsNulloptWhenTermWaitReportsError)
{
  WaitPidNoHangOverrideState override_state;
  override_state.waitpid_results = {-1};
  override_state.errnos = {ECHILD};
  ScopedWaitPidNoHangOverride override_guard(override_state);

  OStreamCapture capture_err(std::cerr);
  const auto exit_code = terminate_and_wait(27182, &override_guard.hooks());

  EXPECT_EQ(override_state.call_count, 1);
  EXPECT_EQ(override_state.last_pid, 27182);
  EXPECT_EQ(override_state.last_options, WNOHANG);
  EXPECT_FALSE(exit_code.has_value());
  EXPECT_EQ(
      capture_err.str(),
      expected_warning_log("Plot generation timed out; terminating python3.") +
          expected_warning_log(
              std::string("Failed to wait for plot generation process: ") +
              std::strerror(ECHILD)));
}

TEST(
    ServerMainPlotProcess, TerminateAndWaitEscalatesToSigkillWhenSigtermIgnored)
{
  const pid_t pid = spawn_sleeping_child(true);
  ASSERT_GT(pid, 0);

  OStreamCapture capture_err(std::cerr);
  const auto exit_code = terminate_and_wait(pid);
  ASSERT_TRUE(exit_code.has_value());
  EXPECT_EQ(*exit_code, 128 + SIGKILL);
  EXPECT_EQ(
      capture_err.str(),
      expected_warning_log("Plot generation timed out; terminating python3."));
}

TEST(ServerMainPlotProcess, RunPlotScriptReturnsNulloptWhenPythonNotFound)
{
  ResolvePythonExecutableOverrideState override_state;
  override_state.candidates = {};
  override_state.is_regular_file_result = false;
  override_state.force_status_error = false;
  ScopedResolvePythonExecutableOverrides overrides(override_state);

  OStreamCapture capture_err(std::cerr);
  const auto exit_code = run_plot_script(
      "/tmp/fake_plot_batch_summary.py", "/tmp/fake_summary.csv",
      "/tmp/fake_output.png", &overrides.hooks());

  EXPECT_FALSE(exit_code.has_value());
  EXPECT_EQ(override_state.candidates_calls, 1);
  EXPECT_EQ(override_state.is_regular_file_calls, 0);
  EXPECT_EQ(
      capture_err.str(),
      expected_warning_log(
          "python3 was not found in the allowlist; skipping plot generation."));
}

TEST(ServerMainPlotProcess, RunPlotScriptReturnsNulloptWhenForkFails)
{
  ResolvePythonExecutableOverrideState python_override_state;
  python_override_state.candidates = {"/bin/sh"};
  python_override_state.is_regular_file_result = true;
  python_override_state.force_status_error = false;
  ScopedResolvePythonExecutableOverrides python_overrides(
      python_override_state);

  RunPlotScriptForkOverrideState fork_override_state;
  fork_override_state.result = -1;
  fork_override_state.error_number = EAGAIN;
  ScopedRunPlotScriptForkOverride fork_override(fork_override_state);

  OStreamCapture capture_err(std::cerr);
  const auto hooks = merge_trace_plot_runtime_hooks(
      {&python_overrides.hooks(), &fork_override.hooks()});
  const auto exit_code = run_plot_script(
      "/tmp/fake_plot_batch_summary.py", "/tmp/fake_summary.csv",
      "/tmp/fake_output.png", &hooks);

  EXPECT_FALSE(exit_code.has_value());
  EXPECT_EQ(python_override_state.candidates_calls, 1);
  EXPECT_EQ(python_override_state.is_regular_file_calls, 1);
  EXPECT_EQ(fork_override_state.call_count, 1);
  EXPECT_EQ(
      capture_err.str(),
      expected_warning_log(
          std::string("Failed to launch python3 for plot generation: ") +
          std::strerror(EAGAIN)));
}

TEST(ServerMainPlotProcess, RunPlotScriptReturnsNonZeroForMissingScript)
{
  const auto summary_path = make_temp_test_path("batch_summary", ".csv");
  const auto output_path = make_temp_test_path("batch_plots", ".png");
  TempFileGuard summary_guard{summary_path};
  TempFileGuard output_guard{output_path};
  {
    std::ofstream summary_file(summary_path);
    ASSERT_TRUE(summary_file.is_open());
    summary_file << "timestamp_ms,latency_ms\n";
  }

  testing::internal::CaptureStderr();
  const auto exit_code = run_plot_script(
      "/definitely/missing_plot_batch_summary.py", summary_path, output_path);
  const std::string err = testing::internal::GetCapturedStderr();
  if (!resolve_python_executable().has_value()) {
    EXPECT_FALSE(exit_code.has_value());
    return;
  }

  ASSERT_TRUE(exit_code.has_value());
  EXPECT_NE(*exit_code, 0);
  EXPECT_NE(
      err.find("/definitely/missing_plot_batch_summary.py"), std::string::npos);
}

}  // namespace
