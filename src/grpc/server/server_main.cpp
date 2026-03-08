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
#include "server_main_override_slot.hpp"
#include "starpu_task_worker/inference_queue.hpp"
#include "starpu_task_worker/starpu_task_worker.hpp"
#include "utils/batching_trace_logger.hpp"
#include "utils/config_loader.hpp"
#include "utils/exceptions.hpp"
#include "utils/logger.hpp"
#include "utils/runtime_config.hpp"

#if defined(STARPU_TESTING)
namespace starpu_server::testing::server_main {
#else
namespace {
#endif
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

#include "server_main_signal_runtime.inl"
#include "server_main_trace_plot_runtime.inl"
#if defined(STARPU_TESTING)
}  // namespace starpu_server::testing::server_main
using starpu_server::testing::server_main::candidate_plot_scripts;
using starpu_server::testing::server_main::
    candidate_plot_scripts_read_symlink_override_for_test;
using starpu_server::testing::server_main::
    CandidatePlotScriptsReadSymlinkOverrideForTestFn;
using starpu_server::testing::server_main::kExecFailedExitCode;
using starpu_server::testing::server_main::kPlotScriptPollInterval;
using starpu_server::testing::server_main::kPlotScriptSearchDepth;
using starpu_server::testing::server_main::kPlotScriptTerminateTimeout;
using starpu_server::testing::server_main::kPlotScriptTimeout;
using starpu_server::testing::server_main::kSignalExitCodeOffset;
using starpu_server::testing::server_main::locate_plot_script;
using starpu_server::testing::server_main::
    locate_plot_script_candidates_override_for_test;
using starpu_server::testing::server_main::locate_plot_script_override_for_test;
using starpu_server::testing::server_main::
    LocatePlotScriptCandidatesOverrideForTestFn;
using starpu_server::testing::server_main::LocatePlotScriptOverrideForTestFn;
using starpu_server::testing::server_main::log_waitpid_error;
using starpu_server::testing::server_main::mark_server_started;
using starpu_server::testing::server_main::mark_server_stopped;
using starpu_server::testing::server_main::plots_output_path;
using starpu_server::testing::server_main::request_server_stop;
using starpu_server::testing::server_main::reset_server_state;
using starpu_server::testing::server_main::
    resolve_python_candidates_override_for_test;
using starpu_server::testing::server_main::resolve_python_executable;
using starpu_server::testing::server_main::
    resolve_python_is_regular_file_override_for_test;
using starpu_server::testing::server_main::
    ResolvePythonCandidatesOverrideForTestFn;
using starpu_server::testing::server_main::
    ResolvePythonIsRegularFileOverrideForTestFn;
using starpu_server::testing::server_main::rethrow_thread_exception_if_any;
using starpu_server::testing::server_main::run_plot_script;
using starpu_server::testing::server_main::
    run_plot_script_fork_override_for_test;
using starpu_server::testing::server_main::run_plot_script_override_for_test;
using starpu_server::testing::server_main::
    run_thread_entry_with_exception_capture;
using starpu_server::testing::server_main::run_trace_plots_if_enabled;
using starpu_server::testing::server_main::RunPlotScriptForkOverrideForTestFn;
using starpu_server::testing::server_main::RunPlotScriptOverrideForTestFn;
using starpu_server::testing::server_main::RuntimeCleanupGuard;
using starpu_server::testing::server_main::server_context;
using starpu_server::testing::server_main::ServerContext;
using starpu_server::testing::server_main::signal_stop_notify_fd;
using starpu_server::testing::server_main::signal_stop_requested_flag;
using starpu_server::testing::server_main::SignalNotificationPipe;
using starpu_server::testing::server_main::stop_server_when_available;
using starpu_server::testing::server_main::terminate_and_wait;
using starpu_server::testing::server_main::terminate_and_wait_override_for_test;
using starpu_server::testing::server_main::TerminateAndWaitOverrideForTestFn;
using starpu_server::testing::server_main::ThreadExceptionState;
using starpu_server::testing::server_main::
    trace_summary_file_path_override_for_test;
using starpu_server::testing::server_main::
    TraceSummaryFilePathOverrideForTestFn;
using starpu_server::testing::server_main::wait_for_exit_blocking;
using starpu_server::testing::server_main::wait_for_exit_with_timeout;
using starpu_server::testing::server_main::wait_for_plot_process;
using starpu_server::testing::server_main::
    wait_for_plot_process_wait_override_for_test;
using starpu_server::testing::server_main::wait_for_signal_notification;
using starpu_server::testing::server_main::
    wait_for_signal_notification_read_override_for_test;
using starpu_server::testing::server_main::wait_status_to_exit_code;
using starpu_server::testing::server_main::
    WaitForPlotProcessWaitOverrideForTestFn;
using starpu_server::testing::server_main::
    WaitForSignalNotificationReadOverrideForTestFn;
using starpu_server::testing::server_main::WaitOutcome;
using starpu_server::testing::server_main::WaitOutcomeResult;
using starpu_server::testing::server_main::waitpid_blocking_override_for_test;
using starpu_server::testing::server_main::waitpid_nohang;
using starpu_server::testing::server_main::waitpid_nohang_override_for_test;
using starpu_server::testing::server_main::WaitPidBlockingOverrideForTestFn;
using starpu_server::testing::server_main::WaitPidNoHangOverrideForTestFn;
using starpu_server::testing::server_main::WaitPidResult;
using starpu_server::testing::server_main::WaitPidState;
#else
}  // namespace
#endif

#include "server_main_bootstrap.inl"
#include "server_main_worker_inventory.inl"

// Test binary already provides gtest_main.
#if !defined(STARPU_TESTING)
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
#endif
