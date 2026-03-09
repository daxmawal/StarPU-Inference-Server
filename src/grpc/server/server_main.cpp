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
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
#include "support/grpc/server/server_main_override_slot.hpp"
#endif  // SONAR_IGNORE_END

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

// clang-format off
#include "server_main_signal_runtime.hpp"
#include "server_main_shutdown_runtime.hpp"
#include "server_main_bootstrap.hpp"
#include "server_main_trace_plot_runtime.hpp"
#include "server_main_worker_inventory.hpp"
// clang-format on
#if defined(STARPU_TESTING)
}  // namespace starpu_server::testing::server_main
#else
}  // namespace
#endif

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
