constexpr auto kPlotScriptTimeout = std::chrono::steady_clock::duration::zero();
constexpr auto kPlotScriptPollInterval = std::chrono::milliseconds(50);
constexpr auto kPlotScriptTerminateTimeout = std::chrono::seconds(1);
constexpr int kSignalExitCodeOffset = 128;
constexpr int kExecFailedExitCode = 127;
constexpr int kPlotScriptSearchDepth = 6;

#include "server_main_python_test_overrides.hpp"

auto
resolve_python_executable() -> std::optional<std::filesystem::path>
{
  static const std::array<std::filesystem::path, 3> kDefaultCandidates = {
      "/usr/bin/python3",
      "/usr/local/bin/python3",
      "/bin/python3",
  };

  std::vector<std::filesystem::path> override_candidates_storage;
  const auto candidates = resolve_python_candidates_for_runtime(
      kDefaultCandidates, override_candidates_storage);

  for (const auto& candidate : candidates) {
    std::error_code status_ec;
    const bool is_regular =
        resolve_python_is_regular_file_for_runtime(candidate, status_ec);
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
STARPU_SERVER_DECLARE_TEST_OVERRIDE_SLOT(
    WaitPidNoHangOverrideForTestFn, waitpid_nohang_override_for_test,
    pid_t (*)(pid_t, int*, int))
#endif  // SONAR_IGNORE_STOP

auto
read_waitpid_nohang(pid_t pid, int* status, int options) -> pid_t
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  return ::starpu_server::testing::server_main::detail::call_override_or(
      waitpid_nohang_override_for_test,
      [](pid_t child_pid, int* child_status, int wait_options) {
        return ::waitpid(child_pid, child_status, wait_options);
      },
      pid, status, options);
#else
  return ::waitpid(pid, status, options);
#endif  // SONAR_IGNORE_STOP
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
STARPU_SERVER_DECLARE_TEST_OVERRIDE_SLOT(
    WaitPidBlockingOverrideForTestFn, waitpid_blocking_override_for_test,
    pid_t (*)(pid_t, int*, int))
#endif  // SONAR_IGNORE_STOP

auto
read_waitpid_blocking(pid_t pid, int* status) -> pid_t
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  return ::starpu_server::testing::server_main::detail::call_override_or(
      waitpid_blocking_override_for_test,
      [](pid_t child_pid, int* child_status, int wait_options) {
        return ::waitpid(child_pid, child_status, wait_options);
      },
      pid, status, 0);
#else
  return ::waitpid(pid, status, 0);
#endif  // SONAR_IGNORE_STOP
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
STARPU_SERVER_DECLARE_TEST_OVERRIDE_SLOT(
    WaitForPlotProcessWaitOverrideForTestFn,
    wait_for_plot_process_wait_override_for_test,
    WaitOutcomeResult (*)(pid_t, std::chrono::steady_clock::duration))
STARPU_SERVER_DECLARE_TEST_OVERRIDE_SLOT(
    TerminateAndWaitOverrideForTestFn, terminate_and_wait_override_for_test,
    std::optional<int> (*)(pid_t))
#endif  // SONAR_IGNORE_STOP

auto
wait_for_plot_process(pid_t pid) -> std::optional<int>
{
  const auto result = [&]() {
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
    return ::starpu_server::testing::server_main::detail::call_override_or(
        wait_for_plot_process_wait_override_for_test,
        [](pid_t child_pid, std::chrono::steady_clock::duration timeout) {
          return wait_for_exit_with_timeout(child_pid, timeout);
        },
        pid, kPlotScriptTimeout);
#else
    return wait_for_exit_with_timeout(pid, kPlotScriptTimeout);
#endif  // SONAR_IGNORE_STOP
  }();
  if (result.outcome == WaitOutcome::Exited) {
    return result.exit_code;
  }
  if (result.outcome == WaitOutcome::Error) {
    return std::nullopt;
  }
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  return ::starpu_server::testing::server_main::detail::call_override_or(
      terminate_and_wait_override_for_test,
      [](pid_t child_pid) { return terminate_and_wait(child_pid); }, pid);
#else
  return terminate_and_wait(pid);
#endif  // SONAR_IGNORE_STOP
}

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
STARPU_SERVER_DECLARE_TEST_OVERRIDE_SLOT(
    RunPlotScriptOverrideForTestFn, run_plot_script_override_for_test,
    std::optional<int> (*)(
        const std::filesystem::path&, const std::filesystem::path&,
        const std::filesystem::path&))
STARPU_SERVER_DECLARE_TEST_OVERRIDE_SLOT(
    RunPlotScriptForkOverrideForTestFn, run_plot_script_fork_override_for_test,
    pid_t (*)())
STARPU_SERVER_DECLARE_TEST_OVERRIDE_SLOT(
    LocatePlotScriptOverrideForTestFn, locate_plot_script_override_for_test,
    std::optional<std::filesystem::path> (*)(
        const starpu_server::RuntimeConfig&))
STARPU_SERVER_DECLARE_TEST_OVERRIDE_SLOT(
    TraceSummaryFilePathOverrideForTestFn,
    trace_summary_file_path_override_for_test,
    std::optional<std::filesystem::path> (*)())
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
    return ::starpu_server::testing::server_main::detail::call_override_or(
        run_plot_script_fork_override_for_test, []() { return fork(); });
#else
    return fork();
#endif  // SONAR_IGNORE_STOP
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

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
STARPU_SERVER_DECLARE_TEST_OVERRIDE_SLOT(
    CandidatePlotScriptsReadSymlinkOverrideForTestFn,
    candidate_plot_scripts_read_symlink_override_for_test,
    std::filesystem::path (*)(const std::filesystem::path&, std::error_code&))
STARPU_SERVER_DECLARE_TEST_OVERRIDE_SLOT(
    LocatePlotScriptCandidatesOverrideForTestFn,
    locate_plot_script_candidates_override_for_test,
    std::vector<std::filesystem::path> (*)())
#endif  // SONAR_IGNORE_STOP

auto
read_symlink_for_candidate_plot_scripts(
    const std::filesystem::path& path,
    std::error_code& ec) -> std::filesystem::path
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  return ::starpu_server::testing::server_main::detail::call_override_or(
      candidate_plot_scripts_read_symlink_override_for_test,
      [](const std::filesystem::path& candidate, std::error_code& code) {
        return std::filesystem::read_symlink(candidate, code);
      },
      path, ec);
#else
  return std::filesystem::read_symlink(path, ec);
#endif  // SONAR_IGNORE_STOP
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
    return ::starpu_server::testing::server_main::detail::call_override_or(
        locate_plot_script_candidates_override_for_test,
        []() { return candidate_plot_scripts(); });
#else
    return candidate_plot_scripts();
#endif  // SONAR_IGNORE_STOP
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
  summary_path_opt =
      ::starpu_server::testing::server_main::detail::call_override_or(
          trace_summary_file_path_override_for_test, []() {
            const auto& tracer = starpu_server::BatchingTraceLogger::instance();
            return tracer.summary_file_path();
          });
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
