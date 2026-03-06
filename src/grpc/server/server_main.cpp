#include <hwloc.h>
#include <starpu.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
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
  std::unique_ptr<grpc::Server> server;
  std::mutex stop_mutex;
  std::condition_variable stop_cv;
  std::atomic<bool> stop_requested{false};
};

auto
signal_stop_requested_flag() -> volatile std::sig_atomic_t&
{
  static volatile std::sig_atomic_t value = 0;
  return value;
}

auto
server_context() -> ServerContext&
{
  static ServerContext ctx;
  return ctx;
}

constexpr auto kPlotScriptTimeout =
    std::chrono::steady_clock::duration::zero();  // Disable timeout.
constexpr auto kPlotScriptPollInterval = std::chrono::milliseconds(50);
constexpr auto kPlotScriptTerminateTimeout = std::chrono::seconds(1);
constexpr int kSignalExitCodeOffset = 128;
constexpr int kExecFailedExitCode = 127;
constexpr int kPlotScriptSearchDepth = 6;

auto
resolve_python_executable() -> std::optional<std::filesystem::path>
{
  static const std::array<std::filesystem::path, 3> kCandidates = {
      "/usr/bin/python3",
      "/usr/local/bin/python3",
      "/bin/python3",
  };
  for (const auto& candidate : kCandidates) {
    if (std::error_code status_ec;
        !std::filesystem::is_regular_file(candidate, status_ec) || status_ec) {
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

auto
waitpid_nohang(pid_t pid, int& status) -> WaitPidResult
{
  using enum WaitPidState;
  while (true) {
    const pid_t result = waitpid(pid, &status, WNOHANG);
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
    const pid_t result = waitpid(pid, &status, 0);
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

auto
wait_for_plot_process(pid_t pid) -> std::optional<int>
{
  const auto result = wait_for_exit_with_timeout(pid, kPlotScriptTimeout);
  if (result.outcome == WaitOutcome::Exited) {
    return result.exit_code;
  }
  if (result.outcome == WaitOutcome::Error) {
    return std::nullopt;
  }
  return terminate_and_wait(pid);
}

auto
run_plot_script(
    const std::filesystem::path& script_path,
    const std::filesystem::path& summary_path,
    const std::filesystem::path& output_path) -> std::optional<int>
{
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

  const pid_t pid = fork();
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

auto
candidate_plot_scripts() -> std::vector<std::filesystem::path>
{
  std::vector<std::filesystem::path> candidates;
  std::error_code exe_ec;
  const auto exe_path = std::filesystem::read_symlink("/proc/self/exe", exe_ec);
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
locate_plot_script(const starpu_server::RuntimeConfig& /*opts*/)
    -> std::optional<std::filesystem::path>
{
  for (const auto& candidate : candidate_plot_scripts()) {
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

  const auto& tracer = starpu_server::BatchingTraceLogger::instance();
  const auto summary_path_opt = tracer.summary_file_path();
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
  signal_stop_requested_flag() = 1;
}

auto
handle_program_arguments(std::span<char const* const> args)
    -> starpu_server::RuntimeConfig
{
  const char* config_path = nullptr;

  auto remaining = args.subspan(1);
  auto require_value = [&](std::string_view flag) {
    if (remaining.empty() || remaining.front() == nullptr) {
      starpu_server::log_fatal(
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
      starpu_server::log_fatal("Unexpected null program argument.\n");
    }

    std::string_view arg{raw_arg};
    if (arg == "--config" || arg == "-c") {
      config_path = require_value(arg);
      continue;
    }
    starpu_server::log_fatal(std::format(
        "Unknown argument '{}'. Only --config/-c is supported; all other "
        "settings must live in the YAML file.\n",
        arg));
  }

  if (config_path == nullptr) {
    starpu_server::log_fatal("Missing required --config argument.\n");
  }

  starpu_server::RuntimeConfig cfg = starpu_server::load_config(config_path);

  if (!cfg.valid) {
    starpu_server::log_fatal("Invalid configuration file.\n");
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

auto
prepare_models_and_warmup(
    const starpu_server::RuntimeConfig& opts,
    starpu_server::StarPUSetup& starpu)
    -> std::tuple<
        torch::jit::script::Module, std::vector<torch::jit::script::Module>,
        std::vector<torch::Tensor>>
{
  auto models = starpu_server::load_model_and_reference_output(opts);
  if (!models) {
    throw starpu_server::ModelLoadingException(
        "Failed to load model or reference outputs");
  }
  auto [model_cpu, models_gpu, reference_outputs] = std::move(*models);
  starpu_server::run_warmup(
      opts, starpu, model_cpu, models_gpu, reference_outputs);
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
launch_threads(
    const starpu_server::RuntimeConfig& opts,
    starpu_server::StarPUSetup& starpu, torch::jit::script::Module& model_cpu,
    std::vector<torch::jit::script::Module>& models_gpu,
    std::vector<torch::Tensor>& reference_outputs,
    starpu_server::InferenceQueue& queue)
{
  queue.reset_counters();
  auto& server_ctx = server_context();
  server_ctx.stop_requested.store(false, std::memory_order_relaxed);
  signal_stop_requested_flag() = 0;

  starpu_server::congestion::start(&queue, make_congestion_config(opts));

  std::jthread notifier_thread([&server_ctx]() {
    constexpr auto kNotifierSleep = std::chrono::milliseconds(10);
    while (signal_stop_requested_flag() == 0) {
      std::this_thread::sleep_for(kNotifierSleep);
    }
    server_ctx.stop_requested.store(true, std::memory_order_relaxed);
    server_ctx.stop_cv.notify_one();
  });

  std::atomic completed_jobs{0};
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

  std::jthread worker_thread(&starpu_server::StarPUTaskRunner::run, &worker);
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

  std::jthread grpc_thread([&]() {
    const auto server_options = make_grpc_server_options(opts);
    const auto model_spec = make_grpc_model_spec(
        opts, expected_input_types, expected_input_dims, expected_input_names,
        expected_output_names);
    starpu_server::RunGrpcServer(
        queue, reference_outputs, model_spec, server_options,
        server_ctx.server);
  });

  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  {
    std::unique_lock lock(server_ctx.stop_mutex);
    server_ctx.stop_cv.wait(lock, [&server_ctx] {
      return server_ctx.stop_requested.load(std::memory_order_relaxed);
    });
  }
  starpu_server::StopServer(server_ctx.server.get());
  queue.shutdown();
  const auto total_jobs = queue.total_pushed();
  if (total_jobs > 0) {
    std::unique_lock lock(all_done_mutex);
    all_done_cv.wait(lock, [&completed_jobs, total_jobs]() {
      const int completed = completed_jobs.load(std::memory_order_acquire);
      if (completed < 0) {
        return true;
      }
      return static_cast<std::size_t>(completed) >= total_jobs;
    });
  }
  starpu_server::congestion::shutdown();
  server_ctx.stop_cv.notify_one();
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
  hwloc_cpuset_t cpuset = starpu_worker_get_hwloc_cpuset(worker_id);
  if (cpuset == nullptr) {
    return {};
  }

  std::vector<int> cores;
  for (int core = hwloc_bitmap_first(cpuset); core != -1;
       core = hwloc_bitmap_next(cpuset, core)) {
    cores.push_back(core);
  }
  hwloc_bitmap_free(cpuset);
  return format_cpu_core_ranges(cores);
}

void
log_worker_inventory(const starpu_server::RuntimeConfig& opts)
{
  const auto total_workers = static_cast<int>(starpu_worker_get_count());
  starpu_server::log_info(
      opts.verbosity,
      std::format("Configured {} StarPU worker(s).", total_workers));

  for (int worker_id = 0; worker_id < total_workers; ++worker_id) {
    const auto type = starpu_worker_get_type(worker_id);
    const int device_id = starpu_worker_get_devid(worker_id);
    const std::string device_label =
        device_id >= 0 ? std::to_string(device_id) : "N/A";
    std::string cpu_affinity;
    if (type == STARPU_CPU_WORKER) {
      const std::string affinity = describe_cpu_affinity(worker_id);
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
