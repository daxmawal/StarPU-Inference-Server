#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "core/inference_runner.hpp"
#include "core/starpu_setup.hpp"
#include "inference_service.hpp"
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
  starpu_server::InferenceQueue* queue_ptr = nullptr;
  std::unique_ptr<grpc::Server> server;
  std::mutex stop_mutex;
  std::condition_variable stop_cv;
  std::atomic<bool> stop_requested{false};
};

auto
server_context() -> ServerContext&
{
  static ServerContext ctx;
  return ctx;
}

auto
shell_quote(const std::string& value) -> std::string
{
  std::string quoted;
  quoted.reserve(value.size() + 2);
  quoted.push_back('\'');
  for (char ch : value) {
    if (ch == '\'') {
      quoted += "'\\''";
    } else {
      quoted.push_back(ch);
    }
  }
  quoted.push_back('\'');
  return quoted;
}

auto
candidate_plot_scripts(const starpu_server::RuntimeConfig& opts)
    -> std::vector<std::filesystem::path>
{
  std::vector<std::filesystem::path> candidates;
  candidates.emplace_back("scripts/plot_batch_summary.py");
  if (!opts.config_path.empty()) {
    const auto config_dir =
        std::filesystem::path(opts.config_path).parent_path();
    candidates.emplace_back(config_dir / "scripts/plot_batch_summary.py");
  }
  std::error_code exe_ec;
  const auto exe_path = std::filesystem::read_symlink("/proc/self/exe", exe_ec);
  if (!exe_ec) {
    candidates.emplace_back(
        std::filesystem::path(exe_path).parent_path() /
        "../scripts/plot_batch_summary.py");
  }
  return candidates;
}

auto
locate_plot_script(const starpu_server::RuntimeConfig& opts)
    -> std::optional<std::filesystem::path>
{
  for (const auto& candidate : candidate_plot_scripts(opts)) {
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

  auto& tracer = starpu_server::BatchingTraceLogger::instance();
  const auto summary_path_opt = tracer.summary_file_path();
  if (!summary_path_opt) {
    starpu_server::log_warning(
        "Tracing was enabled but no batching_trace_summary.csv was produced; "
        "skipping plot generation.");
    return;
  }

  const auto summary_path = *summary_path_opt;
  std::error_code ec;
  if (!std::filesystem::exists(summary_path, ec) || ec) {
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
  const std::string command = std::format(
      "python3 {} {} --output {}", shell_quote(script_path->string()),
      shell_quote(summary_path.string()), shell_quote(output_path.string()));
  const int rc = std::system(command.c_str());
  if (rc != 0) {
    starpu_server::log_warning(std::format(
        "Failed to generate batching latency plots; command '{}' exited with "
        "code {}.",
        command, rc));
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
  server_context().stop_requested.store(true);
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
  cfg.config_path = config_path;

  if (!cfg.valid) {
    starpu_server::log_fatal("Invalid configuration file.\n");
  }

  log_info(cfg.verbosity, std::format("__cplusplus = {}", __cplusplus));
  log_info(cfg.verbosity, std::format("LibTorch version: {}", TORCH_VERSION));
  log_info(cfg.verbosity, std::format("Scheduler       : {}", cfg.scheduler));
  if (!cfg.name.empty()) {
    log_info(cfg.verbosity, std::format("Configuration   : {}", cfg.name));
  }
  log_info(
      cfg.verbosity,
      std::format("Request_nb      : {}", cfg.batching.request_nb));

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

void
launch_threads(
    const starpu_server::RuntimeConfig& opts,
    starpu_server::StarPUSetup& starpu, torch::jit::script::Module& model_cpu,
    std::vector<torch::jit::script::Module>& models_gpu,
    std::vector<torch::Tensor>& reference_outputs)
{
  static starpu_server::InferenceQueue queue;
  auto& server_ctx = server_context();
  server_ctx.queue_ptr = &queue;

  std::jthread notifier_thread([&server_ctx]() {
    constexpr auto kNotifierSleep = std::chrono::milliseconds(10);
    while (!server_ctx.stop_requested.load(std::memory_order_relaxed)) {
      std::this_thread::sleep_for(kNotifierSleep);
    }
    server_ctx.stop_cv.notify_one();
  });

  std::vector<starpu_server::InferenceResult> results;
  std::mutex results_mutex;
  std::atomic completed_jobs{0};
  std::condition_variable all_done_cv;

  starpu_server::StarPUTaskRunnerConfig config{};
  config.queue = &queue;
  config.model_cpu = &model_cpu;
  config.models_gpu = &models_gpu;
  config.starpu = &starpu;
  config.opts = &opts;
  config.results = &results;
  config.results_mutex = &results_mutex;
  config.completed_jobs = &completed_jobs;
  config.all_done_cv = &all_done_cv;
  starpu_server::StarPUTaskRunner worker(config);

  std::jthread worker_thread(&starpu_server::StarPUTaskRunner::run, &worker);
  std::vector<at::ScalarType> expected_input_types;
  if (!opts.models.empty()) {
    expected_input_types.reserve(opts.models[0].inputs.size());
    for (const auto& input : opts.models[0].inputs) {
      expected_input_types.push_back(input.type);
    }
  }
  std::vector<std::vector<int64_t>> expected_input_dims;
  if (!opts.models.empty()) {
    expected_input_dims.reserve(opts.models[0].inputs.size());
    for (const auto& input : opts.models[0].inputs) {
      expected_input_dims.push_back(input.dims);
    }
  }

  std::jthread grpc_thread([&]() {
    std::string default_model_name = opts.name;
    if (default_model_name.empty() && !opts.models.empty()) {
      default_model_name = opts.models[0].name;
    }
    const auto server_options = starpu_server::GrpcServerOptions{
        opts.server_address, opts.batching.max_message_bytes, opts.verbosity,
        std::move(default_model_name)};
    starpu_server::RunGrpcServer(
        queue, reference_outputs, expected_input_types, expected_input_dims,
        opts.batching.max_batch_size, server_options, server_ctx.server);
  });

  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  {
    std::unique_lock lock(server_ctx.stop_mutex);
    server_ctx.stop_cv.wait(
        lock, [] { return server_context().stop_requested.load(); });
  }
  starpu_server::StopServer(server_ctx.server.get());
  if (server_ctx.queue_ptr != nullptr) {
    server_ctx.queue_ptr->shutdown();
  }
  server_ctx.stop_cv.notify_one();
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
    auto [model_cpu, models_gpu, reference_outputs] =
        prepare_models_and_warmup(opts, starpu);
    launch_threads(opts, starpu, model_cpu, models_gpu, reference_outputs);
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
