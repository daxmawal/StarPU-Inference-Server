#include <torch/torch.h>

#include <chrono>
#include <condition_variable>
#include <csignal>
#include <format>
#include <iostream>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "core/inference_runner.hpp"
#include "core/starpu_setup.hpp"
#include "inference_service.hpp"
#include "monitoring/metrics.hpp"
#include "signal_handler.hpp"
#include "starpu_task_worker/inference_queue.hpp"
#include "starpu_task_worker/starpu_task_worker.hpp"
#include "startup_probe.hpp"
#include "trace_plotting.hpp"
#include "utils/batching_trace_logger.hpp"
#include "utils/config_loader.hpp"
#include "utils/exceptions.hpp"
#include "utils/logger.hpp"
#include "utils/runtime_config.hpp"
#include "worker_info.hpp"

namespace starpu_server {

auto
handle_program_arguments(std::span<char const* const> args) -> RuntimeConfig
{
  const char* config_path = nullptr;

  auto remaining = args.subspan(1);
  auto require_value = [&](std::string_view flag) {
    if (remaining.empty() || remaining.front() == nullptr) {
      log_fatal(std::format("Missing value for {} argument.\n", flag));
    }
    const char* value = remaining.front();
    remaining = remaining.subspan(1);
    return value;
  };

  while (!remaining.empty()) {
    const char* raw_arg = remaining.front();
    remaining = remaining.subspan(1);

    if (raw_arg == nullptr) {
      log_fatal("Unexpected null program argument.\n");
    }

    std::string_view arg{raw_arg};
    if (arg == "--config" || arg == "-c") {
      config_path = require_value(arg);
      continue;
    }
    log_fatal(std::format(
        "Unknown argument '{}'. Only --config/-c is supported; all other "
        "settings must live in the YAML file.\n",
        arg));
  }

  if (config_path == nullptr) {
    log_fatal("Missing required --config argument.\n");
  }

  RuntimeConfig cfg = load_config(config_path);
  cfg.config_path = config_path;

  if (!cfg.valid) {
    log_fatal("Invalid configuration file.\n");
  }

  log_info(cfg.verbosity, std::format("__cplusplus = {}", __cplusplus));
  log_info(cfg.verbosity, std::format("LibTorch version: {}", TORCH_VERSION));
  log_info(cfg.verbosity, std::format("Scheduler       : {}", cfg.scheduler));
  if (!cfg.name.empty()) {
    log_info(cfg.verbosity, std::format("Configuration   : {}", cfg.name));
  }

  return cfg;
}

auto
prepare_models_and_warmup(const RuntimeConfig& opts, StarPUSetup& starpu)
    -> std::tuple<
        torch::jit::script::Module, std::vector<torch::jit::script::Module>,
        std::vector<torch::Tensor>>
{
  auto models = load_model_and_reference_output(opts);
  if (!models) {
    throw ModelLoadingException("Failed to load model or reference outputs");
  }
  auto [model_cpu, models_gpu, reference_outputs] = std::move(*models);
  run_warmup(opts, starpu, model_cpu, models_gpu, reference_outputs);
  return {model_cpu, models_gpu, reference_outputs};
}

void
launch_threads(
    const RuntimeConfig& opts, StarPUSetup& starpu,
    torch::jit::script::Module& model_cpu,
    std::vector<torch::jit::script::Module>& models_gpu,
    std::vector<torch::Tensor>& reference_outputs, double measured_throughput)
{
  static InferenceQueue queue;
  auto& server_ctx = server_context();
  server_ctx.queue_ptr = &queue;

  std::jthread notifier_thread([&server_ctx]() {
    constexpr auto kNotifierSleep = std::chrono::milliseconds(10);
    while (!server_ctx.stop_requested.load(std::memory_order_relaxed)) {
      std::this_thread::sleep_for(kNotifierSleep);
    }
    server_ctx.stop_cv.notify_one();
  });

  std::vector<InferenceResult> results;
  std::mutex results_mutex;
  std::atomic completed_jobs{0};
  std::condition_variable all_done_cv;

  StarPUTaskRunnerConfig config{};
  config.queue = &queue;
  config.model_cpu = &model_cpu;
  config.models_gpu = &models_gpu;
  config.starpu = &starpu;
  config.opts = &opts;
  config.results = &results;
  config.results_mutex = &results_mutex;
  config.completed_jobs = &completed_jobs;
  config.all_done_cv = &all_done_cv;
  StarPUTaskRunner worker(config);

  std::jthread worker_thread(&StarPUTaskRunner::run, &worker);
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
    const auto server_options = GrpcServerOptions{
        opts.server_address, opts.batching.max_message_bytes, opts.verbosity,
        std::move(default_model_name), measured_throughput};
    RunGrpcServer(
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
  StopServer(server_ctx.server.get());
  if (server_ctx.queue_ptr != nullptr) {
    server_ctx.queue_ptr->shutdown();
  }
  server_ctx.stop_cv.notify_one();
}

}  // namespace starpu_server

auto
main(int argc, char* argv[]) -> int
{
  try {
    starpu_server::RuntimeConfig opts = starpu_server::handle_program_arguments(
        {argv, static_cast<size_t>(argc)});
    starpu_server::BatchingTraceLogger::instance().configure_from_runtime(opts);
    const bool metrics_ok = starpu_server::init_metrics(opts.metrics_port);
    if (!metrics_ok) {
      starpu_server::log_warning(
          "Metrics server failed to start; continuing without metrics.");
    }
    starpu_server::StarPUSetup starpu(opts);
    starpu_server::log_worker_inventory(opts);
    auto [model_cpu, models_gpu, reference_outputs] =
        starpu_server::prepare_models_and_warmup(opts, starpu);
    const double measured_throughput =
        starpu_server::run_startup_throughput_probe(
            opts, starpu, model_cpu, models_gpu, reference_outputs);
    starpu_server::launch_threads(
        opts, starpu, model_cpu, models_gpu, reference_outputs,
        measured_throughput);
    auto& tracer = starpu_server::BatchingTraceLogger::instance();
    tracer.configure(false, "");
    starpu_server::run_trace_plots_if_enabled(opts);
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
