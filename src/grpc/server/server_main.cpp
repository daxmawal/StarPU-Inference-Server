#include <atomic>
#include <chrono>
#include <csignal>
#include <format>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

#include "core/inference_runner.hpp"
#include "core/starpu_setup.hpp"
#include "inference_service.hpp"
#include "monitoring/metrics.hpp"
#include "starpu_task_worker/inference_queue.hpp"
#include "starpu_task_worker/starpu_task_worker.hpp"
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
  std::optional<int> input_slots_override;

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
    if (arg == "--input-slots" || arg == "--slots") {
      const char* value = require_value(arg);
      try {
        const int parsed = std::stoi(value);
        if (parsed <= 0) {
          throw std::invalid_argument("input-slots must be > 0");
        }
        input_slots_override = parsed;
      }
      catch (const std::exception& e) {
        starpu_server::log_fatal(
            std::format("Invalid --input-slots value: {}\n", e.what()));
      }
      continue;
    }
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
  log_info(
      cfg.verbosity,
      std::format("Request_nb      : {}", cfg.batching.request_nb));

  if (input_slots_override.has_value()) {
    cfg.batching.input_slots = *input_slots_override;
    starpu_server::log_info(
        cfg.verbosity,
        std::format(
            "Overriding input_slots from CLI: {}", cfg.batching.input_slots));
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
    const auto server_options = starpu_server::GrpcServerOptions{
        opts.server_address, opts.batching.max_message_bytes, opts.verbosity};
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
  starpu_server::StopServer(server_ctx.server);
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
    const bool metrics_ok = starpu_server::init_metrics(opts.metrics_port);
    if (!metrics_ok) {
      starpu_server::log_warning(
          "Metrics server failed to start; continuing without metrics.");
    }
    starpu_server::StarPUSetup starpu(opts);
    auto [model_cpu, models_gpu, reference_outputs] =
        prepare_models_and_warmup(opts, starpu);
    launch_threads(opts, starpu, model_cpu, models_gpu, reference_outputs);
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
