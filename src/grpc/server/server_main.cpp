#include <atomic>
#include <chrono>
#include <csignal>
#include <format>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <thread>

#include "core/inference_runner.hpp"
#include "core/starpu_setup.hpp"
#include "core/warmup.hpp"
#include "inference_service.hpp"
#include "monitoring/metrics.hpp"
#include "starpu_task_worker/inference_queue.hpp"
#include "starpu_task_worker/starpu_task_worker.hpp"
#include "utils/config_loader.hpp"
#include "utils/exceptions.hpp"
#include "utils/logger.hpp"
#include "utils/runtime_config.hpp"

namespace {
// Encapsulates state shared between the worker threads and the signal handler
struct ServerContext {
  starpu_server::InferenceQueue* queue_ptr = nullptr;
  std::unique_ptr<grpc::Server> server{};
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
handle_program_arguments(int argc, char* argv[]) -> starpu_server::RuntimeConfig
{
  const char* config_path = nullptr;

  for (int i = 1; i < argc - 1; ++i) {
    std::string_view arg{argv[i]};
    if ((arg == "--config" || arg == "-c") && argv[i + 1] != nullptr) {
      config_path = argv[i + 1];
      break;
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
  log_info(cfg.verbosity, std::format("Iterations      : {}", cfg.iterations));

  return cfg;
}

std::tuple<
    torch::jit::script::Module, std::vector<torch::jit::script::Module>,
    std::vector<torch::Tensor>>
prepare_models_and_warmup(
    const starpu_server::RuntimeConfig& opts,
    starpu_server::StarPUSetup& starpu)
{
  auto models = starpu_server::load_model_and_reference_output(opts);
  if (!models) {
    throw std::runtime_error("Failed to load model or reference outputs");
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
  auto& ctx = server_context();
  ctx.queue_ptr = &queue;

  std::jthread notifier_thread([]() {
    auto& ctx = server_context();
    while (!ctx.stop_requested.load(std::memory_order_relaxed)) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    ctx.stop_cv.notify_one();
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
  expected_input_types.reserve(opts.inputs.size());
  for (const auto& t : opts.inputs) {
    expected_input_types.push_back(t.type);
  }
  std::jthread grpc_thread([&, expected_input_types]() {
    starpu_server::RunGrpcServer(
        queue, reference_outputs, expected_input_types, opts.server_address,
        opts.max_message_bytes, opts.verbosity, ctx.server);
  });

  std::signal(SIGINT, signal_handler);

  {
    std::unique_lock lock(ctx.stop_mutex);
    ctx.stop_cv.wait(
        lock, [] { return server_context().stop_requested.load(); });
  }
  starpu_server::StopServer(ctx.server);
  if (ctx.queue_ptr != nullptr) {
    ctx.queue_ptr->shutdown();
  }
  ctx.stop_cv.notify_one();
}

auto
main(int argc, char* argv[]) -> int
{
  try {
    starpu_server::RuntimeConfig opts = handle_program_arguments(argc, argv);
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
