#include <atomic>
#include <csignal>
#include <iostream>
#include <memory>
#include <thread>

#include "cli/args_parser.hpp"
#include "core/inference_runner.hpp"
#include "core/starpu_setup.hpp"
#include "core/warmup.hpp"
#include "inference_service.hpp"
#include "starpu_task_worker/inference_queue.hpp"
#include "starpu_task_worker/starpu_task_worker.hpp"
#include "utils/exceptions.hpp"
#include "utils/logger.hpp"
#include "utils/runtime_config.hpp"

namespace {
starpu_server::InferenceQueue* g_queue_ptr = nullptr;
std::atomic g_stop_requested(false);
std::mutex g_stop_mutex;
std::condition_variable g_stop_cv;
}  // namespace

void
signal_handler(int /*signal*/)
{
  g_stop_requested.store(true);
  starpu_server::StopServer();
  if (g_queue_ptr != nullptr) {
    g_queue_ptr->shutdown();
  }
  g_stop_cv.notify_one();
}

auto
handle_program_arguments(int argc, char* argv[]) -> starpu_server::RuntimeConfig
{
  const starpu_server::RuntimeConfig opts = starpu_server::parse_arguments(
      std::span<char*>(argv, static_cast<size_t>(argc)));

  if (opts.show_help) {
    starpu_server::display_help("Inference Engine");
    std::exit(0);
  }

  if (!opts.valid) {
    starpu_server::log_fatal("Invalid program options.\n");
  }

  std::cout << "__cplusplus = " << __cplusplus << "\n"
            << "LibTorch version: " << TORCH_VERSION << "\n"
            << "Scheduler       : " << opts.scheduler << "\n"
            << "Iterations      : " << opts.iterations << "\n";

  return opts;
}

std::tuple<
    torch::jit::script::Module, std::vector<torch::jit::script::Module>,
    std::vector<torch::Tensor>>
prepare_models_and_warmup(
    const starpu_server::RuntimeConfig& opts,
    starpu_server::StarPUSetup& starpu)
{
  auto [model_cpu, models_gpu, reference_outputs] =
      starpu_server::load_model_and_reference_output(opts);
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
  g_queue_ptr = &queue;

  std::vector<starpu_server::InferenceResult> results;
  std::mutex results_mutex;
  std::atomic completed_jobs{0};
  std::condition_variable all_done_cv;

  starpu_server::StarPUTaskRunner worker(
      &queue, &model_cpu, &models_gpu, &starpu, &opts, &results, &results_mutex,
      &completed_jobs, &all_done_cv);

  std::jthread worker_thread(&starpu_server::StarPUTaskRunner::run, &worker);
  std::jthread grpc_thread([&]() {
    starpu_server::RunGrpcServer(
        queue, reference_outputs, opts.server_address, opts.max_message_bytes);
  });

  std::signal(SIGINT, signal_handler);

  {
    std::unique_lock lk(g_stop_mutex);
    g_stop_cv.wait(lk, [] { return g_stop_requested.load(); });
  }
}

auto
main(int argc, char* argv[]) -> int
{
  try {
    starpu_server::RuntimeConfig opts = handle_program_arguments(argc, argv);
    starpu_server::StarPUSetup starpu(opts);
    auto [model_cpu, models_gpu, reference_outputs] =
        prepare_models_and_warmup(opts, starpu);
    launch_threads(opts, starpu, model_cpu, models_gpu, reference_outputs);
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