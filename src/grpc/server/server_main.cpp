#include <csignal>
#include <iostream>
#include <memory>
#include <thread>

#include "cli/args_parser.hpp"
#include "core/inference_runner.hpp"
#include "core/starpu_setup.hpp"
#include "core/warmup.hpp"
#include "inference_service.hpp"
#include "server/inference_queue.hpp"
#include "server/server_worker.hpp"
#include "utils/exceptions.hpp"
#include "utils/logger.hpp"
#include "utils/runtime_config.hpp"

static InferenceQueue* g_queue_ptr = nullptr;

void
signal_handler(int /*signal*/)
{
  if (g_queue_ptr) {
    g_queue_ptr->shutdown();
  }
  StopServer();
}

auto
main(int argc, char* argv[]) -> int
{
  const RuntimeConfig opts =
      parse_arguments(std::span<char*>(argv, static_cast<size_t>(argc)));

  if (opts.show_help) {
    display_help("Inference Engine");
    return 0;
  }

  if (!opts.valid) {
    log_fatal("Invalid program options.\n");
  }

  std::cout << "__cplusplus = " << __cplusplus << "\n"
            << "LibTorch version: " << TORCH_VERSION << "\n"
            << "Scheduler       : " << opts.scheduler << "\n"
            << "Iterations      : " << opts.iterations << "\n";

  try {
    StarPUSetup starpu(opts);

    torch::jit::script::Module model_cpu;
    std::vector<torch::jit::script::Module> models_gpu;
    std::vector<torch::Tensor> reference_outputs;
    std::tie(model_cpu, models_gpu, reference_outputs) =
        load_model_and_reference_output(opts);

    run_warmup(opts, starpu, model_cpu, models_gpu, reference_outputs);

    InferenceQueue queue;
    std::vector<InferenceResult> results;
    std::mutex results_mutex;
    std::atomic<unsigned int> completed_jobs = 0;
    std::condition_variable all_done_cv;

    StarPUTaskRunner worker(
        &queue, &model_cpu, &models_gpu, &starpu, &opts, &results,
        &results_mutex, &completed_jobs, &all_done_cv);

    std::jthread worker_thread(&StarPUTaskRunner::run, &worker);

    g_queue_ptr = &queue;
    std::signal(SIGINT, signal_handler);

    RunGrpcServer(
        queue, reference_outputs, opts.server_address, opts.max_message_bytes);

    if (worker_thread.joinable()) {
      worker_thread.join();
    }
  }
  catch (const InferenceEngineException& e) {
    std::cerr << "\033[1;31m[Inference Error] " << e.what() << "\033[0m\n";
    return 2;
  }
  catch (const std::exception& e) {
    std::cerr << "\033[1;31m[General Error] " << e.what() << "\033[0m\n";
    return -1;
  }

  return 0;
}
