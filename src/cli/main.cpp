#include <torch/version.h>

#include <chrono>
#include <exception>
#include <format>
#include <memory>
#include <span>
#include <string_view>

#include "args_parser.hpp"
#include "config_loader.hpp"
#include "exceptions.hpp"
#include "inference_runner.hpp"
#include "logger.hpp"
#include "runtime_config.hpp"
#include "starpu_setup.hpp"

auto
main(int argc, char* argv[]) -> int
{
  std::span<char*> args_span(argv, static_cast<size_t>(argc));

  std::string config_path;
  for (int i = 1; i < argc - 1; ++i) {
    std::string_view arg = argv[i];
    if (arg == "--config" || arg == "-c") {
      config_path = argv[i + 1];
      break;
    }
  }

  starpu_server::RuntimeConfig opts;
  if (!config_path.empty()) {
    opts = starpu_server::load_config(config_path);
    opts.config_path = config_path;
  }

  opts = starpu_server::parse_arguments(args_span, opts);

  if (opts.show_help) {
    starpu_server::display_help("Inference Engine");
    return 0;
  }

  if (!opts.valid) {
    starpu_server::log_fatal("Invalid program options.\n");
  }

  starpu_server::log_info(
      opts.verbosity, std::format("__cplusplus = {}", __cplusplus));
  starpu_server::log_info(
      opts.verbosity, std::format("LibTorch version: {}", TORCH_VERSION));
  starpu_server::log_info(
      opts.verbosity, std::format("Scheduler       : {}", opts.scheduler));
  starpu_server::log_info(
      opts.verbosity, std::format("Iterations      : {}", opts.iterations));

  std::unique_ptr<starpu_server::StarPUSetup> starpu;
  try {
    const auto batch_size_from_config =
        [](const starpu_server::RuntimeConfig& cfg) -> std::size_t {
      if (cfg.models.empty() || cfg.models[0].inputs.empty()) {
        return 1;
      }

      const auto& dims = cfg.models[0].inputs[0].dims;
      if (dims.empty()) {
        return 1;
      }

      const auto dim0 = dims.front();
      return dim0 > 0 ? static_cast<std::size_t>(dim0) : std::size_t{1};
    };

    starpu = std::make_unique<starpu_server::StarPUSetup>(opts);

    const auto start_time = std::chrono::steady_clock::now();
    starpu_server::run_inference_loop(opts, *starpu);
    const auto end_time = std::chrono::steady_clock::now();

    const int iterations = opts.iterations;
    const std::size_t batch_size = batch_size_from_config(opts);
    if (iterations > 0 && batch_size > 0) {
      const std::size_t total_inferences =
          static_cast<std::size_t>(iterations) * batch_size;
      const double duration_s =
          std::chrono::duration<double>(end_time - start_time).count();
      if (duration_s > 0.0) {
        const double throughput =
            static_cast<double>(total_inferences) / duration_s;
        starpu_server::log_stats(
            opts.verbosity,
            std::format(
                "Throughput: {:.3f} inf/s ({} inferences over {:.3f} s)",
                throughput, total_inferences, duration_s));
      }
    }
  }
  catch (const starpu_server::InferenceEngineException& e) {
    starpu_server::log_error(std::format("Inference Error: {}", e.what()));
    return 2;
  }
  catch (const std::exception& e) {
    starpu_server::log_error(std::format("General Error: {}", e.what()));
    return -1;
  }

  return 0;
}
