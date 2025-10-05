#include <torch/version.h>

#include <exception>
#include <format>
#include <iterator>
#include <memory>
#include <span>
#include <string_view>

#include "args_parser.hpp"
#include "config_loader.hpp"
#include "exceptions.hpp"
#include "inference_runner.hpp"
#include "logger.hpp"
#include "perf_observer.hpp"
#include "runtime_config.hpp"
#include "starpu_setup.hpp"

auto
main(int argc, char* argv[]) -> int
{
  std::span<char*> args_span(argv, static_cast<size_t>(argc));

  std::string config_path;
  auto args_without_program = args_span.subspan(1);
  for (auto it = args_without_program.begin(); it != args_without_program.end();
       ++it) {
    std::string_view arg(*it);
    if (arg == "--config" || arg == "-c") {
      if (auto value_it = std::next(it);
          value_it != args_without_program.end()) {
        config_path = *value_it;
      }
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
    starpu_server::perf_observer::reset();
    starpu = std::make_unique<starpu_server::StarPUSetup>(opts);
    starpu_server::run_inference_loop(opts, *starpu);
    if (const auto stats = starpu_server::perf_observer::snapshot()) {
      starpu_server::log_stats(
          opts.verbosity,
          std::format(
              "Throughput: {:.3f} inf/s ({} inferences over {:.3f} s)",
              stats->throughput, stats->total_inferences,
              stats->duration_seconds));
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
