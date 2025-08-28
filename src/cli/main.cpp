#include <torch/version.h>

#include <exception>
#include <iostream>
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

  std::cout << "__cplusplus = " << __cplusplus << "\n"
            << "LibTorch version: " << TORCH_VERSION << "\n"
            << "Scheduler       : " << opts.scheduler << "\n"
            << "Iterations      : " << opts.iterations << "\n";

  std::unique_ptr<starpu_server::StarPUSetup> starpu;
  try {
    starpu = std::make_unique<starpu_server::StarPUSetup>(opts);
    starpu_server::run_inference_loop(opts, *starpu);
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
