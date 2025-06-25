#include <torch/version.h>

#include <exception>
#include <iostream>

#include "args_parser.hpp"
#include "exceptions.hpp"
#include "inference_runner.hpp"
#include "logger.hpp"
#include "runtime_config.hpp"
#include "starpu_setup.hpp"

auto
main(int argc, char* argv[]) -> int
{
  // Parse and validate command-line options
  const starpu_server::RuntimeConfig opts = starpu_server::parse_arguments(
      std::span<char*>(argv, static_cast<size_t>(argc)));

  if (opts.show_help) {
    starpu_server::display_help("Inference Engine");
    return 0;
  }

  if (!opts.valid) {
    starpu_server::log_fatal("Invalid program options.\n");
  }

  // Display configuration summary
  std::cout << "__cplusplus = " << __cplusplus << "\n"
            << "LibTorch version: " << TORCH_VERSION << "\n"
            << "Scheduler       : " << opts.scheduler << "\n"
            << "Iterations      : " << opts.iterations << "\n";

  // Launch inference process
  try {
    starpu_server::StarPUSetup starpu(opts);
    starpu_server::run_inference_loop(opts, starpu);
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
