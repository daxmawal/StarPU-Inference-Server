#include <torch/version.h>

#include <exception>
#include <iostream>

#include "args_parser.hpp"
#include "exceptions.hpp"
#include "inference_runner.hpp"
#include "logger.hpp"
#include "starpu_setup.hpp"

auto
main(int argc, char* argv[]) -> int
{
  // Parse and validate command-line options
  const ProgramOptions opts = parse_arguments(argc, argv);

  if (opts.show_help) {
    display_help("Inference Engine");
    return 0;
  }

  if (!opts.valid) {
    log_fatal("Invalid program options.\n");
  }

  // Display configuration summary
  std::cout << "LibTorch version: " << TORCH_VERSION << "\n"
            << "Scheduler       : " << opts.scheduler << "\n"
            << "Iterations      : " << opts.iterations << "\n";

  // Launch inference process
  try {
    StarPUSetup starpu(opts);
    run_inference_loop(opts, starpu);
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
