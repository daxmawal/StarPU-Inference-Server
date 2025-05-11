#include "args_parser.hpp"
#include "inference_runner.hpp"
#include "starpu_setup.hpp"

int main(int argc, char* argv[]) {
  ProgramOptions opts = parse_arguments(argc, argv);
  if (opts.show_help) {
    display_help("Inference Engine");
    return 0;
  }
  if (!opts.valid) {
    return 1;
  }

  std::cout << "Scheduler  : " << opts.scheduler << "\n";
  std::cout << "Iterations : " << opts.iterations << "\n";

  try {
    StarPUSetup starpu(opts.scheduler.c_str());
    run_inference_loop(opts, starpu);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return -1;
  }

  return 0;
}