#include <torch/torch.h>

#include "exceptions.hpp"
#include "inference_runner.hpp"

int
main(int argc, char* argv[])
{
  const ProgramOptions opts = parse_arguments(argc, argv);
  if (opts.show_help) {
    display_help("Inference Engine");
    return 0;
  }
  if (!opts.valid) {
    return 1;
  }

  std::cout << "LibTorch version: " << TORCH_VERSION << std::endl;
  std::cout << "Scheduler  : " << opts.scheduler << "\n";
  std::cout << "Iterations : " << opts.iterations << "\n";

  try {
    StarPUSetup starpu(opts);
    run_inference_loop(opts, starpu);
  }
  catch (const InferenceEngineException& e) {
    std::cerr << "[Inference Error] " << e.what() << std::endl;
    return 2;
  }
  catch (const std::exception& e) {
    std::cerr << "[General Error] " << e.what() << std::endl;
    return -1;
  }

  return 0;
}