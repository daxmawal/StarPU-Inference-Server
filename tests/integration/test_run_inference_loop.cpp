#include <gtest/gtest.h>
#include <torch/script.h>

#include <filesystem>
#include <string>

#include "../core/inference_runner_test_utils.hpp"
#include "../test_helpers.hpp"
#include "core/inference_runner.hpp"
#include "core/starpu_setup.hpp"
#include "runtime_config.hpp"

namespace fs = std::filesystem;

TEST(RunInferenceLoopIntegration, CpuAddOneModel)
{
  auto model = starpu_server::make_add_one_model();
  const fs::path model_path = fs::temp_directory_path() / "add_one.pt";
  model.save(model_path.string());

  starpu_server::RuntimeConfig opts;
  opts.model_path = model_path.string();
  opts.input_shapes = {{1}};
  opts.input_types = {at::kFloat};
  opts.iterations = 1;
  opts.use_cpu = true;
  opts.use_cuda = false;
  opts.verbosity = starpu_server::VerbosityLevel::Info;

  starpu_server::StarPUSetup starpu(opts);

  starpu_server::CaptureStream capture{std::cout};
  starpu_server::run_inference_loop(opts, starpu);
  const std::string output = capture.str();

  fs::remove(model_path);

  EXPECT_NE(output.find("Job 0 passed"), std::string::npos);
}
