#include <gtest/gtest.h>
#include <torch/script.h>

#include <filesystem>
#include <string>

#include "../core/inference_runner_test_utils.hpp"
#include "../test_helpers.hpp"
#include "core/inference_runner.hpp"
#include "core/starpu_setup.hpp"
#include "runtime_config.hpp"

namespace {
void
run_add_one_integration_test(
    bool use_cpu, bool use_cuda, std::optional<int> device_id = std::nullopt)
{
  auto model = starpu_server::make_add_one_model();
  const std::filesystem::path model_path =
      std::filesystem::temp_directory_path() / "add_one.pt";
  model.save(model_path.string());
  starpu_server::RuntimeConfig opts;
  opts.model_path = model_path.string();
  opts.input_shapes = {{1}};
  opts.input_types = {at::kFloat};
  opts.iterations = 1;
  opts.use_cpu = use_cpu;
  opts.use_cuda = use_cuda;
  if (device_id) {
    opts.device_ids = {*device_id};
  }
  opts.verbosity = starpu_server::VerbosityLevel::Info;
  starpu_server::StarPUSetup starpu(opts);
  starpu_server::CaptureStream capture{std::cout};
  starpu_server::run_inference_loop(opts, starpu);
  const std::string output = capture.str();
  std::filesystem::remove(model_path);
  EXPECT_NE(output.find("Job 0 passed"), std::string::npos);
}
}  // namespace

TEST(RunInferenceLoopIntegration, CpuAddOneModel)
{
  run_add_one_integration_test(true, false);
}

TEST(RunInferenceLoopIntegration, CudaAddOneModel)
{
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available";
  }
  run_add_one_integration_test(false, true, 0);
}
