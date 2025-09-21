#include <gtest/gtest.h>
#include <torch/script.h>

#include <filesystem>
#include <optional>
#include <string>

#include "core/inference_runner.hpp"
#include "core/starpu_setup.hpp"
#include "runtime_config.hpp"
#include "test_helpers.hpp"

namespace {
auto
run_add_one_integration_test(
    bool use_cpu, bool use_cuda, std::optional<int> device_id = std::nullopt,
    bool validate_results = true) -> std::string
{
  auto model = starpu_server::make_add_one_model();
  const std::filesystem::path model_path =
      std::filesystem::temp_directory_path() / "add_one.pt";
  model.save(model_path.string());
  starpu_server::RuntimeConfig opts;
  opts.models.resize(1);
  opts.models[0].path = model_path.string();
  opts.models[0].inputs = {{"input0", {1}, at::kFloat}};
  opts.iterations = 1;
  opts.use_cpu = use_cpu;
  opts.use_cuda = use_cuda;
  opts.validate_results = validate_results;
  if (device_id) {
    opts.device_ids = {*device_id};
  }
  opts.verbosity = starpu_server::VerbosityLevel::Info;
  starpu_server::StarPUSetup starpu(opts);
  starpu_server::CaptureStream capture{std::cout};
  starpu_server::run_inference_loop(opts, starpu);
  const std::string output = capture.str();
  std::filesystem::remove(model_path);
  return output;
}
}  // namespace

TEST(RunInferenceLoopIntegration, CpuAddOneModel)
{
  const auto output = run_add_one_integration_test(true, false);
  EXPECT_NE(output.find("Job 0 passed"), std::string::npos);
}

TEST(RunInferenceLoopIntegration, CudaAddOneModel)
{
  if (!torch::cuda::is_available()) {
    GTEST_SKIP() << "CUDA is not available";
  }
  const auto output = run_add_one_integration_test(false, true, 0);
  EXPECT_NE(output.find("Job 0 passed"), std::string::npos);
}

TEST(RunInferenceLoopIntegration, DisableValidationSkipsChecks)
{
  const auto output =
      run_add_one_integration_test(true, false, std::nullopt, false);
  EXPECT_EQ(output.find("Job 0 passed"), std::string::npos);
  EXPECT_NE(output.find("Result validation disabled"), std::string::npos);
}
