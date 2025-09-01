#include <gtest/gtest.h>
#include <torch/script.h>

#include <array>
#include <chrono>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/inference_params.hpp"
#include "core/inference_runner.hpp"
#include "core/tensor_builder.hpp"
#include "test_helpers.hpp"
#include "test_inference_runner.hpp"


namespace {
const std::vector<int64_t> kShape1{1};
const std::vector<torch::Dtype> kTypesFloat{torch::kFloat32};

inline auto
MakeTempModelPath(const char* base) -> std::filesystem::path
{
  const auto dir = std::filesystem::temp_directory_path();
  const auto time =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  return dir / (std::string(base) + "_" + std::to_string(time) + ".pt");
}

struct WorkerFailOutcome {
  bool threw_runtime_error;
  std::string log;
};

inline auto
RunWorkerThreadFailureCase(const std::filesystem::path& path)
    -> WorkerFailOutcome
{
  using namespace starpu_server;

  RuntimeConfig opts;
  opts.models.resize(1);
  opts.models[0].path = path.string();
  opts.models[0].inputs = {{"input0", {1}, at::kFloat}};
  opts.iterations = 1;
  opts.use_cuda = false;

  StarPUSetup starpu(opts);

  auto original_launcher = get_worker_thread_launcher();
  set_worker_thread_launcher([](StarPUTaskRunner&) -> std::jthread {
    throw std::runtime_error("boom");
  });

  CaptureStream capture{std::cerr};
  bool threw = false;
  try {
    run_inference_loop(opts, starpu);
  }
  catch (const std::runtime_error&) {
    threw = true;
  }
  auto log = capture.str();
  set_worker_thread_launcher(original_launcher);
  return WorkerFailOutcome{threw, std::move(log)};
}
}  // namespace

TEST(InferenceRunner_Robustesse, LoadModelAndReferenceOutputUnsupported)
{
  const auto file = MakeTempModelPath("constant_module");
  auto model = starpu_server::make_constant_model();
  model.save(file.string());

  starpu_server::RuntimeConfig opts;
  opts.models.resize(1);
  opts.models[0].path = file.string();
  opts.models[0].inputs = {{"input0", kShape1, at::kFloat}};
  opts.device_ids = {0};
  opts.use_cuda = false;

  torch::manual_seed(3);
  EXPECT_THROW(
      (void)starpu_server::load_model_and_reference_output(opts),
      starpu_server::UnsupportedModelOutputTypeException);

  std::filesystem::remove(file);
}

namespace starpu_server {
void run_inference(
    InferenceParams* params, const std::vector<void*>& buffers,
    torch::Device device, torch::jit::script::Module* model,
    const std::function<void(const at::Tensor&, void* buffer_ptr)>&
        copy_output_fn);
}

TEST(StarPUSetupRunInference_Integration, BuildsExecutesCopiesAndTimes)
{
  constexpr float kF1 = 1.0F;
  constexpr float kF2 = 2.0F;
  constexpr float kF3 = 3.0F;
  std::array<float, 3> input{kF1, kF2, kF3};
  std::array<float, 3> output{0.0F, 0.0F, 0.0F};

  auto input_iface = starpu_server::make_variable_interface(input.data());
  auto output_iface = starpu_server::make_variable_interface(output.data());

  auto params = starpu_server::make_basic_params(3);
  std::chrono::high_resolution_clock::time_point inference_start;
  params.timing.inference_start_time = &inference_start;

  std::vector<void*> buffers = {&input_iface, &output_iface};
  auto model = starpu_server::make_add_one_model();

  auto before = std::chrono::high_resolution_clock::now();
  starpu_server::run_inference(
      &params, buffers, torch::Device(torch::kCPU), &model,
      [](const at::Tensor& out, void* buffer_ptr) {
        starpu_server::TensorBuilder::copy_output_to_buffer(
            out, buffer_ptr, out.numel(), out.scalar_type());
      });
  auto after = std::chrono::high_resolution_clock::now();

  EXPECT_FLOAT_EQ(output[0], 2.0F);
  EXPECT_FLOAT_EQ(output[1], 3.0F);
  EXPECT_FLOAT_EQ(output[2], 4.0F);
  EXPECT_TRUE(inference_start >= before);
  EXPECT_TRUE(inference_start <= after);
}

TEST(InferenceRunner_Robustesse, LoadModelMissingFile)
{
  starpu_server::RuntimeConfig opts;
  opts.models.resize(1);
  opts.models[0].path = "nonexistent_model.pt";
  opts.models[0].inputs = {{"input0", {1}, at::kFloat}};

  try {
    auto result = starpu_server::load_model_and_reference_output(opts);
    EXPECT_EQ(result, std::nullopt);
  }
  catch (const std::exception&) {
    SUCCEED();
  }
}

TEST(RunInferenceLoop_Robustesse, LoadModelFailureHandledGracefully)
{
  using namespace starpu_server;

  RuntimeConfig opts;
  opts.models.resize(1);
  opts.models[0].path = "nonexistent_model.pt";
  opts.models[0].inputs = {{"input0", {1}, at::kFloat}};
  opts.iterations = 1;
  opts.use_cuda = false;

  StarPUSetup starpu(opts);

  CaptureStream capture{std::cerr};
  run_inference_loop(opts, starpu);
  EXPECT_NE(capture.str().find("Failed to load model"), std::string::npos);
}

TEST(RunInferenceLoop_Robustesse, WorkerThreadExceptionTriggersShutdown)
{
  using namespace starpu_server;

  const auto model_path =
      std::filesystem::temp_directory_path() / "worker_fail.pt";
  make_identity_model().save(model_path.string());

  const auto outcome = RunWorkerThreadFailureCase(model_path);
  EXPECT_TRUE(outcome.threw_runtime_error);
  EXPECT_NE(
      outcome.log.find("Failed to start worker thread: boom"),
      std::string::npos);
  std::filesystem::remove(model_path);
}
