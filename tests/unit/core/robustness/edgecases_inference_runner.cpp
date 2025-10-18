#include <gtest/gtest.h>
#include <torch/script.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/inference_params.hpp"
#include "core/inference_runner.hpp"
#include "core/tensor_builder.hpp"
#include "test_constants.hpp"
#include "test_helpers.hpp"
#include "test_inference_runner.hpp"


namespace {
const std::vector<int64_t> kShape1{1};
const std::vector<torch::Dtype> kTypesFloat{torch::kFloat32};
using starpu_server::test_constants::kF1;
using starpu_server::test_constants::kF2;
using starpu_server::test_constants::kF3;

class ConstantModelConfigTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    model_file_.emplace(
        "constant_model_fixture", starpu_server::make_constant_model());
    base_config_ = starpu_server::make_single_model_runtime_config(
        model_file_->path(), kShape1, at::kFloat);
  }

  [[nodiscard]] starpu_server::RuntimeConfig base_config() const
  {
    return base_config_;
  }

  [[nodiscard]] starpu_server::RuntimeConfig cuda_config(
      std::vector<int> device_ids) const
  {
    auto config = base_config();
    config.devices.use_cuda = true;
    config.devices.ids = std::move(device_ids);
    return config;
  }

  [[nodiscard]] const std::filesystem::path& model_path() const
  {
    return model_file_->path();
  }

 private:
  starpu_server::RuntimeConfig base_config_{};
  std::optional<starpu_server::TemporaryModelFile> model_file_;
};

struct WorkerFailOutcome {
  bool threw_runtime_error;
  std::string log;
};

inline auto
RunWorkerThreadFailureCase(const std::filesystem::path& path)
    -> WorkerFailOutcome
{
  using namespace starpu_server;

  auto opts = make_single_model_runtime_config(path, {1}, at::kFloat);
  opts.batching.request_nb = 1;
  opts.devices.use_cuda = false;

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

TEST_F(ConstantModelConfigTest, LoadModelAndReferenceOutputUnsupported)
{
  auto opts = base_config();
  opts.devices.ids = {0};
  opts.devices.use_cuda = false;

  torch::manual_seed(3);
  EXPECT_THROW(
      (void)starpu_server::load_model_and_reference_output(opts),
      starpu_server::UnsupportedModelOutputTypeException);
}

TEST_F(ConstantModelConfigTest, CloneModelToGpus_InvalidDeviceIdThrows)
{
  auto opts = cuda_config(std::vector<int>{-1});

  EXPECT_THROW(
      (void)starpu_server::load_model_and_reference_output(opts),
      std::runtime_error);
}

namespace starpu_server {
void run_inference(
    InferenceParams* params, const std::vector<void*>& buffers,
    torch::Device device, torch::jit::script::Module* model,
    const std::function<void(const at::Tensor&, std::span<std::byte>)>&
        copy_output_fn);
}

TEST(StarPUSetupRunInference_Integration, BuildsExecutesCopiesAndTimes)
{
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
      [](const at::Tensor& out, std::span<std::byte> buffer) {
        starpu_server::TensorBuilder::copy_output_to_buffer(
            out, buffer, out.numel(), out.scalar_type());
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
  auto opts = starpu_server::make_single_model_runtime_config(
      "nonexistent_model.pt", std::vector<int64_t>{1}, at::kFloat);

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

  auto opts = make_single_model_runtime_config(
      "nonexistent_model.pt", std::vector<int64_t>{1}, at::kFloat);
  opts.batching.request_nb = 1;
  opts.devices.use_cuda = false;

  StarPUSetup starpu(opts);

  CaptureStream capture{std::cerr};
  run_inference_loop(opts, starpu);
  EXPECT_NE(capture.str().find("Failed to load model"), std::string::npos);
}

TEST(RunInferenceLoop_Robustesse, WorkerThreadExceptionTriggersShutdown)
{
  using namespace starpu_server;

  starpu_server::TemporaryModelFile model_file(
      "worker_fail", starpu_server::make_identity_model());

  const auto outcome = RunWorkerThreadFailureCase(model_file.path());
  EXPECT_TRUE(outcome.threw_runtime_error);
  EXPECT_NE(
      outcome.log.find("Failed to start worker thread: boom"),
      std::string::npos);
}

TEST_F(ConstantModelConfigTest, InvalidCudaDeviceLogsError)
{
  using namespace starpu_server;

  auto opts = cuda_config(std::vector<int>{-1});
  opts.batching.request_nb = 1;
  opts.batching.warmup_request_nb = 0;

  StarPUSetup starpu(opts);

  CaptureStream capture{std::cerr};
  EXPECT_NO_THROW(run_inference_loop(opts, starpu));
  const auto log = capture.str();
  EXPECT_NE(
      log.find("Failed to load model or reference outputs:"),
      std::string::npos);
}
