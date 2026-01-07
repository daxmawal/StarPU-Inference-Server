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

using starpu_server::StarpuBufferPtr;

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
    InferenceParams* params, const std::vector<StarpuBufferPtr>& buffers,
    torch::Device device, torch::jit::script::Module* model,
    const std::function<void(const at::Tensor&, std::span<std::byte>)>&
        copy_output_fn);
}

TEST(StarPUSetupRunInference_Integration, BuildsExecutesCopiesAndTimes)
{
  std::array<float, 3> input{kF1, kF2, kF3};
  std::array<float, 3> output{0.0F, 0.0F, 0.0F};

  auto input_iface =
      starpu_server::make_variable_interface(input.data(), input.size());
  auto output_iface =
      starpu_server::make_variable_interface(output.data(), output.size());

  auto params = starpu_server::make_basic_params(3);
  starpu_server::MonotonicClock::time_point inference_start;
  params.timing.inference_start_time = &inference_start;

  std::vector<StarpuBufferPtr> buffers = {&input_iface, &output_iface};
  auto model = starpu_server::make_add_one_model();

  auto before = starpu_server::MonotonicClock::now();
  starpu_server::run_inference(
      &params, buffers, torch::Device(torch::kCPU), &model,
      [](const at::Tensor& out, std::span<std::byte> buffer) {
        starpu_server::TensorBuilder::copy_output_to_buffer(
            out, buffer, out.numel(), out.scalar_type());
      });
  auto after = starpu_server::MonotonicClock::now();

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
