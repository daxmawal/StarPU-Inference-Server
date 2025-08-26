#include <gtest/gtest.h>
#include <torch/script.h>

#include <array>
#include <chrono>
#include <filesystem>
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
}  // namespace

TEST(InferenceRunner_Robustesse, LoadModelAndReferenceOutputUnsupported)
{
  const auto file = MakeTempModelPath("constant_module");
  auto model = starpu_server::make_constant_model();
  model.save(file.string());

  starpu_server::RuntimeConfig opts;
  opts.model_path = file.string();
  opts.inputs = {{"input0", kShape1, at::kFloat}};
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
    const torch::Device device, torch::jit::script::Module* model,
    const std::function<void(const at::Tensor&, void* buffer_ptr)>&
        copy_output_fn);
}

TEST(StarPUSetupRunInference_Integration, BuildsExecutesCopiesAndTimes)
{
  std::array<float, 3> input{1.0F, 2.0F, 3.0F};
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
            out, buffer_ptr, out.numel());
      });
  auto after = std::chrono::high_resolution_clock::now();

  EXPECT_FLOAT_EQ(output[0], 2.0F);
  EXPECT_FLOAT_EQ(output[1], 3.0F);
  EXPECT_FLOAT_EQ(output[2], 4.0F);
  EXPECT_TRUE(inference_start >= before);
  EXPECT_TRUE(inference_start <= after);
}

namespace {
template <typename T, template <typename...> class Template>
struct is_specialization : std::false_type {};

template <template <typename...> class Template, typename... Args>
struct is_specialization<Template<Args...>, Template> : std::true_type {};
}  // namespace

TEST(InferenceRunner_Robustesse, LoadModelMissingFile)
{
  starpu_server::RuntimeConfig opts;
  opts.model_path = "nonexistent_model.pt";
  opts.inputs = {{"input0", {1}, at::kFloat}};

  try {
    auto result = starpu_server::load_model_and_reference_output(opts);
    if constexpr (is_specialization<decltype(result), std::optional>::value) {
      EXPECT_EQ(result, std::nullopt);
    } else {
      EXPECT_TRUE(std::get<1>(result).empty());
      EXPECT_TRUE(std::get<2>(result).empty());
    }
  }
  catch (const std::exception&) {
    SUCCEED();
  }
}
