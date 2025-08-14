#include <gtest/gtest.h>
#include <torch/script.h>

#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

#include "core/inference_runner.hpp"
#include "test_inference_runner.hpp"


namespace {
constexpr int kDeviceId0 = 0;
const std::vector<int64_t> kShape4{4};
const std::vector<int64_t> kShape2{2};
const std::vector<torch::Dtype> kTypesFloat{torch::kFloat32};

inline std::filesystem::path
MakeTempModelPath(const char* base)
{
  const auto dir = std::filesystem::temp_directory_path();
  const auto ts =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  return dir / (std::string(base) + "_" + std::to_string(ts) + ".pt");
}
}  // namespace

TEST(InferenceRunner_Integration, LoadModelAndReferenceOutputCPU)
{
  const auto file = MakeTempModelPath("tiny_module");
  starpu_server::save_mul_two_model(file);

  starpu_server::RuntimeConfig opts;
  opts.model_path = file.string();
  opts.input_shapes = {kShape4};
  opts.input_types = kTypesFloat;
  opts.device_ids = {kDeviceId0};
  opts.use_cuda = false;

  torch::manual_seed(42);
  auto [cpu_model, gpu_models, refs] =
      starpu_server::load_model_and_reference_output(opts);
  EXPECT_TRUE(gpu_models.empty());

  torch::manual_seed(42);
  auto inputs =
      starpu_server::generate_inputs(opts.input_shapes, opts.input_types);
  ASSERT_EQ(refs.size(), 1U);
  EXPECT_TRUE(torch::allclose(refs[0], inputs[0] * 2));

  std::filesystem::remove(file);
}

TEST(InferenceRunner_Integration, LoadModelAndReferenceOutputTuple)
{
  const auto file = MakeTempModelPath("tuple_module");
  auto model = starpu_server::make_tuple_model();
  model.save(file.string());

  starpu_server::RuntimeConfig opts;
  opts.model_path = file.string();
  opts.input_shapes = {kShape2};
  opts.input_types = kTypesFloat;
  opts.device_ids = {kDeviceId0};
  opts.use_cuda = false;

  torch::manual_seed(1);
  auto [cpu_model, gpu_models, refs] =
      starpu_server::load_model_and_reference_output(opts);
  EXPECT_TRUE(gpu_models.empty());

  torch::manual_seed(1);
  auto inputs =
      starpu_server::generate_inputs(opts.input_shapes, opts.input_types);
  ASSERT_EQ(refs.size(), 2U);
  EXPECT_TRUE(torch::allclose(refs[0], inputs[0]));
  EXPECT_TRUE(torch::allclose(refs[1], inputs[0] + 1));

  std::filesystem::remove(file);
}

TEST(InferenceRunner_Integration, LoadModelAndReferenceOutputTensorList)
{
  const auto file = MakeTempModelPath("tensor_list_module");
  auto model = starpu_server::make_tensor_list_model();
  model.save(file.string());

  starpu_server::RuntimeConfig opts;
  opts.model_path = file.string();
  opts.input_shapes = {kShape2};
  opts.input_types = kTypesFloat;
  opts.device_ids = {kDeviceId0};
  opts.use_cuda = false;

  torch::manual_seed(2);
  auto [cpu_model, gpu_models, refs] =
      starpu_server::load_model_and_reference_output(opts);
  EXPECT_TRUE(gpu_models.empty());

  torch::manual_seed(2);
  auto inputs =
      starpu_server::generate_inputs(opts.input_shapes, opts.input_types);
  ASSERT_EQ(refs.size(), 2U);
  EXPECT_TRUE(torch::allclose(refs[0], inputs[0]));
  EXPECT_TRUE(torch::allclose(refs[1], inputs[0] + 1));

  std::filesystem::remove(file);
}
