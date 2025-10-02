#include <gtest/gtest.h>
#include <torch/script.h>

#include <chrono>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "core/inference_runner.hpp"
#include "test_helpers.hpp"
#include "test_inference_runner.hpp"


namespace {
constexpr int kDeviceId0 = 0;
const std::vector<int64_t> kShape4{4};
const std::vector<int64_t> kShape2{2};
const std::vector<torch::Dtype> kTypesFloat{torch::kFloat32};
constexpr int64_t kSeed42 = 42;
constexpr int64_t kSeed1 = 1;
constexpr int64_t kSeed2 = 2;
}  // namespace

struct TestParam {
  std::string name;
  std::function<void(const std::filesystem::path&)> save_module;
  std::vector<starpu_server::TensorConfig> inputs;
  int64_t seed;
  std::size_t expected_outputs;
  std::function<void(
      const std::vector<torch::Tensor>&, const std::vector<torch::Tensor>&)>
      verify;
};

class LoadModelAndReferenceOutputTest
    : public ::testing::TestWithParam<TestParam> {
 protected:
  static void RunScenario(const TestParam& param)
  {
    const auto file = MakeTempModelPath(param.name.c_str());
    param.save_module(file);

    starpu_server::RuntimeConfig opts;
    opts.models.resize(1);
    opts.models[0].path = file.string();
    opts.models[0].inputs = param.inputs;
    opts.device_ids = {kDeviceId0};
    opts.use_cuda = false;

    torch::manual_seed(param.seed);
    auto result = starpu_server::load_model_and_reference_output(opts);
    ASSERT_TRUE(result.has_value());
    const auto& [cpu_model, gpu_models, refs] = result.value();
    (void)cpu_model;
    EXPECT_TRUE(gpu_models.empty());

    torch::manual_seed(param.seed);
    auto inputs = starpu_server::generate_inputs(opts.models[0].inputs);
    ASSERT_EQ(refs.size(), param.expected_outputs);
    param.verify(refs, inputs);

    std::filesystem::remove(file);
  }
};

TEST_P(LoadModelAndReferenceOutputTest, GeneratesExpectedReferences)
{
  RunScenario(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    InferenceRunner_Integration, LoadModelAndReferenceOutputTest,
    ::testing::Values(
        TestParam{
            "LoadModelAndReferenceOutputCPU",
            [](const std::filesystem::path& file) {
              starpu_server::save_mul_two_model(file);
            },
            {{"input0", kShape4, at::kFloat}},
            kSeed42,
            1U,
            [](const std::vector<torch::Tensor>& refs,
               const std::vector<torch::Tensor>& inputs) {
              ASSERT_EQ(inputs.size(), 1U);
              ASSERT_EQ(refs.size(), 1U);
              EXPECT_TRUE(torch::allclose(refs[0], inputs[0] * 2));
            }},
        TestParam{
            "LoadModelAndReferenceOutputTuple",
            [](const std::filesystem::path& file) {
              auto model = starpu_server::make_tuple_model();
              model.save(file.string());
            },
            {{"input0", kShape2, at::kFloat}},
            kSeed1,
            2U,
            [](const std::vector<torch::Tensor>& refs,
               const std::vector<torch::Tensor>& inputs) {
              ASSERT_EQ(inputs.size(), 1U);
              ASSERT_EQ(refs.size(), 2U);
              EXPECT_TRUE(torch::allclose(refs[0], inputs[0]));
              EXPECT_TRUE(torch::allclose(refs[1], inputs[0] + 1));
            }},
        TestParam{
            "LoadModelAndReferenceOutputTensorList",
            [](const std::filesystem::path& file) {
              auto model = starpu_server::make_tensor_list_model();
              model.save(file.string());
            },
            {{"input0", kShape2, at::kFloat}},
            kSeed2,
            2U,
            [](const std::vector<torch::Tensor>& refs,
               const std::vector<torch::Tensor>& inputs) {
              ASSERT_EQ(inputs.size(), 1U);
              ASSERT_EQ(refs.size(), 2U);
              EXPECT_TRUE(torch::allclose(refs[0], inputs[0]));
              EXPECT_TRUE(torch::allclose(refs[1], inputs[0] + 1));
            }}),
    [](const ::testing::TestParamInfo<TestParam>& info) {
      return info.param.name;
    });
