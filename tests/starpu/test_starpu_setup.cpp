#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <starpu.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <chrono>
#include <climits>
#include <functional>
#include <memory>
#include <vector>

#include "core/inference_params.hpp"
#include "core/starpu_setup.hpp"
#include "core/tensor_builder.hpp"
#include "test_utils.hpp"
#include "utils/exceptions.hpp"

struct ExtractTensorsParam {
  c10::IValue input;
  std::vector<at::Tensor> expected;
};

class StarPUSetupExtractTensorsTest
    : public ::testing::TestWithParam<ExtractTensorsParam> {};

TEST(StarPUSetupErrorsTest, GetCudaWorkersNegativeDeviceThrows)
{
  EXPECT_THROW(
      starpu_server::StarPUSetup::get_cuda_workers_by_device({-1}),
      std::invalid_argument);
}

TEST(StarPUSetupErrorsTest, ExtractTensorsFromOutputUnsupportedType)
{
  c10::IValue non_tensor{42};
  EXPECT_THROW(
      starpu_server::extract_tensors_from_output(non_tensor),
      starpu_server::UnsupportedModelOutputTypeException);
}

TEST_P(StarPUSetupExtractTensorsTest, Extract)
{
  const auto& param = GetParam();
  auto outputs = starpu_server::extract_tensors_from_output(param.input);
  ASSERT_EQ(outputs.size(), param.expected.size());
  for (size_t i = 0; i < param.expected.size(); ++i) {
    EXPECT_TRUE(outputs[i].equal(param.expected[i]));
  }
}

INSTANTIATE_TEST_SUITE_P(
    StarPUSetupExtractTensorsTestCases, StarPUSetupExtractTensorsTest,
    ::testing::Values(
        []() {
          at::Tensor tensor = torch::arange(6).view({2, 3});
          return ExtractTensorsParam{c10::IValue(tensor), {tensor}};
        }(),
        []() {
          at::Tensor tensor_1 = torch::ones({2, 2});
          at::Tensor tensor_2 = torch::zeros({1, 3});
          auto tuple = c10::ivalue::Tuple::create({tensor_1, tensor_2});
          return ExtractTensorsParam{c10::IValue(tuple), {tensor_1, tensor_2}};
        }(),
        []() {
          at::Tensor tensor_1 = torch::rand({2});
          at::Tensor tensor_2 = torch::rand({3});
          c10::List<at::Tensor> list;
          list.push_back(tensor_1);
          list.push_back(tensor_2);
          return ExtractTensorsParam{c10::IValue(list), {tensor_1, tensor_2}};
        }()),
    [](const ::testing::TestParamInfo<ExtractTensorsParam>& info) {
      switch (info.index) {
        case 0:
          return std::string{"SingleTensor"};
        case 1:
          return std::string{"TupleOfTensors"};
        case 2:
          return std::string{"TensorList"};
        default:
          return std::string{"Unknown"};
      }
    });

class StarPUSetupCodeletTest : public ::testing::Test {
 protected:
  std::unique_ptr<starpu_server::StarPUSetup> starpu;
  void SetUp() override
  {
    if (!torch::cuda::is_available()) {
      GTEST_SKIP();
    }
    starpu_server::RuntimeConfig opts;
    opts.use_cpu = true;
    opts.use_cuda = true;
    opts.device_ids = {0};
    starpu = std::make_unique<starpu_server::StarPUSetup>(opts);
  }
};

TEST_F(StarPUSetupCodeletTest, GetCodeletNotNull)
{
  EXPECT_NE(starpu->get_codelet(), nullptr);
}

TEST_F(StarPUSetupCodeletTest, GetCudaWorkersSingleDevice)
{
  auto workers = starpu_server::StarPUSetup::get_cuda_workers_by_device({0});
  EXPECT_FALSE(workers.empty());
}

TEST(InferenceCodelet, FieldsAreInitialized)
{
  starpu_server::InferenceCodelet inf_cl;
  auto* codelet = inf_cl.get_codelet();
  EXPECT_EQ(codelet->nbuffers, STARPU_VARIABLE_NBUFFERS);
  EXPECT_NE(codelet->cpu_funcs[0], nullptr);
  EXPECT_NE(codelet->cuda_funcs[0], nullptr);
  EXPECT_EQ(codelet->cuda_flags[0], 1U);
  EXPECT_EQ(codelet->max_parallelism, INT_MAX);
}

TEST(InferenceCodelet, CpuInferenceFuncExecutesAndSetsMetadata)
{
  StarpuRuntimeGuard starpu_guard;
  auto buffers = make_test_buffers();
  auto timing = setup_timing_params(3);
  starpu_server::InferenceCodelet inf_cl;
  auto* codelet = inf_cl.get_codelet();
  auto before = std::chrono::high_resolution_clock::now();
  codelet->cpu_funcs[0](buffers.buffers.data(), &timing.params);
  auto after = std::chrono::high_resolution_clock::now();
  EXPECT_FLOAT_EQ(buffers.output_data[0], 2.0F);
  EXPECT_FLOAT_EQ(buffers.output_data[1], 3.0F);
  EXPECT_FLOAT_EQ(buffers.output_data[2], 4.0F);
  EXPECT_EQ(timing.executed_on, starpu_server::DeviceType::CPU);
  EXPECT_TRUE(timing.start_time >= before);
  EXPECT_TRUE(timing.end_time <= after);
  EXPECT_TRUE(timing.end_time >= timing.start_time);
}
