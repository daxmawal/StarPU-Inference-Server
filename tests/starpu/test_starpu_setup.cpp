#include <gtest/gtest.h>
#include <torch/script.h>

#include <functional>
#include <vector>

#include "core/inference_params.hpp"
#include "core/starpu_setup.hpp"
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
          at::Tensor t1 = torch::ones({2, 2});
          at::Tensor t2 = torch::zeros({1, 3});
          auto tuple = c10::ivalue::Tuple::create({t1, t2});
          return ExtractTensorsParam{c10::IValue(tuple), {t1, t2}};
        }(),
        []() {
          at::Tensor t1 = torch::rand({2});
          at::Tensor t2 = torch::rand({3});
          c10::List<at::Tensor> list;
          list.push_back(t1);
          list.push_back(t2);
          return ExtractTensorsParam{c10::IValue(list), {t1, t2}};
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
