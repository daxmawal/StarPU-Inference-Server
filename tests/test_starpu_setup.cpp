#include <gtest/gtest.h>
#include <torch/script.h>

#include <functional>
#include <vector>

#include "core/inference_params.hpp"
#include "core/starpu_setup.hpp"
#include "utils/exceptions.hpp"

using namespace starpu_server;

TEST(StarPUSetupErrorsTest, ConstructorNegativeDeviceId)
{
  RuntimeConfig cfg;
  cfg.use_cuda = true;
  cfg.device_ids = {-1};
  EXPECT_THROW(StarPUSetup setup(cfg), std::invalid_argument);
}

TEST(StarPUSetupErrorsTest, GetCudaWorkersByDeviceNegativeId)
{
  EXPECT_THROW(
      StarPUSetup::get_cuda_workers_by_device({-1}), std::invalid_argument);
}

TEST(StarPUSetupErrorsTest, ExtractTensorsFromOutputUnsupportedType)
{
  c10::IValue non_tensor{42};
  EXPECT_THROW(
      extract_tensors_from_output(non_tensor),
      UnsupportedModelOutputTypeException);
}

TEST(StarPUSetupExtractTensorsTest, SingleTensor)
{
  at::Tensor tensor = torch::arange(6).view({2, 3});
  c10::IValue iv{tensor};
  auto outputs = extract_tensors_from_output(iv);
  ASSERT_EQ(outputs.size(), 1);
  EXPECT_EQ(outputs[0].sizes().vec(), tensor.sizes().vec());
  EXPECT_TRUE(outputs[0].equal(tensor));
}

TEST(StarPUSetupExtractTensorsTest, TupleOfTensors)
{
  at::Tensor t1 = torch::ones({2, 2});
  at::Tensor t2 = torch::zeros({1, 3});
  auto tuple = c10::ivalue::Tuple::create({t1, t2});
  c10::IValue iv{tuple};
  auto outputs = extract_tensors_from_output(iv);
  ASSERT_EQ(outputs.size(), 2);
  EXPECT_TRUE(outputs[0].equal(t1));
  EXPECT_TRUE(outputs[1].equal(t2));
}

TEST(StarPUSetupExtractTensorsTest, TensorList)
{
  at::Tensor t1 = torch::rand({2});
  at::Tensor t2 = torch::rand({3});
  c10::List<at::Tensor> list;
  list.push_back(t1);
  list.push_back(t2);
  c10::IValue iv{list};
  auto outputs = extract_tensors_from_output(iv);
  ASSERT_EQ(outputs.size(), 2);
  EXPECT_TRUE(outputs[0].equal(t1));
  EXPECT_TRUE(outputs[1].equal(t2));
}
