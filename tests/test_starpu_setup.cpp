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