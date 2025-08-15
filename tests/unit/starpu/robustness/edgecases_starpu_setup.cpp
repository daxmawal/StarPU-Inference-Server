#include <gtest/gtest.h>
#include <torch/script.h>

#include <stdexcept>

#include "core/starpu_setup.hpp"
#include "utils/exceptions.hpp"

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
