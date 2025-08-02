#include <gtest/gtest.h>

#include "core/starpu_setup.hpp"

using namespace starpu_server;

TEST(StarPUSetupTest, GetCudaWorkersByDeviceNegativeId)
{
  EXPECT_THROW(
      StarPUSetup::get_cuda_workers_by_device({-1}), std::invalid_argument);
}