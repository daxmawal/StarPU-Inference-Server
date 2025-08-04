#include <gtest/gtest.h>

#include "utils/device_type.hpp"

TEST(DeviceTypeTest, ToString)
{
  EXPECT_STREQ(starpu_server::to_string(starpu_server::DeviceType::CPU), "CPU");
  EXPECT_STREQ(
      starpu_server::to_string(starpu_server::DeviceType::CUDA), "CUDA");
  EXPECT_STREQ(
      starpu_server::to_string(starpu_server::DeviceType::Unknown), "Unknown");
  EXPECT_STREQ(
      starpu_server::to_string(static_cast<starpu_server::DeviceType>(42)),
      "InvalidDeviceType");
}
