#include <gtest/gtest.h>

#include "utils/device_type.hpp"

using namespace starpu_server;

TEST(DeviceTypeTest, ToString)
{
  EXPECT_STREQ(to_string(DeviceType::CPU), "CPU");
  EXPECT_STREQ(to_string(DeviceType::CUDA), "CUDA");
  EXPECT_STREQ(to_string(DeviceType::Unknown), "Unknown");
  EXPECT_STREQ(to_string(static_cast<DeviceType>(42)), "InvalidDeviceType");
}
