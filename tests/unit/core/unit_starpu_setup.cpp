#include <gtest/gtest.h>

#include "core/starpu_setup.hpp"
#include "utils/runtime_config.hpp"

TEST(StarPUSetup_Unit, DuplicateDeviceIdsThrows)
{
  starpu_server::RuntimeConfig opts;
  opts.use_cuda = true;
  opts.device_ids = {0, 0};
  EXPECT_THROW(
      { starpu_server::StarPUSetup setup(opts); }, std::invalid_argument);
}
