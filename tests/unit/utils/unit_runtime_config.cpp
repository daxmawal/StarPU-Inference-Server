#include <gtest/gtest.h>

#include <limits>

#include "utils/runtime_config.hpp"

TEST(RuntimeConfig, ComputeMaxMessageBytesThrowsOnNumelOverflow)
{
  starpu_server::TensorConfig t;
  t.dims = {std::numeric_limits<int64_t>::max(), 3};
  t.type = at::kFloat;

  EXPECT_THROW(
      starpu_server::compute_max_message_bytes(
          1, std::vector<starpu_server::TensorConfig>{t}, {}),
      starpu_server::MessageSizeOverflowException);
}
