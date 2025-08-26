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

TEST(RuntimeConfig, ComputeMaxMessageBytesThrowsOnPerSampleBytesOverflow)
{
  starpu_server::TensorConfig t1;
  t1.dims = {static_cast<int64_t>(std::numeric_limits<size_t>::max() / 4)};
  t1.type = at::kFloat;

  starpu_server::TensorConfig t2;
  t2.dims = {2};
  t2.type = at::kFloat;

  EXPECT_THROW(
      starpu_server::compute_max_message_bytes(
          1, std::vector<starpu_server::TensorConfig>{t1},
          std::vector<starpu_server::TensorConfig>{t2}),
      starpu_server::MessageSizeOverflowException);
}
