#include <gtest/gtest.h>
#include <torch/torch.h>

#include <vector>

#include "core/tensor_builder.hpp"

TEST(RunInference_Unit, CopyOutputToBufferCopiesData)
{
  auto t = torch::tensor({1.0f, 2.0f, 3.5f, -4.0f, 0.25f}, torch::kFloat32);
  std::vector<float> dst(5, 0.0f);
  starpu_server::TensorBuilder::copy_output_to_buffer(t, dst.data(), t.numel());
  ASSERT_EQ(dst.size(), 5U);
  EXPECT_FLOAT_EQ(dst[0], 1.0f);
  EXPECT_FLOAT_EQ(dst[1], 2.0f);
  EXPECT_FLOAT_EQ(dst[2], 3.5f);
  EXPECT_FLOAT_EQ(dst[3], -4.0f);
  EXPECT_FLOAT_EQ(dst[4], 0.25f);
}
