#include <gtest/gtest.h>

#include "core/tensor_builder.hpp"
#include "utils/exceptions.hpp"

namespace starpu_server {

class TensorBuilderTestAccessor {
 public:
  static at::Tensor from_raw_ptr(
      uintptr_t ptr, at::ScalarType type, const std::vector<int64_t>& shape,
      torch::Device device)
  {
    return TensorBuilder::from_raw_ptr(ptr, type, shape, device);
  }
};

}  // namespace starpu_server

using namespace starpu_server;

TEST(TensorBuilder, FromRawPtrFloat)
{
  float buffer[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int64_t> shape = {2, 2};
  torch::Device device(torch::kCPU);
  auto tensor = TensorBuilderTestAccessor::from_raw_ptr(
      reinterpret_cast<uintptr_t>(buffer), at::kFloat, shape, device);

  EXPECT_EQ(tensor.sizes(), (torch::IntArrayRef{2, 2}));
  EXPECT_EQ(tensor.dtype(), torch::kFloat);
  EXPECT_EQ(tensor.device(), device);
  EXPECT_EQ(tensor.data_ptr<float>(), buffer);
}

TEST(TensorBuilder, FromRawPtrInt)
{
  int32_t buffer[3] = {1, 2, 3};
  std::vector<int64_t> shape = {3};
  torch::Device device(torch::kCPU);
  auto tensor = TensorBuilderTestAccessor::from_raw_ptr(
      reinterpret_cast<uintptr_t>(buffer), at::kInt, shape, device);

  EXPECT_EQ(tensor.sizes(), (torch::IntArrayRef{3}));
  EXPECT_EQ(tensor.dtype(), torch::kInt);
  EXPECT_EQ(tensor.device(), device);
  EXPECT_EQ(tensor.data_ptr<int32_t>(), buffer);
}

TEST(TensorBuilder, CopyOutputToBufferFloat)
{
  auto tensor = torch::tensor(
      {1.0f, 2.0f, 3.0f}, torch::TensorOptions().dtype(at::kFloat));
  float buffer[3] = {0.0f, 0.0f, 0.0f};
  TensorBuilder::copy_output_to_buffer(tensor, buffer, 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(buffer[i], tensor[i].item<float>());
  }
}

TEST(TensorBuilder, CopyOutputToBufferInt)
{
  auto tensor =
      torch::tensor({1, 2, 3, 4}, torch::TensorOptions().dtype(at::kInt));
  int32_t buffer[4] = {0, 0, 0, 0};
  TensorBuilder::copy_output_to_buffer(tensor, buffer, 4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(buffer[i], tensor[i].item<int32_t>());
  }
}

TEST(TensorBuilder, CopyOutputToBufferSizeMismatch)
{
  auto tensor = torch::tensor({1, 2}, torch::TensorOptions().dtype(at::kInt));
  int32_t buffer[2];
  EXPECT_THROW(
      TensorBuilder::copy_output_to_buffer(tensor, buffer, 3),
      InferenceExecutionException);
}
