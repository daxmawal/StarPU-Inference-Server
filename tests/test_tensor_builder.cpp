#include <gtest/gtest.h>

#include <complex>

#define private public
#include "core/tensor_builder.hpp"
#undef private

#include "utils/exceptions.hpp"

using namespace starpu_server;

TEST(TensorBuilder, FromRawPtrFloat)
{
  float buffer[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int64_t> shape = {2, 2};
  torch::Device device(torch::kCPU);
  auto tensor = TensorBuilder::from_raw_ptr(
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
  auto tensor = TensorBuilder::from_raw_ptr(
      reinterpret_cast<uintptr_t>(buffer), at::kInt, shape, device);

  EXPECT_EQ(tensor.sizes(), (torch::IntArrayRef{3}));
  EXPECT_EQ(tensor.dtype(), torch::kInt);
  EXPECT_EQ(tensor.device(), device);
  EXPECT_EQ(tensor.data_ptr<int32_t>(), buffer);
}

TEST(TensorBuilder, FromRawPtrUnsupportedHalf)
{
  uint16_t buffer[1] = {0};
  std::vector<int64_t> shape = {1};
  torch::Device device(torch::kCPU);
  EXPECT_THROW(
      TensorBuilder::from_raw_ptr(
          reinterpret_cast<uintptr_t>(buffer), at::kHalf, shape, device),
      InferenceExecutionException);
}

TEST(TensorBuilder, FromRawPtrUnsupportedComplex)
{
  std::complex<float> buffer[1] = {std::complex<float>(0.0f, 0.0f)};
  std::vector<int64_t> shape = {1};
  torch::Device device(torch::kCPU);
  EXPECT_THROW(
      TensorBuilder::from_raw_ptr(
          reinterpret_cast<uintptr_t>(buffer), at::kComplexFloat, shape,
          device),
      InferenceExecutionException);
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

TEST(TensorBuilder, CopyOutputToBufferDouble)
{
  auto tensor =
      torch::tensor({1.0, 2.0, 3.0}, torch::TensorOptions().dtype(at::kDouble));
  double buffer[3] = {0.0, 0.0, 0.0};
  TensorBuilder::copy_output_to_buffer(tensor, buffer, 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(buffer[i], tensor[i].item<double>());
  }
}

TEST(TensorBuilder, CopyOutputToBufferLong)
{
  auto tensor =
      torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kLong));
  int64_t buffer[3] = {0, 0, 0};
  TensorBuilder::copy_output_to_buffer(tensor, buffer, 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(buffer[i], tensor[i].item<int64_t>());
  }
}

TEST(TensorBuilder, CopyOutputToBufferShort)
{
  auto tensor =
      torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kShort));
  int16_t buffer[3] = {0, 0, 0};
  TensorBuilder::copy_output_to_buffer(tensor, buffer, 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(buffer[i], tensor[i].item<int16_t>());
  }
}

TEST(TensorBuilder, CopyOutputToBufferChar)
{
  auto tensor =
      torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kChar));
  int8_t buffer[3] = {0, 0, 0};
  TensorBuilder::copy_output_to_buffer(tensor, buffer, 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(buffer[i], tensor[i].item<int8_t>());
  }
}

TEST(TensorBuilder, CopyOutputToBufferByte)
{
  auto tensor =
      torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kByte));
  uint8_t buffer[3] = {0, 0, 0};
  TensorBuilder::copy_output_to_buffer(tensor, buffer, 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(buffer[i], tensor[i].item<uint8_t>());
  }
}

TEST(TensorBuilder, CopyOutputToBufferBool)
{
  auto tensor = torch::tensor(
      {true, false, true}, torch::TensorOptions().dtype(at::kBool));
  bool buffer[3] = {false, false, false};
  TensorBuilder::copy_output_to_buffer(tensor, buffer, 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(buffer[i], tensor[i].item<bool>());
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

TEST(TensorBuilder, CopyOutputToBufferExpectedNumelTooSmall)
{
  auto tensor =
      torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kInt));
  int32_t buffer[3];
  EXPECT_THROW(
      TensorBuilder::copy_output_to_buffer(tensor, buffer, 2),
      InferenceExecutionException);
}

TEST(TensorBuilder, CopyOutputToBufferUnsupportedType)
{
  auto tensor =
      torch::zeros({2}, torch::TensorOptions().dtype(at::kComplexFloat));
  float buffer[2] = {0.0f, 0.0f};
  EXPECT_THROW(
      TensorBuilder::copy_output_to_buffer(tensor, buffer, 2),
      InferenceExecutionException);
}
