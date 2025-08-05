#include <gtest/gtest.h>

#include <complex>
#include <vector>

#define private public
#include "core/tensor_builder.hpp"
#undef private

#include "utils/exceptions.hpp"

namespace {
float float_buffer[4] = {1.0f, 2.0f, 3.0f, 4.0f};
int32_t int_buffer[3] = {1, 2, 3};
double double_buffer[2] = {1.0, 2.0};
int64_t long_buffer[2] = {1, 2};
int16_t short_buffer[2] = {1, 2};
int8_t char_buffer[2] = {1, 2};
uint8_t byte_buffer[2] = {1, 2};
bool bool_buffer[2] = {true, false};

struct FromRawPtrParam {
  at::ScalarType dtype;
  void* buffer;
  std::vector<int64_t> shape;
};
}  // namespace

class TensorBuilderFromRawPtr
    : public ::testing::TestWithParam<FromRawPtrParam> {
 protected:
  torch::Device device{torch::kCPU};
  torch::Tensor build_tensor() const
  {
    const auto& param = GetParam();
    return starpu_server::TensorBuilder::from_raw_ptr(
        reinterpret_cast<uintptr_t>(param.buffer), param.dtype, param.shape,
        device);
  }
};

TEST_P(TensorBuilderFromRawPtr, ConstructsTensor)
{
  const auto& param = GetParam();
  auto tensor = build_tensor();
  EXPECT_EQ(tensor.sizes(), torch::IntArrayRef(param.shape));
  EXPECT_EQ(tensor.dtype(), param.dtype);
  EXPECT_EQ(tensor.device(), device);
  switch (param.dtype) {
    case at::kFloat:
      EXPECT_EQ(tensor.data_ptr<float>(), param.buffer);
      break;
    case at::kInt:
      EXPECT_EQ(tensor.data_ptr<int32_t>(), param.buffer);
      break;
    case at::kDouble:
      EXPECT_EQ(tensor.data_ptr<double>(), param.buffer);
      break;
    case at::kLong:
      EXPECT_EQ(tensor.data_ptr<int64_t>(), param.buffer);
      break;
    case at::kShort:
      EXPECT_EQ(tensor.data_ptr<int16_t>(), param.buffer);
      break;
    case at::kChar:
      EXPECT_EQ(tensor.data_ptr<int8_t>(), param.buffer);
      break;
    case at::kByte:
      EXPECT_EQ(tensor.data_ptr<uint8_t>(), param.buffer);
      break;
    case at::kBool:
      EXPECT_EQ(tensor.data_ptr<bool>(), param.buffer);
      break;
    default:
      FAIL() << "Unsupported dtype";
  }
}

INSTANTIATE_TEST_SUITE_P(
    TensorBuilder, TensorBuilderFromRawPtr,
    ::testing::Values(
        FromRawPtrParam{at::kFloat, float_buffer, {2, 2}},
        FromRawPtrParam{at::kInt, int_buffer, {3}},
        FromRawPtrParam{at::kDouble, double_buffer, {2}},
        FromRawPtrParam{at::kLong, long_buffer, {2}},
        FromRawPtrParam{at::kShort, short_buffer, {2}},
        FromRawPtrParam{at::kChar, char_buffer, {2}},
        FromRawPtrParam{at::kByte, byte_buffer, {2}},
        FromRawPtrParam{at::kBool, bool_buffer, {2}}));

TEST(TensorBuilder, FromRawPtrUnsupportedHalf)
{
  uint16_t buffer[1] = {0};
  std::vector<int64_t> shape = {1};
  torch::Device device(torch::kCPU);
  EXPECT_THROW(
      starpu_server::TensorBuilder::from_raw_ptr(
          reinterpret_cast<uintptr_t>(buffer), at::kHalf, shape, device),
      starpu_server::InferenceExecutionException);
}

TEST(TensorBuilder, FromRawPtrUnsupportedComplex)
{
  std::complex<float> buffer[1] = {std::complex<float>(0.0f, 0.0f)};
  std::vector<int64_t> shape = {1};
  torch::Device device(torch::kCPU);
  EXPECT_THROW(
      starpu_server::TensorBuilder::from_raw_ptr(
          reinterpret_cast<uintptr_t>(buffer), at::kComplexFloat, shape,
          device),
      starpu_server::InferenceExecutionException);
}

TEST(TensorBuilder, CopyOutputToBufferFloat)
{
  auto tensor = torch::tensor(
      {1.0f, 2.0f, 3.0f}, torch::TensorOptions().dtype(at::kFloat));
  float buffer[3] = {0.0f, 0.0f, 0.0f};
  starpu_server::TensorBuilder::copy_output_to_buffer(tensor, buffer, 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(buffer[i], tensor[i].item<float>());
  }
}

TEST(TensorBuilder, CopyOutputToBufferInt)
{
  auto tensor =
      torch::tensor({1, 2, 3, 4}, torch::TensorOptions().dtype(at::kInt));
  int32_t buffer[4] = {0, 0, 0, 0};
  starpu_server::TensorBuilder::copy_output_to_buffer(tensor, buffer, 4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(buffer[i], tensor[i].item<int32_t>());
  }
}

TEST(TensorBuilder, CopyOutputToBufferDouble)
{
  auto tensor =
      torch::tensor({1.0, 2.0, 3.0}, torch::TensorOptions().dtype(at::kDouble));
  double buffer[3] = {0.0, 0.0, 0.0};
  starpu_server::TensorBuilder::copy_output_to_buffer(tensor, buffer, 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_DOUBLE_EQ(buffer[i], tensor[i].item<double>());
  }
}

TEST(TensorBuilder, CopyOutputToBufferLong)
{
  auto tensor =
      torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kLong));
  int64_t buffer[3] = {0, 0, 0};
  starpu_server::TensorBuilder::copy_output_to_buffer(tensor, buffer, 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(buffer[i], tensor[i].item<int64_t>());
  }
}

TEST(TensorBuilder, CopyOutputToBufferShort)
{
  auto tensor =
      torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kShort));
  int16_t buffer[3] = {0, 0, 0};
  starpu_server::TensorBuilder::copy_output_to_buffer(tensor, buffer, 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(buffer[i], tensor[i].item<int16_t>());
  }
}

TEST(TensorBuilder, CopyOutputToBufferChar)
{
  auto tensor =
      torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kChar));
  int8_t buffer[3] = {0, 0, 0};
  starpu_server::TensorBuilder::copy_output_to_buffer(tensor, buffer, 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(buffer[i], tensor[i].item<int8_t>());
  }
}

TEST(TensorBuilder, CopyOutputToBufferByte)
{
  auto tensor =
      torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kByte));
  uint8_t buffer[3] = {0, 0, 0};
  starpu_server::TensorBuilder::copy_output_to_buffer(tensor, buffer, 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(buffer[i], tensor[i].item<uint8_t>());
  }
}

TEST(TensorBuilder, CopyOutputToBufferBool)
{
  auto tensor = torch::tensor(
      {true, false, true}, torch::TensorOptions().dtype(at::kBool));
  bool buffer[3] = {false, false, false};
  starpu_server::TensorBuilder::copy_output_to_buffer(tensor, buffer, 3);
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(buffer[i], tensor[i].item<bool>());
  }
}

TEST(TensorBuilder, CopyOutputToBufferSizeMismatch)
{
  auto tensor = torch::tensor({1, 2}, torch::TensorOptions().dtype(at::kInt));
  int32_t buffer[2];
  EXPECT_THROW(
      starpu_server::TensorBuilder::copy_output_to_buffer(tensor, buffer, 3),
      starpu_server::InferenceExecutionException);
}

TEST(TensorBuilder, CopyOutputToBufferExpectedNumelTooSmall)
{
  auto tensor =
      torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kInt));
  int32_t buffer[3];
  EXPECT_THROW(
      starpu_server::TensorBuilder::copy_output_to_buffer(tensor, buffer, 2),
      starpu_server::InferenceExecutionException);
}

TEST(TensorBuilder, CopyOutputToBufferUnsupportedType)
{
  auto tensor =
      torch::zeros({2}, torch::TensorOptions().dtype(at::kComplexFloat));
  float buffer[2] = {0.0f, 0.0f};
  EXPECT_THROW(
      starpu_server::TensorBuilder::copy_output_to_buffer(tensor, buffer, 2),
      starpu_server::InferenceExecutionException);
}

TEST(TensorBuilder, FromStarpuBuffersTooManyInputs)
{
  starpu_server::InferenceParams params;
  params.num_inputs = starpu_server::InferLimits::MaxInputs + 1;
  std::vector<void*> buffers(params.num_inputs, nullptr);
  torch::Device device(torch::kCPU);
  EXPECT_THROW(
      starpu_server::TensorBuilder::from_starpu_buffers(
          &params, buffers, device),
      starpu_server::InferenceExecutionException);
}

TEST(TensorBuilder, FromStarpuBuffersSuccess)
{
  float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  starpu_variable_interface fake_var;
  fake_var.ptr = reinterpret_cast<uintptr_t>(data);
  std::vector<void*> buffers = {&fake_var};
  starpu_server::InferenceParams params;
  params.num_inputs = 1;
  params.layout.input_types = {at::kFloat};
  params.layout.num_dims = {2};
  params.layout.dims = {{2, 2}};
  torch::Device device(torch::kCPU);
  auto tensors = starpu_server::TensorBuilder::from_starpu_buffers(
      &params, buffers, device);
  ASSERT_EQ(tensors.size(), 1);
  EXPECT_EQ(tensors[0].sizes(), (torch::IntArrayRef{2, 2}));
  EXPECT_EQ(tensors[0].dtype(), torch::kFloat);
  EXPECT_EQ(tensors[0].device(), device);
  EXPECT_EQ(tensors[0].data_ptr<float>(), data);
}

TEST(TensorBuilder, FromRawPtrUnsupportedQuantized)
{
  uint8_t dummy = 0;
  std::vector<int64_t> shape = {1};
  torch::Device device(torch::kCPU);
  EXPECT_THROW(
      starpu_server::TensorBuilder::from_raw_ptr(
          reinterpret_cast<uintptr_t>(&dummy), at::kQInt8, shape, device),
      starpu_server::InferenceExecutionException);
}
