#include <gtest/gtest.h>

#include <algorithm>
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

float copy_float_buffer[3];
int32_t copy_int_buffer[3];
double copy_double_buffer[3];
int64_t copy_long_buffer[3];
int16_t copy_short_buffer[3];
int8_t copy_char_buffer[3];
uint8_t copy_byte_buffer[3];
bool copy_bool_buffer[3];

struct FromRawPtrParam {
  at::ScalarType dtype;
  void* buffer;
  std::vector<int64_t> shape;
};

struct CopyOutputParam {
  at::ScalarType dtype;
  void* buffer;
  torch::Tensor tensor;
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

class TensorBuilderCopyOutputToBuffer
    : public ::testing::TestWithParam<CopyOutputParam> {};

TEST_P(TensorBuilderCopyOutputToBuffer, CopiesCorrectly)
{
  const auto& param = GetParam();
  const int numel = param.tensor.numel();
  switch (param.dtype) {
    case at::kFloat: {
      auto* buffer = static_cast<float*>(param.buffer);
      std::fill_n(buffer, numel, 0.0f);
      starpu_server::TensorBuilder::copy_output_to_buffer(
          param.tensor, buffer, numel);
      for (int i = 0; i < numel; ++i) {
        EXPECT_FLOAT_EQ(buffer[i], param.tensor[i].item<float>());
      }
      break;
    }
    case at::kInt: {
      auto* buffer = static_cast<int32_t*>(param.buffer);
      std::fill_n(buffer, numel, 0);
      starpu_server::TensorBuilder::copy_output_to_buffer(
          param.tensor, buffer, numel);
      for (int i = 0; i < numel; ++i) {
        EXPECT_EQ(buffer[i], param.tensor[i].item<int32_t>());
      }
      break;
    }
    case at::kDouble: {
      auto* buffer = static_cast<double*>(param.buffer);
      std::fill_n(buffer, numel, 0.0);
      starpu_server::TensorBuilder::copy_output_to_buffer(
          param.tensor, buffer, numel);
      for (int i = 0; i < numel; ++i) {
        EXPECT_DOUBLE_EQ(buffer[i], param.tensor[i].item<double>());
      }
      break;
    }
    case at::kLong: {
      auto* buffer = static_cast<int64_t*>(param.buffer);
      std::fill_n(buffer, numel, 0);
      starpu_server::TensorBuilder::copy_output_to_buffer(
          param.tensor, buffer, numel);
      for (int i = 0; i < numel; ++i) {
        EXPECT_EQ(buffer[i], param.tensor[i].item<int64_t>());
      }
      break;
    }
    case at::kShort: {
      auto* buffer = static_cast<int16_t*>(param.buffer);
      std::fill_n(buffer, numel, 0);
      starpu_server::TensorBuilder::copy_output_to_buffer(
          param.tensor, buffer, numel);
      for (int i = 0; i < numel; ++i) {
        EXPECT_EQ(buffer[i], param.tensor[i].item<int16_t>());
      }
      break;
    }
    case at::kChar: {
      auto* buffer = static_cast<int8_t*>(param.buffer);
      std::fill_n(buffer, numel, 0);
      starpu_server::TensorBuilder::copy_output_to_buffer(
          param.tensor, buffer, numel);
      for (int i = 0; i < numel; ++i) {
        EXPECT_EQ(buffer[i], param.tensor[i].item<int8_t>());
      }
      break;
    }
    case at::kByte: {
      auto* buffer = static_cast<uint8_t*>(param.buffer);
      std::fill_n(buffer, numel, 0);
      starpu_server::TensorBuilder::copy_output_to_buffer(
          param.tensor, buffer, numel);
      for (int i = 0; i < numel; ++i) {
        EXPECT_EQ(buffer[i], param.tensor[i].item<uint8_t>());
      }
      break;
    }
    case at::kBool: {
      auto* buffer = static_cast<bool*>(param.buffer);
      std::fill_n(buffer, numel, false);
      starpu_server::TensorBuilder::copy_output_to_buffer(
          param.tensor, buffer, numel);
      for (int i = 0; i < numel; ++i) {
        EXPECT_EQ(buffer[i], param.tensor[i].item<bool>());
      }
      break;
    }
    default:
      FAIL() << "Unsupported dtype";
  }
}

INSTANTIATE_TEST_SUITE_P(
    TensorBuilder, TensorBuilderCopyOutputToBuffer,
    ::testing::Values(
        CopyOutputParam{
            at::kFloat, copy_float_buffer,
            torch::tensor(
                {1.0f, 2.0f, 3.0f}, torch::TensorOptions().dtype(at::kFloat))},
        CopyOutputParam{
            at::kInt, copy_int_buffer,
            torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kInt))},
        CopyOutputParam{
            at::kDouble, copy_double_buffer,
            torch::tensor(
                {1.0, 2.0, 3.0}, torch::TensorOptions().dtype(at::kDouble))},
        CopyOutputParam{
            at::kLong, copy_long_buffer,
            torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kLong))},
        CopyOutputParam{
            at::kShort, copy_short_buffer,
            torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kShort))},
        CopyOutputParam{
            at::kChar, copy_char_buffer,
            torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kChar))},
        CopyOutputParam{
            at::kByte, copy_byte_buffer,
            torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kByte))},
        CopyOutputParam{
            at::kBool, copy_bool_buffer,
            torch::tensor(
                {true, false, true},
                torch::TensorOptions().dtype(at::kBool))}));

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
