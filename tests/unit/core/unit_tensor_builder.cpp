#include <gtest/gtest.h>

#include <array>
#include <vector>

#define private public
#include "core/tensor_builder.hpp"
#undef private

#include "utils/exceptions.hpp"

namespace {
constexpr std::array<int64_t, 2> kShape2x2{{2, 2}};
constexpr std::array<int64_t, 1> kShape1{{1}};

std::array<float, 4> float_buffer{1.F, 2.F, 3.F, 4.F};
std::array<int32_t, 3> int_buffer{1, 2, 3};
std::array<double, 2> double_buffer{1.0, 2.0};
std::array<int64_t, 2> long_buffer{1, 2};
std::array<int16_t, 2> short_buffer{1, 2};
std::array<int8_t, 2> char_buffer{1, 2};
std::array<uint8_t, 2> byte_buffer{1, 2};
std::array<bool, 2> bool_buffer{true, false};

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

class TensorBuilderFromRawPtr_Unit
    : public ::testing::TestWithParam<FromRawPtrParam> {
 protected:
  torch::Device device{torch::kCPU};
  [[nodiscard]] auto build() const -> torch::Tensor
  {
    const auto& param = GetParam();
    return starpu_server::TensorBuilder::from_raw_ptr(
        reinterpret_cast<uintptr_t>(param.buffer), param.dtype, param.shape,
        device);
  }
};

TEST_P(TensorBuilderFromRawPtr_Unit, ConstructsTensorWithCorrectViewAndPtr)
{
  const auto& param = GetParam();
  auto tensor = build();
  EXPECT_EQ(tensor.sizes(), torch::IntArrayRef(param.shape));
  EXPECT_EQ(tensor.dtype(), param.dtype);
  EXPECT_EQ(tensor.device(), torch::Device(torch::kCPU));
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
    TensorBuilder, TensorBuilderFromRawPtr_Unit,
    ::testing::Values(
        FromRawPtrParam{
            at::kFloat, float_buffer.data(), {kShape2x2[0], kShape2x2[1]}},
        FromRawPtrParam{at::kInt, int_buffer.data(), {3}},
        FromRawPtrParam{at::kDouble, double_buffer.data(), {2}},
        FromRawPtrParam{at::kLong, long_buffer.data(), {2}},
        FromRawPtrParam{at::kShort, short_buffer.data(), {2}},
        FromRawPtrParam{at::kChar, char_buffer.data(), {2}},
        FromRawPtrParam{at::kByte, byte_buffer.data(), {2}},
        FromRawPtrParam{at::kBool, bool_buffer.data(), {2}}));

class TensorBuilderCopyBuffer_Unit
    : public ::testing::TestWithParam<CopyOutputParam> {};

TEST_P(TensorBuilderCopyBuffer_Unit, CopiesToRawBuffer)
{
  const auto& param = GetParam();
  const auto numel = static_cast<size_t>(param.tensor.numel());
  switch (param.dtype) {
    case at::kFloat: {
      auto* buff = static_cast<float*>(param.buffer);
      std::fill_n(buff, numel, 0.F);
      starpu_server::TensorBuilder::copy_output_to_buffer(
          param.tensor, buff, numel);
      for (size_t i = 0; i < numel; ++i) {
        EXPECT_FLOAT_EQ(buff[i], param.tensor[i].item<float>());
      }
      break;
    }
    case at::kInt: {
      auto* buff = static_cast<int32_t*>(param.buffer);
      std::fill_n(buff, numel, 0);
      starpu_server::TensorBuilder::copy_output_to_buffer(
          param.tensor, buff, numel);
      for (size_t i = 0; i < numel; ++i) {
        EXPECT_EQ(buff[i], param.tensor[i].item<int32_t>());
      }
      break;
    }
    case at::kDouble: {
      auto* buff = static_cast<double*>(param.buffer);
      std::fill_n(buff, numel, 0.0);
      starpu_server::TensorBuilder::copy_output_to_buffer(
          param.tensor, buff, numel);
      for (size_t i = 0; i < numel; ++i) {
        EXPECT_DOUBLE_EQ(buff[i], param.tensor[i].item<double>());
      }
      break;
    }
    case at::kLong: {
      auto* buff = static_cast<int64_t*>(param.buffer);
      std::fill_n(buff, numel, 0);
      starpu_server::TensorBuilder::copy_output_to_buffer(
          param.tensor, buff, numel);
      for (size_t i = 0; i < numel; ++i) {
        EXPECT_EQ(buff[i], param.tensor[i].item<int64_t>());
      }
      break;
    }
    case at::kShort: {
      auto* buff = static_cast<int16_t*>(param.buffer);
      std::fill_n(buff, numel, 0);
      starpu_server::TensorBuilder::copy_output_to_buffer(
          param.tensor, buff, numel);
      for (size_t i = 0; i < numel; ++i) {
        EXPECT_EQ(buff[i], param.tensor[i].item<int16_t>());
      }
      break;
    }
    case at::kChar: {
      auto* buff = static_cast<int8_t*>(param.buffer);
      std::fill_n(buff, numel, 0);
      starpu_server::TensorBuilder::copy_output_to_buffer(
          param.tensor, buff, numel);
      for (size_t i = 0; i < numel; ++i) {
        EXPECT_EQ(buff[i], param.tensor[i].item<int8_t>());
      }
      break;
    }
    case at::kByte: {
      auto* buff = static_cast<uint8_t*>(param.buffer);
      std::fill_n(buff, numel, 0);
      starpu_server::TensorBuilder::copy_output_to_buffer(
          param.tensor, buff, numel);
      for (size_t i = 0; i < numel; ++i) {
        EXPECT_EQ(buff[i], param.tensor[i].item<uint8_t>());
      }
      break;
    }
    case at::kBool: {
      auto* buff = static_cast<bool*>(param.buffer);
      std::fill_n(buff, numel, false);
      starpu_server::TensorBuilder::copy_output_to_buffer(
          param.tensor, buff, numel);
      for (size_t i = 0; i < numel; ++i) {
        EXPECT_EQ(buff[i], param.tensor[i].item<bool>());
      }
      break;
    }
    default:
      FAIL() << "Unsupported dtype";
  }
}

namespace {
std::array<float, 3> copy_float_buffer{};
std::array<int32_t, 3> copy_int_buffer{};
std::array<double, 3> copy_double_buffer{};
std::array<int64_t, 3> copy_long_buffer{};
std::array<int16_t, 3> copy_short_buffer{};
std::array<int8_t, 3> copy_char_buffer{};
std::array<uint8_t, 3> copy_byte_buffer{};
std::array<bool, 3> copy_bool_buffer{};
}  // namespace

INSTANTIATE_TEST_SUITE_P(
    TensorBuilder, TensorBuilderCopyBuffer_Unit,
    ::testing::Values(
        CopyOutputParam{
            at::kFloat, copy_float_buffer.data(),
            torch::tensor(
                {1.F, 2.F, 3.F}, torch::TensorOptions().dtype(at::kFloat))},
        CopyOutputParam{
            at::kInt, copy_int_buffer.data(),
            torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kInt))},
        CopyOutputParam{
            at::kDouble, copy_double_buffer.data(),
            torch::tensor(
                {1., 2., 3.}, torch::TensorOptions().dtype(at::kDouble))},
        CopyOutputParam{
            at::kLong, copy_long_buffer.data(),
            torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kLong))},
        CopyOutputParam{
            at::kShort, copy_short_buffer.data(),
            torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kShort))},
        CopyOutputParam{
            at::kChar, copy_char_buffer.data(),
            torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kChar))},
        CopyOutputParam{
            at::kByte, copy_byte_buffer.data(),
            torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kByte))},
        CopyOutputParam{
            at::kBool, copy_bool_buffer.data(),
            torch::tensor(
                {true, false, true},
                torch::TensorOptions().dtype(at::kBool))}));

TEST(TensorBuilder_Unit, FromStarpuBuffersSuccess)
{
  std::array<float, 4> data{1.F, 2.F, 3.F, 4.F};
  starpu_variable_interface var;
  var.ptr = reinterpret_cast<uintptr_t>(data.data());
  std::vector<void*> buffers = {&var};

  starpu_server::InferenceParams params;
  params.num_inputs = 1;
  params.layout.input_types = {at::kFloat};
  params.layout.num_dims = {2};
  params.layout.dims = {{2, 2}};

  const auto device = torch::Device(torch::kCPU);
  auto tensors = starpu_server::TensorBuilder::from_starpu_buffers(
      &params, buffers, device);
  ASSERT_EQ(tensors.size(), 1U);
  EXPECT_EQ(tensors[0].sizes(), (torch::IntArrayRef{2, 2}));
  EXPECT_EQ(tensors[0].dtype(), torch::kFloat);
  EXPECT_EQ(tensors[0].device(), device);
  EXPECT_EQ(tensors[0].data_ptr<float>(), data.data());
}
