#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstddef>
#include <span>
#include <vector>

#define private public
#include "core/tensor_builder.hpp"
#undef private

#include "test_constants.hpp"
#include "utils/exceptions.hpp"

namespace {
constexpr std::array<int64_t, 2> kShape2x2{{2, 2}};
constexpr std::array<int64_t, 1> kShape1{{1}};

using starpu_server::test_constants::kD1;
using starpu_server::test_constants::kD2;
using starpu_server::test_constants::kD3;
using starpu_server::test_constants::kF1;
using starpu_server::test_constants::kF2;
using starpu_server::test_constants::kF3;
using starpu_server::test_constants::kF4;

const std::array<float, 4> float_buffer{kF1, kF2, kF3, kF4};
const std::array<int32_t, 3> int_buffer{1, 2, 3};
const std::array<double, 2> double_buffer{kD1, kD2};
const std::array<int64_t, 2> long_buffer{1, 2};
const std::array<int16_t, 2> short_buffer{1, 2};
const std::array<int8_t, 2> char_buffer{1, 2};
const std::array<uint8_t, 2> byte_buffer{1, 2};
const std::array<bool, 2> bool_buffer{true, false};
struct CopyOutputParam;

inline void
expect_data_ptr_eq(torch::Tensor& tensor, const void* expected)
{
  EXPECT_EQ(static_cast<const void*>(tensor.data_ptr()), expected);
}

template <typename T>
inline void
expect_equal_value(T lhs, T rhs)
{
  EXPECT_EQ(lhs, rhs);
}

template <>
inline void
expect_equal_value<float>(float lhs, float rhs)
{
  EXPECT_FLOAT_EQ(lhs, rhs);
}

template <>
inline void
expect_equal_value<double>(double lhs, double rhs)
{
  EXPECT_DOUBLE_EQ(lhs, rhs);
}

template <typename T>
inline auto
zero_value() -> T
{
  return T{};
}

template <>
inline auto
zero_value<bool>() -> bool
{
  return false;
}

template <typename T>
inline void
verify_copy_typed(
    void* buffer, const torch::Tensor& tensor, at::ScalarType dtype)
{
  const auto numel = static_cast<size_t>(tensor.numel());
  auto* buff = static_cast<T*>(buffer);
  std::fill_n(buff, numel, zero_value<T>());
  auto buffer_bytes = std::as_writable_bytes(std::span<T>(buff, numel));
  starpu_server::TensorBuilder::copy_output_to_buffer(
      tensor, buffer_bytes, static_cast<int64_t>(numel), dtype);
  std::span<T> sbuff{buff, numel};
  for (size_t i = 0; i < sbuff.size(); ++i) {
    expect_equal_value<T>(sbuff[i], tensor[static_cast<int64_t>(i)].item<T>());
  }
}

struct FromRawPtrParam {
  at::ScalarType dtype;
  const void* buffer;
  std::vector<int64_t> shape;
};
struct CopyOutputParam {
  at::ScalarType dtype;
  void* buffer;
  torch::Tensor tensor;
};
}  // namespace

inline void
verify_copy_by_dtype(const CopyOutputParam& param)
{
  switch (param.dtype) {
    case at::kFloat:
      verify_copy_typed<float>(param.buffer, param.tensor, param.dtype);
      break;
    case at::kInt:
      verify_copy_typed<int32_t>(param.buffer, param.tensor, param.dtype);
      break;
    case at::kDouble:
      verify_copy_typed<double>(param.buffer, param.tensor, param.dtype);
      break;
    case at::kLong:
      verify_copy_typed<int64_t>(param.buffer, param.tensor, param.dtype);
      break;
    case at::kShort:
      verify_copy_typed<int16_t>(param.buffer, param.tensor, param.dtype);
      break;
    case at::kChar:
      verify_copy_typed<int8_t>(param.buffer, param.tensor, param.dtype);
      break;
    case at::kByte:
      verify_copy_typed<uint8_t>(param.buffer, param.tensor, param.dtype);
      break;
    case at::kBool:
      verify_copy_typed<bool>(param.buffer, param.tensor, param.dtype);
      break;
    default:
      FAIL() << "Unsupported dtype";
  }
}

class TensorBuilderFromRawPtr_Unit
    : public ::testing::TestWithParam<FromRawPtrParam> {
 protected:
  [[nodiscard]] auto build() const -> torch::Tensor
  {
    const auto& param = GetParam();
    return starpu_server::TensorBuilder::from_raw_ptr(
        std::bit_cast<uintptr_t>(param.buffer), param.dtype, param.shape,
        device_);
  }

 private:
  torch::Device device_{torch::kCPU};
};

TEST_P(TensorBuilderFromRawPtr_Unit, ConstructsTensorWithCorrectViewAndPtr)
{
  const auto& param = GetParam();
  auto tensor = build();
  EXPECT_EQ(tensor.sizes(), torch::IntArrayRef(param.shape));
  EXPECT_EQ(tensor.dtype(), param.dtype);
  EXPECT_EQ(tensor.device(), torch::Device(torch::kCPU));
  expect_data_ptr_eq(tensor, param.buffer);
}

INSTANTIATE_TEST_SUITE_P(
    TensorBuilder, TensorBuilderFromRawPtr_Unit,
    ::testing::Values(
        FromRawPtrParam{
            at::kFloat,
            static_cast<const void*>(float_buffer.data()),
            {kShape2x2[0], kShape2x2[1]}},
        FromRawPtrParam{
            at::kInt, static_cast<const void*>(int_buffer.data()), {3}},
        FromRawPtrParam{
            at::kDouble, static_cast<const void*>(double_buffer.data()), {2}},
        FromRawPtrParam{
            at::kLong, static_cast<const void*>(long_buffer.data()), {2}},
        FromRawPtrParam{
            at::kShort, static_cast<const void*>(short_buffer.data()), {2}},
        FromRawPtrParam{
            at::kChar, static_cast<const void*>(char_buffer.data()), {2}},
        FromRawPtrParam{
            at::kByte, static_cast<const void*>(byte_buffer.data()), {2}},
        FromRawPtrParam{
            at::kBool, static_cast<const void*>(bool_buffer.data()), {2}}));

class TensorBuilderCopyBuffer_Unit
    : public ::testing::TestWithParam<CopyOutputParam> {};

TEST_P(TensorBuilderCopyBuffer_Unit, CopiesToRawBuffer)
{
  const auto& param = GetParam();
  verify_copy_by_dtype(param);
}

namespace {
inline auto
copy_float_buffer_ref() -> std::array<float, 3>&
{
  static std::array<float, 3> buf{};
  return buf;
}
inline auto
copy_int_buffer_ref() -> std::array<int32_t, 3>&
{
  static std::array<int32_t, 3> buf{};
  return buf;
}
inline auto
copy_double_buffer_ref() -> std::array<double, 3>&
{
  static std::array<double, 3> buf{};
  return buf;
}
inline auto
copy_long_buffer_ref() -> std::array<int64_t, 3>&
{
  static std::array<int64_t, 3> buf{};
  return buf;
}
inline auto
copy_short_buffer_ref() -> std::array<int16_t, 3>&
{
  static std::array<int16_t, 3> buf{};
  return buf;
}
inline auto
copy_char_buffer_ref() -> std::array<int8_t, 3>&
{
  static std::array<int8_t, 3> buf{};
  return buf;
}
inline auto
copy_byte_buffer_ref() -> std::array<uint8_t, 3>&
{
  static std::array<uint8_t, 3> buf{};
  return buf;
}
inline auto
copy_bool_buffer_ref() -> std::array<bool, 3>&
{
  static std::array<bool, 3> buf{};
  return buf;
}
}  // namespace

INSTANTIATE_TEST_SUITE_P(
    TensorBuilder, TensorBuilderCopyBuffer_Unit,
    ::testing::Values(
        CopyOutputParam{
            at::kFloat, copy_float_buffer_ref().data(),
            torch::tensor(
                {kF1, kF2, kF3}, torch::TensorOptions().dtype(at::kFloat))},
        CopyOutputParam{
            at::kInt, copy_int_buffer_ref().data(),
            torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kInt))},
        CopyOutputParam{
            at::kDouble, copy_double_buffer_ref().data(),
            torch::tensor(
                {kD1, kD2, kD3}, torch::TensorOptions().dtype(at::kDouble))},
        CopyOutputParam{
            at::kLong, copy_long_buffer_ref().data(),
            torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kLong))},
        CopyOutputParam{
            at::kShort, copy_short_buffer_ref().data(),
            torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kShort))},
        CopyOutputParam{
            at::kChar, copy_char_buffer_ref().data(),
            torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kChar))},
        CopyOutputParam{
            at::kByte, copy_byte_buffer_ref().data(),
            torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kByte))},
        CopyOutputParam{
            at::kBool, copy_bool_buffer_ref().data(),
            torch::tensor(
                {true, false, true},
                torch::TensorOptions().dtype(at::kBool))}));

TEST(TensorBuilder_Unit, FromStarpuBuffersSuccess)
{
  std::array<float, 4> data{kF1, kF2, kF3, kF4};
  starpu_variable_interface var{};
  var.ptr = std::bit_cast<uintptr_t>(data.data());
  std::vector<void*> buffers = {&var};

  starpu_server::InferenceParams params;
  params.num_inputs = 1;
  params.layout.input_types = {at::kFloat};
  params.layout.num_dims = {2};
  params.layout.dims = {{2, 2}};
  params.limits.max_inputs = starpu_server::InferLimits::MaxInputs;
  params.limits.max_dims = starpu_server::InferLimits::MaxDims;
  params.limits.max_models_gpu = starpu_server::InferLimits::MaxModelsGPU;

  const auto device = torch::Device(torch::kCPU);
  auto tensors = starpu_server::TensorBuilder::from_starpu_buffers(
      &params, buffers, device);
  ASSERT_EQ(tensors.size(), 1U);
  EXPECT_EQ(tensors[0].sizes(), (torch::IntArrayRef{2, 2}));
  EXPECT_EQ(tensors[0].dtype(), torch::kFloat);
  EXPECT_EQ(tensors[0].device(), device);
  EXPECT_EQ(tensors[0].data_ptr<float>(), data.data());
}
