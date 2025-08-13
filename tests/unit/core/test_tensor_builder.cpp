#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <vector>

#define private public
#include "core/tensor_builder.hpp"
#undef private

#include "utils/exceptions.hpp"

namespace {
// ---- Constantes lisibles pour éviter les nombres magiques ----
constexpr int64_t kDim1 = 1;
constexpr int64_t kDim2 = 2;
constexpr int64_t kDim3 = 3;

constexpr std::array<int64_t, 2> kShape2x2{{kDim2, kDim2}};
constexpr std::array<int64_t, 1> kShape1{{kDim1}};
constexpr std::array<int64_t, 1> kShape3{{kDim3}};
constexpr std::array<int64_t, 1> kShape2{{kDim2}};

constexpr std::size_t kElems2 = 2;
constexpr std::size_t kElems3 = 3;

// Zéros typés pour les remplissages
constexpr float kZeroF = 0.0F;
constexpr double kZeroD = 0.0;
constexpr int32_t kZeroI = 0;
constexpr int64_t kZeroL = 0;
constexpr int16_t kZeroS = 0;
constexpr int8_t kZeroC = 0;
constexpr uint8_t kZeroB = 0;
constexpr bool kZeroBool = false;

// ---- Buffers d'entrée (depuis pointeurs bruts) ----
std::array<float, 4> float_buffer{1.0F, 2.0F, 3.0F, 4.0F};
std::array<int32_t, 3> int_buffer{1, 2, 3};
std::array<double, 2> double_buffer{1.0, 2.0};
std::array<int64_t, 2> long_buffer{1, 2};
std::array<int16_t, 2> short_buffer{1, 2};
std::array<int8_t, 2> char_buffer{1, 2};
std::array<uint8_t, 2> byte_buffer{1, 2};
std::array<bool, 2> bool_buffer{true, false};

// ---- Buffers de copie (sorties) ----
std::array<float, 3> copy_float_buffer{};
std::array<int32_t, 3> copy_int_buffer{};
std::array<double, 3> copy_double_buffer{};
std::array<int64_t, 3> copy_long_buffer{};
std::array<int16_t, 3> copy_short_buffer{};
std::array<int8_t, 3> copy_char_buffer{};
std::array<uint8_t, 3> copy_byte_buffer{};
std::array<bool, 3> copy_bool_buffer{};

// ---- Paramètres pour tests paramétrés ----
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

  [[nodiscard]] auto build_tensor() -> torch::Tensor const
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
        FromRawPtrParam{
            at::kFloat, float_buffer.data(), {kShape2x2[0], kShape2x2[1]}},
        FromRawPtrParam{at::kInt, int_buffer.data(), {kShape3[0]}},
        FromRawPtrParam{at::kDouble, double_buffer.data(), {kShape2[0]}},
        FromRawPtrParam{at::kLong, long_buffer.data(), {kShape2[0]}},
        FromRawPtrParam{at::kShort, short_buffer.data(), {kShape2[0]}},
        FromRawPtrParam{at::kChar, char_buffer.data(), {kShape2[0]}},
        FromRawPtrParam{at::kByte, byte_buffer.data(), {kShape2[0]}},
        FromRawPtrParam{at::kBool, bool_buffer.data(), {kShape2[0]}}));

TEST(TensorBuilder, FromRawPtrUnsupportedHalf)
{
  std::array<uint16_t, 1> buffer{0};
  const std::vector<int64_t> shape{kShape1.begin(), kShape1.end()};
  const torch::Device device(torch::kCPU);
  EXPECT_THROW(
      starpu_server::TensorBuilder::from_raw_ptr(
          reinterpret_cast<uintptr_t>(buffer.data()), at::kHalf, shape, device),
      starpu_server::InferenceExecutionException);
}

TEST(TensorBuilder, FromRawPtrUnsupportedComplex)
{
  std::array<std::complex<float>, 1> buffer{
      std::complex<float>(kZeroF, kZeroF)};
  const std::vector<int64_t> shape{kShape1.begin(), kShape1.end()};
  const torch::Device device(torch::kCPU);
  EXPECT_THROW(
      starpu_server::TensorBuilder::from_raw_ptr(
          reinterpret_cast<uintptr_t>(buffer.data()), at::kComplexFloat, shape,
          device),
      starpu_server::InferenceExecutionException);
}

class TensorBuilderCopyOutputToBuffer
    : public ::testing::TestWithParam<CopyOutputParam> {};

TEST_P(TensorBuilderCopyOutputToBuffer, CopiesCorrectly)
{
  const auto& param = GetParam();
  const auto numel = static_cast<std::size_t>(param.tensor.numel());

  switch (param.dtype) {
    case at::kFloat: {
      auto* buffer = static_cast<float*>(param.buffer);
      std::fill_n(buffer, numel, kZeroF);
      starpu_server::TensorBuilder::copy_output_to_buffer(
          param.tensor, buffer, numel);
      for (std::size_t i = 0; i < numel; ++i) {
        EXPECT_FLOAT_EQ(buffer[i], param.tensor[i].item<float>());
      }
      break;
    }
    case at::kInt: {
      auto* buffer = static_cast<int32_t*>(param.buffer);
      std::fill_n(buffer, numel, kZeroI);
      starpu_server::TensorBuilder::copy_output_to_buffer(
          param.tensor, buffer, numel);
      for (std::size_t i = 0; i < numel; ++i) {
        EXPECT_EQ(buffer[i], param.tensor[i].item<int32_t>());
      }
      break;
    }
    case at::kDouble: {
      auto* buffer = static_cast<double*>(param.buffer);
      std::fill_n(buffer, numel, kZeroD);
      starpu_server::TensorBuilder::copy_output_to_buffer(
          param.tensor, buffer, numel);
      for (std::size_t i = 0; i < numel; ++i) {
        EXPECT_DOUBLE_EQ(buffer[i], param.tensor[i].item<double>());
      }
      break;
    }
    case at::kLong: {
      auto* buffer = static_cast<int64_t*>(param.buffer);
      std::fill_n(buffer, numel, kZeroL);
      starpu_server::TensorBuilder::copy_output_to_buffer(
          param.tensor, buffer, numel);
      for (std::size_t i = 0; i < numel; ++i) {
        EXPECT_EQ(buffer[i], param.tensor[i].item<int64_t>());
      }
      break;
    }
    case at::kShort: {
      auto* buffer = static_cast<int16_t*>(param.buffer);
      std::fill_n(buffer, numel, kZeroS);
      starpu_server::TensorBuilder::copy_output_to_buffer(
          param.tensor, buffer, numel);
      for (std::size_t i = 0; i < numel; ++i) {
        EXPECT_EQ(buffer[i], param.tensor[i].item<int16_t>());
      }
      break;
    }
    case at::kChar: {
      auto* buffer = static_cast<int8_t*>(param.buffer);
      std::fill_n(buffer, numel, kZeroC);
      starpu_server::TensorBuilder::copy_output_to_buffer(
          param.tensor, buffer, numel);
      for (std::size_t i = 0; i < numel; ++i) {
        EXPECT_EQ(buffer[i], param.tensor[i].item<int8_t>());
      }
      break;
    }
    case at::kByte: {
      auto* buffer = static_cast<uint8_t*>(param.buffer);
      std::fill_n(buffer, numel, kZeroB);
      starpu_server::TensorBuilder::copy_output_to_buffer(
          param.tensor, buffer, numel);
      for (std::size_t i = 0; i < numel; ++i) {
        EXPECT_EQ(buffer[i], param.tensor[i].item<uint8_t>());
      }
      break;
    }
    case at::kBool: {
      auto* buffer = static_cast<bool*>(param.buffer);
      std::fill_n(buffer, numel, kZeroBool);
      starpu_server::TensorBuilder::copy_output_to_buffer(
          param.tensor, buffer, numel);
      for (std::size_t i = 0; i < numel; ++i) {
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
            at::kFloat, copy_float_buffer.data(),
            torch::tensor(
                {1.0F, 2.0F, 3.0F}, torch::TensorOptions().dtype(at::kFloat))},
        CopyOutputParam{
            at::kInt, copy_int_buffer.data(),
            torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kInt))},
        CopyOutputParam{
            at::kDouble, copy_double_buffer.data(),
            torch::tensor(
                {1.0, 2.0, 3.0}, torch::TensorOptions().dtype(at::kDouble))},
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

TEST(TensorBuilder, CopyOutputToBufferSizeMismatch)
{
  auto tensor = torch::tensor({1, 2}, torch::TensorOptions().dtype(at::kInt));
  int32_t buffer[kElems2];
  EXPECT_THROW(
      starpu_server::TensorBuilder::copy_output_to_buffer(
          tensor, buffer, kElems3),
      starpu_server::InferenceExecutionException);
}

TEST(TensorBuilder, CopyOutputToBufferExpectedNumelTooSmall)
{
  auto tensor =
      torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kInt));
  int32_t buffer[kElems3];
  EXPECT_THROW(
      starpu_server::TensorBuilder::copy_output_to_buffer(
          tensor, buffer, kElems2),
      starpu_server::InferenceExecutionException);
}

TEST(TensorBuilder, CopyOutputToBufferUnsupportedType)
{
  auto tensor =
      torch::zeros({kDim2}, torch::TensorOptions().dtype(at::kComplexFloat));
  std::array<float, kElems2> buffer{kZeroF, kZeroF};
  EXPECT_THROW(
      starpu_server::TensorBuilder::copy_output_to_buffer(
          tensor, buffer.data(), kElems2),
      starpu_server::InferenceExecutionException);
}

TEST(TensorBuilder, FromStarpuBuffersTooManyInputs)
{
  starpu_server::InferenceParams params;
  params.num_inputs = starpu_server::InferLimits::MaxInputs + 1;
  std::vector<void*> buffers(params.num_inputs, nullptr);
  const torch::Device device(torch::kCPU);
  EXPECT_THROW(
      starpu_server::TensorBuilder::from_starpu_buffers(
          &params, buffers, device),
      starpu_server::InferenceExecutionException);
}

TEST(TensorBuilder, FromStarpuBuffersSuccess)
{
  // Données d'entrée 2x2
  std::array<float, 4> data{1.0F, 2.0F, 3.0F, 4.0F};

  starpu_variable_interface fake_var;
  fake_var.ptr = reinterpret_cast<uintptr_t>(data.data());

  std::vector<void*> buffers = {&fake_var};

  starpu_server::InferenceParams params;
  params.num_inputs = 1;
  params.layout.input_types = {at::kFloat};
  params.layout.num_dims = {static_cast<int>(kShape2x2.size())};
  params.layout.dims = {{kShape2x2[0], kShape2x2[1]}};

  const torch::Device device(torch::kCPU);

  auto tensors = starpu_server::TensorBuilder::from_starpu_buffers(
      &params, buffers, device);
  ASSERT_EQ(tensors.size(), 1U);
  EXPECT_EQ(
      tensors[0].sizes(), (torch::IntArrayRef{kShape2x2[0], kShape2x2[1]}));
  EXPECT_EQ(tensors[0].dtype(), torch::kFloat);
  EXPECT_EQ(tensors[0].device(), device);
  EXPECT_EQ(tensors[0].data_ptr<float>(), data.data());
}

TEST(TensorBuilder, FromRawPtrUnsupportedQuantized)
{
  uint8_t dummy = 0;
  const std::vector<int64_t> shape{kShape1.begin(), kShape1.end()};
  const torch::Device device(torch::kCPU);
  EXPECT_THROW(
      starpu_server::TensorBuilder::from_raw_ptr(
          reinterpret_cast<uintptr_t>(&dummy), at::kQInt8, shape, device),
      starpu_server::InferenceExecutionException);
}
