#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <complex>
#include <vector>

#define private public
#include "core/tensor_builder.hpp"
#undef private

#include "utils/exceptions.hpp"

namespace {
constexpr size_t kElems2 = 2, kElems3 = 3;
constexpr float kF1 = 1.0F;
constexpr float kF2 = 2.0F;
constexpr float kF3 = 3.0F;
constexpr float kF4 = 4.0F;
}  // namespace

TEST(TensorBuilder_Robustesse, FromRawPtrUnsupportedHalf)
{
  std::array<uint16_t, 1> buf{0};
  EXPECT_THROW(
      starpu_server::TensorBuilder::from_raw_ptr(
          std::bit_cast<uintptr_t>(buf.data()), at::kHalf,
          std::vector<int64_t>{1}, torch::kCPU),
      starpu_server::InferenceExecutionException);
}

TEST(TensorBuilder_Robustesse, FromRawPtrUnsupportedComplex)
{
  std::array<std::complex<float>, 1> buf{std::complex<float>(0.F, 0.F)};
  EXPECT_THROW(
      starpu_server::TensorBuilder::from_raw_ptr(
          std::bit_cast<uintptr_t>(buf.data()), at::kComplexFloat,
          std::vector<int64_t>{1}, torch::kCPU),
      starpu_server::InferenceExecutionException);
}

TEST(TensorBuilder_Robustesse, FromRawPtrUnsupportedQuantized)
{
  uint8_t dummy{};
  EXPECT_THROW(
      starpu_server::TensorBuilder::from_raw_ptr(
          std::bit_cast<uintptr_t>(&dummy), at::kQInt8, std::vector<int64_t>{1},
          torch::kCPU),
      starpu_server::InferenceExecutionException);
}

TEST(TensorBuilder_Robustesse, CopyOutputToBufferSizeMismatch_TooLarge)
{
  auto tensor = torch::tensor({1, 2}, torch::TensorOptions().dtype(at::kInt));
  std::array<int32_t, kElems2> buf{};
  EXPECT_THROW(
      starpu_server::TensorBuilder::copy_output_to_buffer(
          tensor, buf.data(), kElems3, tensor.scalar_type()),
      starpu_server::InferenceExecutionException);
}

TEST(TensorBuilder_Robustesse, CopyOutputToBufferSizeMismatch_TooSmall)
{
  auto tensor =
      torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kInt));
  std::array<int32_t, kElems2> buf{};
  EXPECT_THROW(
      starpu_server::TensorBuilder::copy_output_to_buffer(
          tensor, buf.data(), kElems2, tensor.scalar_type()),
      starpu_server::InferenceExecutionException);
}

TEST(TensorBuilder_Robustesse, CopyOutputToBufferNonContiguous)
{
  auto tensor = torch::tensor(
      {{kF1, kF2}, {kF3, kF4}}, torch::TensorOptions().dtype(at::kFloat));
  auto transposed = tensor.transpose(0, 1);
  EXPECT_FALSE(transposed.is_contiguous());
  std::array<float, 4> buf{};
  EXPECT_THROW(
      starpu_server::TensorBuilder::copy_output_to_buffer(
          transposed, buf.data(), transposed.numel(), transposed.scalar_type()),
      starpu_server::InferenceExecutionException);
}

TEST(TensorBuilder_Robustesse, CopyOutputToBufferNullPointer)
{
  auto tensor = torch::tensor({1, 2}, torch::TensorOptions().dtype(at::kInt));
  int32_t* null_ptr = nullptr;
  EXPECT_THROW(
      starpu_server::TensorBuilder::copy_output_to_buffer(
          tensor, null_ptr, tensor.numel(), tensor.scalar_type()),
      starpu_server::InferenceExecutionException);
}

TEST(TensorBuilder_Robustesse, CopyOutputToBufferTypeMismatch_FloatVsInt)
{
  auto tensor =
      torch::tensor({kF1, kF2}, torch::TensorOptions().dtype(at::kFloat));
  std::array<float, kElems2> buf{};
  EXPECT_THROW(
      starpu_server::TensorBuilder::copy_output_to_buffer(
          tensor, buf.data(), tensor.numel(), at::kInt),
      starpu_server::InferenceExecutionException);
}

TEST(TensorBuilder_Robustesse, CopyOutputToBufferTypeMismatch_IntVsFloat)
{
  auto tensor = torch::tensor({1, 2}, torch::TensorOptions().dtype(at::kInt));
  std::array<int32_t, kElems2> buf{};
  EXPECT_THROW(
      starpu_server::TensorBuilder::copy_output_to_buffer(
          tensor, buf.data(), tensor.numel(), at::kFloat),
      starpu_server::InferenceExecutionException);
}

TEST(TensorBuilder_Robustesse, FromStarpuBuffersTooManyInputs)
{
  starpu_server::InferenceParams params;
  params.num_inputs = starpu_server::InferLimits::MaxInputs + 1;
  params.limits.max_inputs = starpu_server::InferLimits::MaxInputs;
  std::vector<void*> buffers(params.num_inputs, nullptr);
  EXPECT_THROW(
      starpu_server::TensorBuilder::from_starpu_buffers(
          &params, buffers, torch::kCPU),
      starpu_server::InferenceExecutionException);
}
