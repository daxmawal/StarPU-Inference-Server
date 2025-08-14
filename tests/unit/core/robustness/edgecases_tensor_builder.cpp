#include <gtest/gtest.h>

#include <array>
#include <complex>
#include <vector>

#define private public
#include "core/tensor_builder.hpp"
#undef private

#include "utils/exceptions.hpp"

namespace {
constexpr size_t kElems2 = 2, kElems3 = 3;
}

TEST(TensorBuilder_Robustesse, FromRawPtrUnsupportedHalf)
{
  std::array<uint16_t, 1> buf{0};
  EXPECT_THROW(
      starpu_server::TensorBuilder::from_raw_ptr(
          reinterpret_cast<uintptr_t>(buf.data()), at::kHalf,
          std::vector<int64_t>{1}, torch::kCPU),
      starpu_server::InferenceExecutionException);
}

TEST(TensorBuilder_Robustesse, FromRawPtrUnsupportedComplex)
{
  std::array<std::complex<float>, 1> buf{std::complex<float>(0.f, 0.f)};
  EXPECT_THROW(
      starpu_server::TensorBuilder::from_raw_ptr(
          reinterpret_cast<uintptr_t>(buf.data()), at::kComplexFloat,
          std::vector<int64_t>{1}, torch::kCPU),
      starpu_server::InferenceExecutionException);
}

TEST(TensorBuilder_Robustesse, FromRawPtrUnsupportedQuantized)
{
  uint8_t dummy{};
  EXPECT_THROW(
      starpu_server::TensorBuilder::from_raw_ptr(
          reinterpret_cast<uintptr_t>(&dummy), at::kQInt8,
          std::vector<int64_t>{1}, torch::kCPU),
      starpu_server::InferenceExecutionException);
}

TEST(TensorBuilder_Robustesse, CopyOutputToBufferSizeMismatch_TooLarge)
{
  auto t = torch::tensor({1, 2}, torch::TensorOptions().dtype(at::kInt));
  int32_t buf[kElems2];
  EXPECT_THROW(
      starpu_server::TensorBuilder::copy_output_to_buffer(t, buf, kElems3),
      starpu_server::InferenceExecutionException);
}

TEST(TensorBuilder_Robustesse, CopyOutputToBufferSizeMismatch_TooSmall)
{
  auto t = torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kInt));
  int32_t buf[kElems3];
  EXPECT_THROW(
      starpu_server::TensorBuilder::copy_output_to_buffer(t, buf, kElems2),
      starpu_server::InferenceExecutionException);
}

TEST(TensorBuilder_Robustesse, CopyOutputToBufferUnsupportedType)
{
  auto t = torch::zeros({2}, torch::TensorOptions().dtype(at::kComplexFloat));
  std::array<float, 2> buf{0.f, 0.f};
  EXPECT_THROW(
      starpu_server::TensorBuilder::copy_output_to_buffer(t, buf.data(), 2),
      starpu_server::InferenceExecutionException);
}

TEST(TensorBuilder_Robustesse, FromStarpuBuffersTooManyInputs)
{
  starpu_server::InferenceParams params;
  params.num_inputs = starpu_server::InferLimits::MaxInputs + 1;
  std::vector<void*> buffers(params.num_inputs, nullptr);
  EXPECT_THROW(
      starpu_server::TensorBuilder::from_starpu_buffers(
          &params, buffers, torch::kCPU),
      starpu_server::InferenceExecutionException);
}
