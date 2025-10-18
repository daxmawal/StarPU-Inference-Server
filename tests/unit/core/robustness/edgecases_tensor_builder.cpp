#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <complex>
#include <cstddef>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

#define private public
#include "core/tensor_builder.hpp"
#undef private

#include "test_constants.hpp"
#include "utils/exceptions.hpp"

namespace {
constexpr size_t kElems2 = 2, kElems3 = 3;
using starpu_server::test_constants::kF1;
using starpu_server::test_constants::kF2;
using starpu_server::test_constants::kF3;
using starpu_server::test_constants::kF4;
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

struct CopyOutputToBufferParam {
  struct BufferInfo {
    std::span<std::byte> data;
    int64_t expected_elements;
    at::ScalarType expected_dtype;
    std::shared_ptr<void> storage;
  };

  std::string description;
  std::function<torch::Tensor()> tensor_factory;
  std::function<BufferInfo(const torch::Tensor&)> buffer_factory;
};

class CopyOutputToBufferTest
    : public ::testing::TestWithParam<CopyOutputToBufferParam> {};

TEST_P(CopyOutputToBufferTest, CopyOutputToBufferThrows)
{
  const auto& param = GetParam();
  auto tensor = param.tensor_factory();
  auto buffer = param.buffer_factory(tensor);

  EXPECT_THROW(
      starpu_server::TensorBuilder::copy_output_to_buffer(
          tensor, buffer.data, buffer.expected_elements, buffer.expected_dtype),
      starpu_server::InferenceExecutionException);
}

INSTANTIATE_TEST_SUITE_P(
    CopyOutputToBufferFailureCases, CopyOutputToBufferTest,
    ::testing::Values(
        CopyOutputToBufferParam{
            "SizeTooLarge",
            []() {
              return torch::tensor(
                  {1, 2}, torch::TensorOptions().dtype(at::kInt));
            },
            [](const torch::Tensor& tensor) {
              auto buffer = std::make_shared<std::vector<int32_t>>(kElems2);
              auto buffer_view =
                  std::span<int32_t>(buffer->data(), buffer->size());
              auto span_bytes = std::as_writable_bytes(buffer_view);
              return CopyOutputToBufferParam::BufferInfo{
                  span_bytes, static_cast<int64_t>(kElems3),
                  tensor.scalar_type(),
                  std::shared_ptr<void>(
                      buffer, static_cast<void*>(buffer->data()))};
            }},
        CopyOutputToBufferParam{
            "SizeTooSmall",
            []() {
              return torch::tensor(
                  {1, 2, 3}, torch::TensorOptions().dtype(at::kInt));
            },
            [](const torch::Tensor& tensor) {
              auto buffer = std::make_shared<std::vector<int32_t>>(kElems2);
              auto buffer_view =
                  std::span<int32_t>(buffer->data(), buffer->size());
              auto span_bytes = std::as_writable_bytes(buffer_view);
              return CopyOutputToBufferParam::BufferInfo{
                  span_bytes, static_cast<int64_t>(kElems2),
                  tensor.scalar_type(),
                  std::shared_ptr<void>(
                      buffer, static_cast<void*>(buffer->data()))};
            }},
        CopyOutputToBufferParam{
            "NonContiguous",
            []() {
              auto tensor = torch::tensor(
                  {{kF1, kF2}, {kF3, kF4}},
                  torch::TensorOptions().dtype(at::kFloat));
              return tensor.transpose(0, 1);
            },
            [](const torch::Tensor& tensor) {
              EXPECT_FALSE(tensor.is_contiguous());
              auto buffer =
                  std::make_shared<std::vector<float>>(tensor.numel());
              auto buffer_view =
                  std::span<float>(buffer->data(), buffer->size());
              auto span_bytes = std::as_writable_bytes(buffer_view);
              return CopyOutputToBufferParam::BufferInfo{
                  span_bytes, tensor.numel(), tensor.scalar_type(),
                  std::shared_ptr<void>(
                      buffer, static_cast<void*>(buffer->data()))};
            }},
        CopyOutputToBufferParam{
            "NullPointer",
            []() {
              return torch::tensor(
                  {1, 2}, torch::TensorOptions().dtype(at::kInt));
            },
            [](const torch::Tensor& tensor) {
              return CopyOutputToBufferParam::BufferInfo{
                  std::span<std::byte>(),
                  tensor.numel(),
                  tensor.scalar_type(),
                  {}};
            }},
        CopyOutputToBufferParam{
            "TypeMismatchFloatVsInt",
            []() {
              return torch::tensor(
                  {kF1, kF2}, torch::TensorOptions().dtype(at::kFloat));
            },
            [](const torch::Tensor& tensor) {
              auto buffer =
                  std::make_shared<std::vector<float>>(tensor.numel());
              auto buffer_view =
                  std::span<float>(buffer->data(), buffer->size());
              auto span_bytes = std::as_writable_bytes(buffer_view);
              return CopyOutputToBufferParam::BufferInfo{
                  span_bytes, tensor.numel(), at::kInt,
                  std::shared_ptr<void>(
                      buffer, static_cast<void*>(buffer->data()))};
            }},
        CopyOutputToBufferParam{
            "TypeMismatchIntVsFloat",
            []() {
              return torch::tensor(
                  {1, 2}, torch::TensorOptions().dtype(at::kInt));
            },
            [](const torch::Tensor& tensor) {
              auto buffer =
                  std::make_shared<std::vector<int32_t>>(tensor.numel());
              auto buffer_view =
                  std::span<int32_t>(buffer->data(), buffer->size());
              auto span_bytes = std::as_writable_bytes(buffer_view);
              return CopyOutputToBufferParam::BufferInfo{
                  span_bytes, tensor.numel(), at::kFloat,
                  std::shared_ptr<void>(
                      buffer, static_cast<void*>(buffer->data()))};
            }}),
    [](const ::testing::TestParamInfo<CopyOutputToBufferParam>& info) {
      return info.param.description;
    });

TEST(TensorBuilder_Robustesse, FromStarpuBuffersTooManyInputs)
{
  starpu_server::InferenceParams params;
  params.num_inputs = starpu_server::InferLimits::MaxInputs + 1;
  params.limits.max_inputs = starpu_server::InferLimits::MaxInputs;
  std::vector<void*> buffers(params.num_inputs, nullptr);
  EXPECT_THROW(
      [[maybe_unused]] auto _ =
          starpu_server::TensorBuilder::from_starpu_buffers(
              &params, buffers, torch::kCPU),
      starpu_server::InferenceExecutionException);
}
