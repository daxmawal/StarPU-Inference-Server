#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Include the implementation to exercise helpers with internal linkage.
#include "grpc/client/inference_client.cpp"

namespace {
template <typename T>
auto
make_raw_data(const std::vector<T>& values) -> std::string
{
  std::string raw(values.size() * sizeof(T), '\0');
  std::memcpy(raw.data(), values.data(), raw.size());
  return raw;
}

template <typename T>
void
expect_matches(
    const std::vector<double>& actual, const std::vector<T>& expected)
{
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t idx = 0; idx < actual.size(); ++idx) {
    EXPECT_DOUBLE_EQ(actual[idx], static_cast<double>(expected[idx]))
        << "Mismatch at index " << idx;
  }
}
}  // namespace

namespace starpu_server {

TEST(InferenceClientHelpers, AppendConvertedValuesPrimitives)
{
  std::vector<double> destination;
  const std::vector<float> floats = {1.25F, -3.5F, 0.0F};
  const auto float_raw = make_raw_data(floats);
  append_converted_values<float>(
      destination, std::string_view(float_raw), floats.size());
  expect_matches(destination, floats);

  destination.clear();
  const std::vector<int32_t> ints = {42, -17, 8192};
  const auto int_raw = make_raw_data(ints);
  append_converted_values<int32_t>(
      destination, std::string_view(int_raw), ints.size());
  expect_matches(destination, ints);
}

TEST(InferenceClientHelpers, AppendConvertedValuesHalfPrecision)
{
  std::vector<double> destination;
  const std::vector<c10::Half> halves = {c10::Half(0.5F), c10::Half(-10.25F)};
  const auto raw = make_raw_data(halves);
  append_converted_values<c10::Half>(
      destination, std::string_view(raw), halves.size());
  ASSERT_EQ(destination.size(), halves.size());
  for (std::size_t idx = 0; idx < halves.size(); ++idx) {
    EXPECT_NEAR(destination[idx], static_cast<float>(halves[idx]), 1e-6)
        << "Mismatch at index " << idx;
  }
}

TEST(InferenceClientHelpers, AppendConvertedValuesBFloat16)
{
  std::vector<double> destination;
  const std::vector<c10::BFloat16> bfloats = {
      c10::BFloat16(1.0F), c10::BFloat16(-2.5F), c10::BFloat16(3.75F)};
  const auto raw = make_raw_data(bfloats);
  append_converted_values<c10::BFloat16>(
      destination, std::string_view(raw), bfloats.size());
  ASSERT_EQ(destination.size(), bfloats.size());
  for (std::size_t idx = 0; idx < bfloats.size(); ++idx) {
    EXPECT_NEAR(destination[idx], static_cast<float>(bfloats[idx]), 1e-3)
        << "Mismatch at index " << idx;
  }
}

TEST(InferenceClientHelpers, DecodeOutputValuesRespectsLimitAndDatatype)
{
  inference::ModelInferResponse::InferOutputTensor tensor;
  tensor.set_datatype("FP64");

  const std::vector<double> values = {1.5, -0.25, 8.0};
  const auto raw = make_raw_data(values);

  const auto limited = decode_output_values(tensor, std::string_view(raw), 2U);
  ASSERT_EQ(limited.size(), 2U);
  EXPECT_DOUBLE_EQ(limited[0], values[0]);
  EXPECT_DOUBLE_EQ(limited[1], values[1]);

  const auto bounded = decode_output_values(tensor, std::string_view(raw), 10U);
  ASSERT_EQ(bounded.size(), values.size());
  expect_matches(bounded, values);

  const auto empty = decode_output_values(tensor, std::string_view(raw), 0U);
  EXPECT_TRUE(empty.empty());
}

TEST(InferenceClientHelpers, DecodeOutputValuesSupportsBFloat16)
{
  inference::ModelInferResponse::InferOutputTensor tensor;
  tensor.set_datatype("BF16");

  const std::vector<c10::BFloat16> expected = {
      c10::BFloat16(0.0F), c10::BFloat16(-1.25F), c10::BFloat16(6.0F)};
  const auto raw = make_raw_data(expected);
  const auto decoded =
      decode_output_values(tensor, std::string_view(raw), expected.size());

  ASSERT_EQ(decoded.size(), expected.size());
  for (std::size_t idx = 0; idx < decoded.size(); ++idx) {
    EXPECT_NEAR(decoded[idx], static_cast<float>(expected[idx]), 1e-3)
        << "Mismatch at index " << idx;
  }
}

TEST(InferenceClientHelpers, DecodeOutputValuesRejectsUnsupportedDatatype)
{
  inference::ModelInferResponse::InferOutputTensor tensor;
  tensor.set_datatype("COMPLEX64");
  const std::string raw(16, '\0');
  EXPECT_THROW(
      decode_output_values(tensor, std::string_view(raw), 1U),
      std::invalid_argument);
}

}  // namespace starpu_server
