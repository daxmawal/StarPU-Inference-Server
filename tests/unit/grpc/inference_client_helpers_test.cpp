#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#define private public
#include "grpc/client/inference_client.hpp"
#undef private
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

constexpr const char* kTestServerAddress = "dns:///localhost:50051";

auto
make_test_channel() -> std::shared_ptr<grpc::Channel>
{
  return grpc::CreateChannel(
      kTestServerAddress, grpc::InsecureChannelCredentials());
}

inline void
append_raw_bytes(
    inference::ModelInferResponse& response, const std::string& raw)
{
  response.add_raw_output_contents()->assign(raw.data(), raw.size());
}

template <typename Fn>
auto
capture_stderr(Fn&& fn) -> std::string
{
  testing::internal::CaptureStderr();
  std::forward<Fn>(fn)();
  return testing::internal::GetCapturedStderr();
}

}  // namespace

namespace starpu_server {
namespace {
template <typename T>
auto
value_to_double(const T& value) -> double
{
  if constexpr (std::is_same_v<T, c10::Half>) {
    return static_cast<double>(static_cast<float>(value));
  } else if constexpr (std::is_same_v<T, c10::BFloat16>) {
    return static_cast<double>(static_cast<float>(value));
  } else if constexpr (std::is_same_v<T, bool>) {
    return value ? 1.0 : 0.0;
  } else {
    return static_cast<double>(value);
  }
}

template <typename T>
void
expect_decode_roundtrip(
    std::string_view datatype, const std::vector<T>& values, double tol = 0.0)
{
  inference::ModelInferResponse::InferOutputTensor tensor;
  tensor.set_datatype(std::string(datatype));
  const auto raw = make_raw_data(values);
  const auto decoded =
      decode_output_values(tensor, std::string_view(raw), values.size());
  ASSERT_EQ(decoded.size(), values.size());
  for (std::size_t idx = 0; idx < values.size(); ++idx) {
    const double expected = value_to_double(values[idx]);
    if (tol == 0.0) {
      EXPECT_DOUBLE_EQ(decoded[idx], expected) << "Mismatch at index " << idx;
    } else {
      EXPECT_NEAR(decoded[idx], expected, tol) << "Mismatch at index " << idx;
    }
  }
}
}  // namespace

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

TEST(InferenceClientHelpers, DecodeOutputValuesHandlesAllScalarTypes)
{
  expect_decode_roundtrip("FP32", std::vector<float>{1.25F, -3.5F, 0.0F});
  expect_decode_roundtrip("FP64", std::vector<double>{0.5, -8.25, 12.0});
  expect_decode_roundtrip(
      "FP16", std::vector<c10::Half>{c10::Half(1.5F), c10::Half(-0.75F)}, 1e-5);
  expect_decode_roundtrip(
      "BF16",
      std::vector<c10::BFloat16>{
          c10::BFloat16(2.0F), c10::BFloat16(-1.0F), c10::BFloat16(0.125F)},
      1e-3);
  expect_decode_roundtrip("INT32", std::vector<int32_t>{42, -17, 8192});
  expect_decode_roundtrip(
      "INT64",
      std::vector<int64_t>{std::numeric_limits<int32_t>::max(), -9000, 0});
  expect_decode_roundtrip("INT16", std::vector<int16_t>{-32768, 1234, 0});
  expect_decode_roundtrip("INT8", std::vector<int8_t>{-128, -1, 127});
  expect_decode_roundtrip("UINT8", std::vector<uint8_t>{0U, 17U, 255U});
  expect_decode_roundtrip("BOOL", std::vector<uint8_t>{0U, 1U, 0U});
}

TEST(InferenceClientHelpers, ValidateServerResponseIgnoresMissingExpected)
{
  auto channel = make_test_channel();
  InferenceClient client(channel, VerbosityLevel::Silent);
  AsyncClientCall call;
  call.request_id = 101;

  const auto err =
      capture_stderr([&] { client.validate_server_response(call); });
  EXPECT_TRUE(err.empty());
}

TEST(InferenceClientHelpers, ValidateServerResponseIgnoresEmptySummary)
{
  auto channel = make_test_channel();
  InferenceClient client(channel, VerbosityLevel::Silent);
  AsyncClientCall call;
  call.request_id = 102;
  call.expected_outputs = InferenceClient::OutputSummary{};

  const auto err =
      capture_stderr([&] { client.validate_server_response(call); });
  EXPECT_TRUE(err.empty());
}

TEST(InferenceClientHelpers, ValidateServerResponseSkipsEmptyExpectedEntry)
{
  auto channel = make_test_channel();
  InferenceClient client(channel, VerbosityLevel::Silent);
  AsyncClientCall call;
  call.request_id = 103;
  call.expected_outputs = InferenceClient::OutputSummary{{}};
  call.reply.add_outputs()->set_datatype("FP32");
  const auto raw = make_raw_data(std::vector<float>{1.0F});
  append_raw_bytes(call.reply, raw);

  const auto err =
      capture_stderr([&] { client.validate_server_response(call); });
  EXPECT_TRUE(err.empty());
}

TEST(InferenceClientHelpers, ValidateServerResponseWarnsWhenNoOutputs)
{
  auto channel = make_test_channel();
  InferenceClient client(channel, VerbosityLevel::Silent);
  AsyncClientCall call;
  call.request_id = 200;
  call.expected_outputs = InferenceClient::OutputSummary{{1.0}};

  const auto err =
      capture_stderr([&] { client.validate_server_response(call); });
  EXPECT_NE(
      err.find("server response does not contain outputs"), std::string::npos);
}

TEST(InferenceClientHelpers, ValidateServerResponseWarnsWhenRawMissing)
{
  auto channel = make_test_channel();
  InferenceClient client(channel, VerbosityLevel::Silent);
  AsyncClientCall call;
  call.request_id = 201;
  call.expected_outputs = InferenceClient::OutputSummary{{4.0}, {8.0}};

  call.reply.add_outputs()->set_datatype("FP32");
  call.reply.add_outputs()->set_datatype("FP32");

  const auto raw = make_raw_data(std::vector<float>{4.0F});
  append_raw_bytes(call.reply, raw);

  const auto err =
      capture_stderr([&] { client.validate_server_response(call); });
  EXPECT_NE(err.find("missing raw contents"), std::string::npos);
}

TEST(InferenceClientHelpers, ValidateServerResponseLogsDecodeFailure)
{
  auto channel = make_test_channel();
  InferenceClient client(channel, VerbosityLevel::Silent);
  AsyncClientCall call;
  call.request_id = 202;
  call.expected_outputs = InferenceClient::OutputSummary{{5.0}};

  auto* tensor = call.reply.add_outputs();
  tensor->set_datatype("COMPLEX64");
  const std::string raw(8, '\0');
  append_raw_bytes(call.reply, raw);

  const auto err =
      capture_stderr([&] { client.validate_server_response(call); });
  EXPECT_NE(err.find("failed to decode server output"), std::string::npos);
}

TEST(InferenceClientHelpers, ValidateServerResponseWarnsOnCountMismatch)
{
  auto channel = make_test_channel();
  InferenceClient client(channel, VerbosityLevel::Silent);
  AsyncClientCall call;
  call.request_id = 203;
  call.expected_outputs = InferenceClient::OutputSummary{{1.0, 2.0}};

  auto* tensor = call.reply.add_outputs();
  tensor->set_datatype("FP32");
  const auto raw = make_raw_data(std::vector<float>{1.0F});
  append_raw_bytes(call.reply, raw);

  const auto err =
      capture_stderr([&] { client.validate_server_response(call); });
  EXPECT_NE(err.find("expected 2 values, decoded 1"), std::string::npos);
}

TEST(InferenceClientHelpers, ValidateServerResponseWarnsOnValueMismatch)
{
  auto channel = make_test_channel();
  InferenceClient client(channel, VerbosityLevel::Silent);
  AsyncClientCall call;
  call.request_id = 204;
  call.expected_outputs = InferenceClient::OutputSummary{{1.0, 2.0}};

  auto* tensor = call.reply.add_outputs();
  tensor->set_datatype("FP32");
  const auto raw = make_raw_data(std::vector<float>{1.0F, 2.01F});
  append_raw_bytes(call.reply, raw);

  const auto err =
      capture_stderr([&] { client.validate_server_response(call); });
  EXPECT_NE(err.find("value 1 mismatch"), std::string::npos);
}

TEST(InferenceClientHelpers, ValidateServerResponseLogsTraceOnSuccess)
{
  auto channel = make_test_channel();
  InferenceClient client(channel, VerbosityLevel::Trace);
  AsyncClientCall call;
  call.request_id = 205;
  call.expected_outputs = InferenceClient::OutputSummary{{3.5, -7.25}};

  auto* tensor = call.reply.add_outputs();
  tensor->set_datatype("FP32");
  const auto raw = make_raw_data(std::vector<float>{3.5F, -7.25F});
  append_raw_bytes(call.reply, raw);

  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  client.validate_server_response(call);
  const std::string err = testing::internal::GetCapturedStderr();
  const std::string out = testing::internal::GetCapturedStdout();

  EXPECT_TRUE(err.empty());
  EXPECT_NE(out.find("validated on 2 values"), std::string::npos);
}

TEST(InferenceClientHelpers, ValidateServerResponseWarnsWhenServerOutputsTooFew)
{
  auto channel = make_test_channel();
  InferenceClient client(channel, VerbosityLevel::Silent);
  AsyncClientCall call;
  call.request_id = 206;
  call.expected_outputs = InferenceClient::OutputSummary{{9.0}, {10.0}};

  auto* tensor = call.reply.add_outputs();
  tensor->set_datatype("FP32");
  const auto raw = make_raw_data(std::vector<float>{9.0F});
  append_raw_bytes(call.reply, raw);

  const auto err =
      capture_stderr([&] { client.validate_server_response(call); });
  EXPECT_NE(
      err.find("server returned 1 outputs but 2 were available"),
      std::string::npos);
}

}  // namespace starpu_server
