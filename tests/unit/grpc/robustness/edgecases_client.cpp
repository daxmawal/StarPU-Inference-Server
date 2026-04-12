#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <limits>

#include "grpc/client/inference_client.hpp"
#include "test_helpers.hpp"

TEST(ClientArgs, InvalidTypeIsDetected)
{
  constexpr std::size_t kArgc = 5;
  std::array<const char*, kArgc> argv = {
      "prog", "--input", "input:1:unknown", "--request-number", "1"};
  ::testing::internal::CaptureStderr();
  auto cfg = starpu_server::parse_client_args(std::span{argv});
  const auto err = ::testing::internal::GetCapturedStderr();
  EXPECT_FALSE(cfg.valid);
  EXPECT_NE(err.find("Unsupported type: unknown"), std::string::npos);
}

TEST(ClientArgs, MissingValueMarksConfigInvalid)
{
  constexpr std::size_t kArgc = 2;
  std::array<const char*, kArgc> argv = {"prog", "--input"};
  auto cfg = starpu_server::parse_client_args(std::span{argv});
  EXPECT_FALSE(cfg.valid);
}

TEST(ClientArgs, NegativeRequestNbMarkedInvalid)
{
  constexpr std::size_t kArgc = 5;
  std::array<const char*, kArgc> argv = {
      "prog", "--input", "input:1:float32", "--request-number", "-3"};
  ::testing::internal::CaptureStderr();
  auto cfg = starpu_server::parse_client_args(std::span{argv});
  const auto err = ::testing::internal::GetCapturedStderr();
  EXPECT_FALSE(cfg.valid);
  EXPECT_NE(err.find("Must be > 0."), std::string::npos);
}

TEST(ClientArgs, InvalidVerboseValuesMarkedInvalid)
{
  constexpr std::size_t kArgc = 5;
  std::array<const char*, kArgc> neg = {
      "prog", "--input", "input:1:float32", "--verbose", "-1"};
  ::testing::internal::CaptureStderr();
  EXPECT_FALSE(starpu_server::parse_client_args(std::span{neg}).valid);
  const auto neg_err = ::testing::internal::GetCapturedStderr();
  EXPECT_NE(neg_err.find("Invalid verbosity level"), std::string::npos);

  std::array<const char*, kArgc> high = {
      "prog", "--input", "input:1:float32", "--verbose", "5"};
  ::testing::internal::CaptureStderr();
  EXPECT_FALSE(starpu_server::parse_client_args(std::span{high}).valid);
  const auto high_err = ::testing::internal::GetCapturedStderr();
  EXPECT_NE(high_err.find("Invalid verbosity level"), std::string::npos);

  std::array<const char*, kArgc> str = {
      "prog", "--input", "input:1:float32", "--verbose", "foo"};
  ::testing::internal::CaptureStderr();
  EXPECT_FALSE(starpu_server::parse_client_args(std::span{str}).valid);
  const auto str_err = ::testing::internal::GetCapturedStderr();
  EXPECT_NE(str_err.find("Invalid verbosity level"), std::string::npos);
}

TEST(ClientArgs, VerboseValueOutOfRangeThrowsInvalidArgument)
{
  const std::string big = std::to_string(std::numeric_limits<int>::max()) + "0";
  EXPECT_THROW(
      starpu_server::parse_verbosity_level(big), std::invalid_argument);
}

TEST(InferenceClient, ServerIsLiveReturnsFalseWhenUnavailable)
{
  auto channel = grpc::CreateChannel(
      "127.0.0.1:59999", grpc::InsecureChannelCredentials());
  starpu_server::InferenceClient client(
      channel, starpu_server::VerbosityLevel::Silent);
  ::testing::internal::CaptureStderr();
  EXPECT_FALSE(client.ServerIsLive());
  const auto err = ::testing::internal::GetCapturedStderr();
  EXPECT_NE(err.find("Connection refused"), std::string::npos);
}

TEST(InferenceClient, ServerIsReadyReturnsFalseWhenUnavailable)
{
  auto channel = grpc::CreateChannel(
      "127.0.0.1:59998", grpc::InsecureChannelCredentials());
  starpu_server::InferenceClient client(
      channel, starpu_server::VerbosityLevel::Silent);
  ::testing::internal::CaptureStderr();
  EXPECT_FALSE(client.ServerIsReady());
  const auto err = ::testing::internal::GetCapturedStderr();
  EXPECT_NE(err.find("Connection refused"), std::string::npos);
}

class ParseVerbosityLevelInvalid
    : public ::testing::TestWithParam<const char*> {};

TEST_P(ParseVerbosityLevelInvalid, ThrowsInvalidArgument)
{
  EXPECT_THROW(
      starpu_server::parse_verbosity_level(GetParam()), std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(
    ParseVerbosityLevel, ParseVerbosityLevelInvalid,
    ::testing::Values("-1", "5", "foo", ""));
