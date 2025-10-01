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
      "prog", "--shape", "1", "--type", "unknown"};
  auto cfg = starpu_server::parse_client_args(std::span{argv});
  EXPECT_FALSE(cfg.valid);
}

TEST(ClientArgs, NegativeIterationsMarkedInvalid)
{
  constexpr std::size_t kArgc = 5;
  std::array<const char*, kArgc> argv = {
      "prog", "--shape", "1", "--iterations", "-3"};
  auto cfg = starpu_server::parse_client_args(std::span{argv});
  EXPECT_FALSE(cfg.valid);
}

TEST(ClientArgs, InvalidVerboseValuesMarkedInvalid)
{
  constexpr std::size_t kArgc = 5;
  std::array<const char*, kArgc> neg = {
      "prog", "--shape", "1", "--verbose", "-1"};
  EXPECT_FALSE(starpu_server::parse_client_args(std::span{neg}).valid);

  std::array<const char*, kArgc> high = {
      "prog", "--shape", "1", "--verbose", "5"};
  EXPECT_FALSE(starpu_server::parse_client_args(std::span{high}).valid);

  std::array<const char*, kArgc> str = {
      "prog", "--shape", "1", "--verbose", "foo"};
  EXPECT_FALSE(starpu_server::parse_client_args(std::span{str}).valid);
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
  EXPECT_FALSE(client.ServerIsLive());
}

TEST(InferenceClient, ServerIsReadyReturnsFalseWhenUnavailable)
{
  auto channel = grpc::CreateChannel(
      "127.0.0.1:59998", grpc::InsecureChannelCredentials());
  starpu_server::InferenceClient client(
      channel, starpu_server::VerbosityLevel::Silent);
  EXPECT_FALSE(client.ServerIsReady());
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
