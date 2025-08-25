#include <gtest/gtest.h>

#include "grpc/client/inference_client.hpp"
#include "test_helpers.hpp"

namespace starpu_server {
auto parse_verbosity_level(const std::string& val) -> VerbosityLevel;
}

TEST(ClientArgs, InvalidTypeIsDetected)
{
  const char* argv[] = {"prog", "--shape", "1", "--type", "unknown"};
  auto cfg = starpu_server::parse_client_args(std::span{argv});
  EXPECT_FALSE(cfg.valid);
}

TEST(ClientArgs, NegativeIterationsMarkedInvalid)
{
  const char* argv[] = {"prog", "--shape", "1", "--iterations", "-3"};
  auto cfg = starpu_server::parse_client_args(std::span{argv});
  EXPECT_FALSE(cfg.valid);
}

TEST(ClientArgs, InvalidVerboseValuesMarkedInvalid)
{
  const char* neg[] = {"prog", "--shape", "1", "--verbose", "-1"};
  EXPECT_FALSE(starpu_server::parse_client_args(std::span{neg}).valid);

  const char* high[] = {"prog", "--shape", "1", "--verbose", "5"};
  EXPECT_FALSE(starpu_server::parse_client_args(std::span{high}).valid);

  const char* str[] = {"prog", "--shape", "1", "--verbose", "foo"};
  EXPECT_FALSE(starpu_server::parse_client_args(std::span{str}).valid);
}

TEST(ClientArgs, VerboseValueOutOfRangeThrows)
{
  const std::string big = std::to_string(std::numeric_limits<int>::max()) + "0";
  EXPECT_THROW(starpu_server::parse_verbosity_level(big), std::out_of_range);
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
    ::testing::Values("-1", "5", "foo"));
