#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "grpc/client/client_args.hpp"

namespace starpu_server {
VerbosityLevel parse_verbosity_level(const std::string& val);
}

TEST(ClientArgs, ParsesValidArguments)
{
  const char* argv[] = {
      "prog",
      "--shape",
      "1x3x224x224",
      "--type",
      "float32",
      "--server",
      "localhost:1234",
      "--model",
      "my_model",
      "--version",
      "2",
      "--iterations",
      "5",
      "--delay",
      "10",
      "--verbose",
      "2"};
  auto cfg = starpu_server::parse_client_args(std::span{argv});
  EXPECT_TRUE(cfg.valid);
  EXPECT_EQ(cfg.shape, (std::vector<int64_t>{1, 3, 224, 224}));
  EXPECT_EQ(cfg.type, at::kFloat);
  EXPECT_EQ(cfg.server_address, "localhost:1234");
  EXPECT_EQ(cfg.model_name, "my_model");
  EXPECT_EQ(cfg.model_version, "2");
  EXPECT_EQ(cfg.iterations, 5);
  EXPECT_EQ(cfg.delay_ms, 10);
  EXPECT_EQ(cfg.verbosity, starpu_server::VerbosityLevel::Stats);
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

TEST(ClientArgs, VerboseLevels)
{
  using enum starpu_server::VerbosityLevel;
  const std::array<std::pair<const char*, starpu_server::VerbosityLevel>, 4>
      cases = {{{"0", Silent}, {"1", Info}, {"3", Debug}, {"4", Trace}}};
  for (const auto& [level_str, expected] : cases) {
    const char* argv[] = {"prog", "--shape", "1", "--verbose", level_str};
    auto cfg = starpu_server::parse_client_args(std::span{argv});
    ASSERT_TRUE(cfg.valid);
    EXPECT_EQ(cfg.verbosity, expected);
  }
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
