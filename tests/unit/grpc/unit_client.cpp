#include <gtest/gtest.h>

#include "grpc/client/inference_client.hpp"
#include "test_helpers.hpp"

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
