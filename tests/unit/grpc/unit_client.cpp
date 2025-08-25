#include <gtest/gtest.h>

#include <utility>

#include "grpc/client/client_args.hpp"
#include "grpc/client/inference_client.hpp"
#include "grpc/server/inference_service.hpp"
#include "test_helpers.hpp"

TEST(ClientArgs, ParsesValidArguments)
{
  const char* argv[] = {
      "prog",
      "--input",
      "input:1x3x224x224:float32",
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
  ASSERT_EQ(cfg.inputs.size(), 1U);
  EXPECT_EQ(cfg.inputs[0].shape, (std::vector<int64_t>{1, 3, 224, 224}));
  EXPECT_EQ(cfg.inputs[0].type, at::kFloat);
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

TEST(ClientArgsHelp, ContainsKeyOptions)
{
  testing::internal::CaptureStdout();
  starpu_server::display_client_help("prog");
  const std::string out = testing::internal::GetCapturedStdout();
  EXPECT_NE(out.find("--iterations"), std::string::npos);
  EXPECT_NE(out.find("--delay"), std::string::npos);
  EXPECT_NE(out.find("--shape"), std::string::npos);
  EXPECT_NE(out.find("--type"), std::string::npos);
  EXPECT_NE(out.find("--input"), std::string::npos);
  EXPECT_NE(out.find("--server"), std::string::npos);
  EXPECT_NE(out.find("--model"), std::string::npos);
  EXPECT_NE(out.find("--version"), std::string::npos);
  EXPECT_NE(out.find("--verbose"), std::string::npos);
  EXPECT_NE(out.find("--help"), std::string::npos);
}


TEST(InferenceClient, ModelIsReadyReturnsTrue)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> reference_outputs;
  auto server = starpu_server::start_test_grpc_server(queue, reference_outputs);

  auto channel = grpc::CreateChannel(
      "127.0.0.1:" + std::to_string(server.port),
      grpc::InsecureChannelCredentials());
  starpu_server::InferenceClient client(
      channel, starpu_server::VerbosityLevel::Silent);
  EXPECT_TRUE(client.ModelIsReady("example", "1"));

  starpu_server::StopServer(server.server);
  server.thread.join();
  EXPECT_EQ(server.server, nullptr);
}

TEST(InferenceClient, ModelIsReadyReturnsFalseWhenUnavailable)
{
  auto channel = grpc::CreateChannel(
      "127.0.0.1:59997", grpc::InsecureChannelCredentials());
  starpu_server::InferenceClient client(
      channel, starpu_server::VerbosityLevel::Silent);
  EXPECT_FALSE(client.ModelIsReady("example", "1"));
}

class ParseVerbosityLevelValid
    : public ::testing::TestWithParam<
          std::pair<const char*, starpu_server::VerbosityLevel>> {};

TEST_P(ParseVerbosityLevelValid, ReturnsExpectedEnum)
{
  const auto& [input, expected] = GetParam();
  EXPECT_EQ(starpu_server::parse_verbosity_level(input), expected);
}

INSTANTIATE_TEST_SUITE_P(
    ParseVerbosityLevel, ParseVerbosityLevelValid,
    ::testing::Values(
        std::pair{"0", starpu_server::VerbosityLevel::Silent},
        std::pair{"1", starpu_server::VerbosityLevel::Info},
        std::pair{"2", starpu_server::VerbosityLevel::Stats},
        std::pair{"3", starpu_server::VerbosityLevel::Debug},
        std::pair{"4", starpu_server::VerbosityLevel::Trace}));
