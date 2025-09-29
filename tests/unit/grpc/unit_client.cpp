#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include "grpc/client/client_args.hpp"
#include "grpc/client/inference_client.hpp"
#include "grpc/server/inference_service.hpp"
#include "test_helpers.hpp"

TEST(ClientArgs, ParsesValidArguments)
{
  auto argv = std::to_array<const char*>(
      {"prog", "--input", "input:1x3x224x224:float32", "--server",
       "localhost:1234", "--model", "my_model", "--version", "2",
       "--iterations", "5", "--delay", "10", "--verbose", "2"});
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

TEST(ClientArgs, ShapeOverridesExistingInput)
{
  auto argv = std::to_array<const char*>(
      {"prog", "--input", "input:1:float32", "--shape", "2"});
  auto cfg = starpu_server::parse_client_args(std::span{argv});
  ASSERT_TRUE(cfg.valid);
  ASSERT_EQ(cfg.inputs.size(), 1U);
  EXPECT_EQ(cfg.inputs[0].shape, (std::vector<int64_t>{2}));
  EXPECT_EQ(cfg.shape, (std::vector<int64_t>{2}));
}

TEST(ClientArgs, HelpFlagSetsShowHelp)
{
  const auto help_flags = std::to_array<const char*>({"--help", "-h"});
  for (const auto* flag : help_flags) {
    SCOPED_TRACE(flag);
    auto argv = std::to_array<const char*>({"prog", flag});
    auto cfg = starpu_server::parse_client_args(std::span{argv});
    EXPECT_TRUE(cfg.show_help);
    EXPECT_TRUE(cfg.valid);
    EXPECT_TRUE(cfg.inputs.empty());
    EXPECT_TRUE(cfg.shape.empty());
  }
}

TEST(ClientArgs, MissingInputMarksConfigInvalid)
{
  auto argv = std::to_array<const char*>({"prog"});
  testing::internal::CaptureStderr();
  auto cfg = starpu_server::parse_client_args(std::span{argv});
  const std::string err = testing::internal::GetCapturedStderr();
  EXPECT_FALSE(cfg.valid);
  EXPECT_TRUE(cfg.inputs.empty());
  EXPECT_NE(err.find("--input option is required."), std::string::npos);
}

TEST(ClientArgs, VerboseLevels)
{
  using enum starpu_server::VerbosityLevel;
  const auto cases =
      std::to_array<std::pair<const char*, starpu_server::VerbosityLevel>>(
          {{"0", Silent}, {"1", Info}, {"3", Debug}, {"4", Trace}});
  for (const auto& [level_str, expected] : cases) {
    auto argv = std::to_array<const char*>(
        {"prog", "--shape", "1", "--verbose", level_str});
    auto cfg = starpu_server::parse_client_args(std::span{argv});
    ASSERT_TRUE(cfg.valid);
    EXPECT_EQ(cfg.verbosity, expected);
  }
}

TEST(ClientArgs, RejectsNonPositiveShapeDims)
{
  auto argv_neg = std::to_array<const char*>({"prog", "--shape", "1x-3x224"});
  auto cfg_neg = starpu_server::parse_client_args(std::span{argv_neg});
  EXPECT_FALSE(cfg_neg.valid);

  auto argv_zero = std::to_array<const char*>({"prog", "--shape", "1x0x224"});
  auto cfg_zero = starpu_server::parse_client_args(std::span{argv_zero});
  EXPECT_FALSE(cfg_zero.valid);
}

TEST(ClientArgs, RejectsMalformedShapeTokens)
{
  const auto cases =
      std::to_array<const char*>({"1xax2", "9223372036854775808", ""});
  for (const auto* shape : cases) {
    auto argv = std::to_array<const char*>({"prog", "--shape", shape});
    auto cfg = starpu_server::parse_client_args(std::span{argv});
    EXPECT_FALSE(cfg.valid);
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

class ParseInputTypeCase
    : public ::testing::TestWithParam<std::pair<const char*, at::ScalarType>> {
};

TEST_P(ParseInputTypeCase, ParsesExpectedType)
{
  const auto& [type_str, expected] = GetParam();
  const std::string arg = std::string{"input:1:"} + type_str;
  auto argv = std::to_array<const char*>({"prog", "--input", arg.c_str()});
  auto cfg = starpu_server::parse_client_args(std::span{argv});
  ASSERT_TRUE(cfg.valid);
  ASSERT_EQ(cfg.inputs.size(), 1U);
  EXPECT_EQ(cfg.inputs[0].type, expected);
}

INSTANTIATE_TEST_SUITE_P(
    SupportedTypes, ParseInputTypeCase,
    ::testing::Values(
        std::pair{"float32", at::kFloat}, std::pair{"float64", at::kDouble},
        std::pair{"float16", at::kHalf}, std::pair{"bfloat16", at::kBFloat16},
        std::pair{"int32", at::kInt}, std::pair{"int64", at::kLong},
        std::pair{"int16", at::kShort}, std::pair{"int8", at::kChar},
        std::pair{"uint8", at::kByte}, std::pair{"bool", at::kBool},
        std::pair{"complex64", at::kComplexFloat},
        std::pair{"complex128", at::kComplexDouble}));


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
  EXPECT_TRUE(client.ModelIsReady({"example", "1"}));

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
  EXPECT_FALSE(client.ModelIsReady({"example", "1"}));
}

TEST(InferenceClient, RejectsMismatchedTensorCount)
{
  auto channel =
      grpc::CreateChannel("localhost:0", grpc::InsecureChannelCredentials());
  starpu_server::InferenceClient client(
      channel, starpu_server::VerbosityLevel::Silent);

  starpu_server::ClientConfig cfg;
  cfg.model_name = "example";
  cfg.model_version = "1";
  cfg.inputs.push_back({"input0", {1}, at::kFloat});

  std::vector<torch::Tensor> tensors = {torch::zeros({1}), torch::zeros({1})};

  EXPECT_THROW(client.AsyncModelInfer(tensors, cfg), std::invalid_argument);
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

TEST(InferenceClient, AsyncCompleteRpcExitsAfterShutdown)
{
  auto channel = grpc::CreateChannel(
      "127.0.0.1:59996", grpc::InsecureChannelCredentials());
  starpu_server::InferenceClient client(
      channel, starpu_server::VerbosityLevel::Silent);

  std::thread cq_thread(
      &starpu_server::InferenceClient::AsyncCompleteRpc, &client);
  constexpr int kSleepMs = 50;
  std::this_thread::sleep_for(std::chrono::milliseconds(kSleepMs));
  client.Shutdown();
  cq_thread.join();
  SUCCEED();
}
