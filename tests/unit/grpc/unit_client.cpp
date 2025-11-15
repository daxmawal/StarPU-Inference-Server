#include <ATen/core/ScalarType.h>
#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "grpc/client/client_args.hpp"
#define private public
#include "grpc/client/inference_client.hpp"
#undef private
#include "grpc/server/inference_service.hpp"
#include "test_helpers.hpp"

TEST(ClientArgs, ParsesValidArguments)
{
  auto argv = std::to_array<const char*>(
      {"prog", "--input", "input:1x3x224x224:float32", "--server",
       "localhost:1234", "--model", "my_model", "--request-number", "5",
       "--delay", "10", "--verbose", "2"});
  auto cfg = starpu_server::parse_client_args(std::span{argv});
  EXPECT_TRUE(cfg.valid);
  ASSERT_EQ(cfg.inputs.size(), 1U);
  EXPECT_EQ(cfg.inputs[0].shape, (std::vector<int64_t>{1, 3, 224, 224}));
  EXPECT_EQ(cfg.inputs[0].type, at::kFloat);
  EXPECT_EQ(cfg.server_address, "localhost:1234");
  EXPECT_EQ(cfg.model_name, "my_model");
  EXPECT_EQ(cfg.model_version, "1");
  EXPECT_EQ(cfg.request_nb, 5);
  EXPECT_EQ(cfg.delay_us, 10);
  EXPECT_EQ(cfg.verbosity, starpu_server::VerbosityLevel::Stats);
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

TEST(ClientArgs, RejectsUnknownArguments)
{
  auto argv = std::to_array<const char*>({"prog", "--bogus"});
  testing::internal::CaptureStderr();
  auto cfg = starpu_server::parse_client_args(std::span{argv});
  const std::string err = testing::internal::GetCapturedStderr();
  EXPECT_FALSE(cfg.valid);
  EXPECT_NE(err.find("Unknown argument"), std::string::npos);
}

TEST(ClientArgs, RejectsMalformedInputFormat)
{
  auto argv = std::to_array<const char*>({"prog", "--input", "bad"});
  testing::internal::CaptureStderr();
  auto cfg = starpu_server::parse_client_args(std::span{argv});
  const std::string err = testing::internal::GetCapturedStderr();
  EXPECT_FALSE(cfg.valid);
  EXPECT_NE(err.find("Input must be NAME:SHAPE:TYPE"), std::string::npos);
}

TEST(ClientArgs, ParsesClientModelPathWhenProvided)
{
  auto argv = std::to_array<const char*>(
      {"prog", "--input", "input:1:float32", "--client-model",
       "/tmp/model.pt"});
  auto cfg = starpu_server::parse_client_args(std::span{argv});
  ASSERT_TRUE(cfg.valid);
  EXPECT_EQ(cfg.client_model_path, "/tmp/model.pt");
}

TEST(ClientArgs, MissingClientModelPathValueFailsParsing)
{
  auto argv = std::to_array<const char*>(
      {"prog", "--input", "input:1:float32", "--client-model"});
  auto cfg = starpu_server::parse_client_args(std::span{argv});
  EXPECT_FALSE(cfg.valid);
  EXPECT_TRUE(cfg.client_model_path.empty());
}

TEST(ClientArgs, RejectsNegativeDelay)
{
  auto argv = std::to_array<const char*>(
      {"prog", "--input", "input:1x3:float32", "--delay", "-1"});
  testing::internal::CaptureStderr();
  auto cfg = starpu_server::parse_client_args(std::span{argv});
  const std::string err = testing::internal::GetCapturedStderr();
  EXPECT_FALSE(cfg.valid);
  EXPECT_NE(err.find("Must be >= 0."), std::string::npos);
}

TEST(ClientArgs, VerboseLevels)
{
  using enum starpu_server::VerbosityLevel;
  const auto cases =
      std::to_array<std::pair<const char*, starpu_server::VerbosityLevel>>(
          {{"0", Silent}, {"1", Info}, {"3", Debug}, {"4", Trace}});
  for (const auto& [level_str, expected] : cases) {
    auto argv = std::to_array<const char*>(
        {"prog", "--input", "input:1:float32", "--verbose", level_str});
    auto cfg = starpu_server::parse_client_args(std::span{argv});
    ASSERT_TRUE(cfg.valid);
    EXPECT_EQ(cfg.verbosity, expected);
  }
}

TEST(ClientArgs, RejectsNonPositiveShapeDims)
{
  auto argv_neg =
      std::to_array<const char*>({"prog", "--input", "input:1x-3x224:float32"});
  auto cfg_neg = starpu_server::parse_client_args(std::span{argv_neg});
  EXPECT_FALSE(cfg_neg.valid);

  auto argv_zero =
      std::to_array<const char*>({"prog", "--input", "input:1x0x224:float32"});
  auto cfg_zero = starpu_server::parse_client_args(std::span{argv_zero});
  EXPECT_FALSE(cfg_zero.valid);
}

TEST(ClientArgs, RejectsMalformedShapeTokens)
{
  const auto cases =
      std::to_array<const char*>({"1xax2", "9223372036854775808", ""});
  for (const auto* shape : cases) {
    const std::string spec = std::string{"input:"} + shape + ":float32";
    auto argv = std::to_array<const char*>({"prog", "--input", spec.c_str()});
    auto cfg = starpu_server::parse_client_args(std::span{argv});
    EXPECT_FALSE(cfg.valid);
  }
}

TEST(ClientArgsHelp, ContainsKeyOptions)
{
  testing::internal::CaptureStdout();
  starpu_server::display_client_help("prog");
  const std::string out = testing::internal::GetCapturedStdout();
  EXPECT_NE(out.find("--request-number"), std::string::npos);
  EXPECT_NE(out.find("--delay"), std::string::npos);
  EXPECT_NE(out.find("--input"), std::string::npos);
  EXPECT_NE(out.find("--server"), std::string::npos);
  EXPECT_NE(out.find("--model"), std::string::npos);
  EXPECT_NE(out.find("--client-model"), std::string::npos);
  EXPECT_NE(out.find("--verbose"), std::string::npos);
  EXPECT_NE(out.find("--help"), std::string::npos);
  EXPECT_EQ(out.find("--shape"), std::string::npos);
  EXPECT_EQ(out.find("--type"), std::string::npos);
  EXPECT_EQ(out.find("--version"), std::string::npos);
}

TEST(InferenceClientDetermineInferenceCount, HandlesEdgeCases)
{
  const auto determine = [](const starpu_server::ClientConfig& cfg) {
    return starpu_server::InferenceClient::determine_inference_count(cfg);
  };

  starpu_server::ClientConfig cfg;
  EXPECT_EQ(determine(cfg), 1U);

  cfg.inputs.clear();
  {
    starpu_server::InputConfig input;
    input.name = "zero_batch";
    input.shape = {0, 3, 224};
    cfg.inputs.push_back(input);
  }
  {
    starpu_server::InputConfig input;
    input.name = "negative_batch";
    input.shape = {-4, 3, 224};
    cfg.inputs.push_back(input);
  }
  {
    starpu_server::InputConfig input;
    input.name = "valid_batch";
    input.shape = {5, 3, 224};
    cfg.inputs.push_back(input);
  }
  EXPECT_EQ(determine(cfg), 5U);

  cfg.inputs.clear();
  {
    starpu_server::InputConfig input;
    input.name = "first_valid";
    input.shape = {8, 3};
    cfg.inputs.push_back(input);
  }
  {
    starpu_server::InputConfig input;
    input.name = "conflicting";
    input.shape = {3, 3};
    cfg.inputs.push_back(input);
  }
  EXPECT_EQ(determine(cfg), 8U);

  cfg.inputs.clear();
  {
    starpu_server::InputConfig input;
    input.name = "empty_shape";
    cfg.inputs.push_back(input);
  }
  {
    starpu_server::InputConfig input;
    input.name = "zero_dim";
    input.shape = {0};
    cfg.inputs.push_back(input);
  }
  {
    starpu_server::InputConfig input;
    input.name = "negative_dim";
    input.shape = {-1, 2};
    cfg.inputs.push_back(input);
  }
  EXPECT_EQ(determine(cfg), 1U);
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

  starpu_server::StopServer(server.server.get());
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

TEST(InferenceClientLatencySummary, SkipsEmptyMetric)
{
  auto channel = grpc::CreateChannel(
      "localhost:59998", grpc::InsecureChannelCredentials());
  starpu_server::InferenceClient client(
      channel, starpu_server::VerbosityLevel::Silent);

  client.verbosity_ = starpu_server::VerbosityLevel::Info;
  client.latency_records_.roundtrip_ms.push_back(1.23);
  client.latency_records_.server_queue_ms.push_back(0.45);

  testing::internal::CaptureStdout();
  client.log_latency_summary();
  const std::string output = testing::internal::GetCapturedStdout();

  EXPECT_NE(output.find("latency"), std::string::npos);
  EXPECT_NE(output.find("queue"), std::string::npos);
  EXPECT_EQ(output.find("client_overhead"), std::string::npos);
}

TEST(InferenceClientLatencySummary, HandlesZeroElapsedTime)
{
  auto channel = grpc::CreateChannel(
      "localhost:59995", grpc::InsecureChannelCredentials());
  starpu_server::InferenceClient client(
      channel, starpu_server::VerbosityLevel::Stats);

  const auto now = std::chrono::high_resolution_clock::now();
  client.first_request_time_ = now;
  client.last_response_time_ = now;
  client.total_inference_count_ = 3;
  const starpu_server::InferenceClient::LatencySample sample{
      1.0, 0.9, 0.8, 0.7,  0.65, 0.6,  0.5, 0.4,
      0.3, 0.2, 0.1, 0.05, 0.04, 0.03, 0.02};
  client.record_latency(sample);

  starpu_server::CaptureStream capture{std::cout};
  client.log_latency_summary();
  const std::string output = capture.str();

  EXPECT_NE(
      output.find(
          "throughput: 3 inferences (elapsed time too small to compute rate)"),
      std::string::npos);
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

TEST(InferenceClient, RejectsUnsupportedTensorType)
{
  auto channel =
      grpc::CreateChannel("localhost:0", grpc::InsecureChannelCredentials());
  starpu_server::InferenceClient client(
      channel, starpu_server::VerbosityLevel::Silent);

  starpu_server::ClientConfig cfg;
  cfg.model_name = "example";
  cfg.model_version = "1";
  cfg.inputs.push_back({"input0", {1}, at::kFloat});

  std::vector<torch::Tensor> tensors = {
      torch::zeros({1}, torch::dtype(at::kDouble))};

  EXPECT_THROW(
      {
        try {
          client.AsyncModelInfer(tensors, cfg);
          client.Shutdown();
        }
        catch (...) {
          client.Shutdown();
          throw;
        }
      },
      std::invalid_argument);
}

TEST(InferenceClient, ConvertsNonContiguousCpuTensor)
{
  auto channel =
      grpc::CreateChannel("localhost:0", grpc::InsecureChannelCredentials());
  starpu_server::InferenceClient client(
      channel, starpu_server::VerbosityLevel::Info);

  starpu_server::ClientConfig cfg;
  cfg.model_name = "example";
  cfg.model_version = "1";
  cfg.inputs.push_back({"input0", {2, 2}, at::kFloat});

  auto tensor = torch::arange(4, torch::kFloat).view({2, 2}).transpose(0, 1);

  testing::internal::CaptureStdout();
  try {
    client.AsyncModelInfer({tensor}, cfg);
    client.Shutdown();
  }
  catch (...) {
    client.Shutdown();
    testing::internal::GetCapturedStdout();
    throw;
  }
  const std::string output = testing::internal::GetCapturedStdout();
  EXPECT_NE(
      output.find("not on CPU or non-contiguous, converting"),
      std::string::npos);
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
