#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <array>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "grpc/client/client_args.hpp"
#include "grpc/client/inference_client.hpp"
#include "grpc/server/inference_service.hpp"
#include "test_helpers.hpp"

namespace starpu_server {
VerbosityLevel parse_verbosity_level(const std::string& val);
}

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
  EXPECT_EQ(cfg.inputs[0].name, "input");
  EXPECT_EQ(cfg.inputs[0].shape, (std::vector<int64_t>{1, 3, 224, 224}));
  EXPECT_EQ(cfg.inputs[0].type, at::kFloat);
  EXPECT_EQ(cfg.server_address, "localhost:1234");
  EXPECT_EQ(cfg.model_name, "my_model");
  EXPECT_EQ(cfg.model_version, "2");
  EXPECT_EQ(cfg.iterations, 5);
  EXPECT_EQ(cfg.delay_ms, 10);
  EXPECT_EQ(cfg.verbosity, starpu_server::VerbosityLevel::Stats);
}

TEST(ClientArgs, InvalidTypeIsDetected)
{
  const char* argv[] = {"prog", "--input", "foo:1:unknown"};
  auto cfg = starpu_server::parse_client_args(std::span{argv});
  EXPECT_FALSE(cfg.valid);
}

TEST(ClientArgs, ParsesMultipleInputs)
{
  const char* argv[] = {
      "prog",
      "--input",
      "input_ids:1x8:int64",
      "--input",
      "attention_mask:1x8:int64",
      "--input",
      "token_type_ids:1x8:int64"};
  auto cfg = starpu_server::parse_client_args(std::span{argv});
  ASSERT_TRUE(cfg.valid);
  ASSERT_EQ(cfg.inputs.size(), 3U);
  EXPECT_EQ(cfg.inputs[0].name, "input_ids");
  EXPECT_EQ(cfg.inputs[1].name, "attention_mask");
  EXPECT_EQ(cfg.inputs[2].name, "token_type_ids");
  EXPECT_EQ(cfg.inputs[0].shape, (std::vector<int64_t>{1, 8}));
  EXPECT_EQ(cfg.inputs[0].type, at::kLong);
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

TEST(InferenceClient, ShutdownClosesCompletionQueue)
{
  grpc::ChannelArguments ch_args;
  auto channel = grpc::CreateCustomChannel(
      "localhost:0", grpc::InsecureChannelCredentials(), ch_args);
  starpu_server::InferenceClient client(
      channel, starpu_server::VerbosityLevel::Silent);

  std::jthread cq_thread(
      &starpu_server::InferenceClient::AsyncCompleteRpc, &client);

  starpu_server::ClientConfig cfg;
  cfg.inputs = {{"input", {1}, at::kFloat}};
  torch::Tensor tensor = torch::zeros(
      cfg.inputs[0].shape, torch::TensorOptions().dtype(cfg.inputs[0].type));

  client.AsyncModelInfer({tensor}, cfg);
  client.Shutdown();

  SUCCEED();
}

TEST(InferenceClient, ServerIsLiveReturnsTrue)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> reference_outputs;
  auto server = starpu_server::start_test_grpc_server(queue, reference_outputs);

  auto channel = grpc::CreateChannel(
      "127.0.0.1:" + std::to_string(server.port),
      grpc::InsecureChannelCredentials());
  starpu_server::InferenceClient client(
      channel, starpu_server::VerbosityLevel::Silent);
  EXPECT_TRUE(client.ServerIsLive());

  starpu_server::StopServer(server.server);
  server.thread.join();
  EXPECT_EQ(server.server, nullptr);
}

TEST(InferenceClient, ServerIsLiveReturnsFalseWhenUnavailable)
{
  auto channel = grpc::CreateChannel(
      "127.0.0.1:59999", grpc::InsecureChannelCredentials());
  starpu_server::InferenceClient client(
      channel, starpu_server::VerbosityLevel::Silent);
  EXPECT_FALSE(client.ServerIsLive());
}

TEST(InferenceClient, ServerIsReadyReturnsTrue)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> reference_outputs;
  auto server = starpu_server::start_test_grpc_server(queue, reference_outputs);

  auto channel = grpc::CreateChannel(
      "127.0.0.1:" + std::to_string(server.port),
      grpc::InsecureChannelCredentials());
  starpu_server::InferenceClient client(
      channel, starpu_server::VerbosityLevel::Silent);
  EXPECT_TRUE(client.ServerIsReady());

  starpu_server::StopServer(server.server);
  server.thread.join();
  EXPECT_EQ(server.server, nullptr);
}

TEST(InferenceClient, ServerIsReadyReturnsFalseWhenUnavailable)
{
  auto channel = grpc::CreateChannel(
      "127.0.0.1:59998", grpc::InsecureChannelCredentials());
  starpu_server::InferenceClient client(
      channel, starpu_server::VerbosityLevel::Silent);
  EXPECT_FALSE(client.ServerIsReady());
}

TEST(InferenceClient, AsyncCompleteRpcSuccess)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> reference_outputs = {torch::zeros({1})};
  auto server = starpu_server::start_test_grpc_server(queue, reference_outputs);

  std::vector<torch::Tensor> worker_outputs = {torch::tensor({1.0f})};
  auto worker = starpu_server::run_single_job(queue, worker_outputs);

  auto channel = grpc::CreateChannel(
      "127.0.0.1:" + std::to_string(server.port),
      grpc::InsecureChannelCredentials());
  starpu_server::InferenceClient client(
      channel, starpu_server::VerbosityLevel::Info);

  testing::internal::CaptureStdout();
  std::jthread cq_thread(
      &starpu_server::InferenceClient::AsyncCompleteRpc, &client);

  starpu_server::ClientConfig cfg;
  cfg.inputs = {{"input", {1}, at::kFloat}};
  torch::Tensor tensor = torch::zeros(
      cfg.inputs[0].shape, torch::TensorOptions().dtype(cfg.inputs[0].type));

  client.AsyncModelInfer({tensor}, cfg);

  client.Shutdown();
  cq_thread.join();
  auto logs = testing::internal::GetCapturedStdout();
  EXPECT_NE(logs.find("Request ID 0"), std::string::npos);
  EXPECT_NE(logs.find("latency"), std::string::npos);

  starpu_server::StopServer(server.server);
  server.thread.join();
  EXPECT_EQ(server.server, nullptr);
}

TEST(InferenceClient, AsyncModelInferHandlesBertInputs)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> reference_outputs = {torch::zeros({1})};
  auto server = starpu_server::start_test_grpc_server(queue, reference_outputs);

  auto channel = grpc::CreateChannel(
      "127.0.0.1:" + std::to_string(server.port),
      grpc::InsecureChannelCredentials());
  starpu_server::InferenceClient client(
      channel, starpu_server::VerbosityLevel::Silent);

  std::jthread cq_thread(
      &starpu_server::InferenceClient::AsyncCompleteRpc, &client);

  starpu_server::ClientConfig cfg;
  cfg.inputs = {
      {"input_ids", {1, 8}, at::kLong}, {"attention_mask", {1, 8}, at::kLong}};
  std::vector<torch::Tensor> tensors = {
      torch::zeros({1, 8}, torch::TensorOptions().dtype(at::kLong)),
      torch::zeros({1, 8}, torch::TensorOptions().dtype(at::kLong))};

  client.AsyncModelInfer(tensors, cfg);

  client.Shutdown();
  cq_thread.join();

  starpu_server::StopServer(server.server);
  server.thread.join();
  EXPECT_EQ(server.server, nullptr);
}
