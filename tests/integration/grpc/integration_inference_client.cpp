#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "grpc/client/inference_client.hpp"
#include "grpc/server/inference_service.hpp"
#include "test_helpers.hpp"

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

  starpu_server::StopServer(server.server.get());
  server.thread.join();
  EXPECT_EQ(server.server, nullptr);
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

  starpu_server::StopServer(server.server.get());
  server.thread.join();
  EXPECT_EQ(server.server, nullptr);
}

TEST(InferenceClient, AsyncCompleteRpcSuccess)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> reference_outputs = {torch::zeros({1})};
  auto server = starpu_server::start_test_grpc_server(queue, reference_outputs);

  std::vector<torch::Tensor> worker_outputs = {torch::tensor({1.0F})};
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
  EXPECT_NE(logs.find("request_latency"), std::string::npos);
  EXPECT_NE(logs.find("response_latency"), std::string::npos);

  starpu_server::StopServer(server.server.get());
  server.thread.join();
  EXPECT_EQ(server.server, nullptr);
}
