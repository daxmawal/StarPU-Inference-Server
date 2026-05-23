#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <future>

#include "grpc/client/inference_client.hpp"
#include "grpc/server/inference_service.hpp"
#include "support/grpc/client/inference_client_test_api.hpp"
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
  cfg.model_name = "model";
  cfg.model_version = "1";
  torch::Tensor tensor = torch::zeros(
      cfg.inputs[0].shape, torch::TensorOptions().dtype(cfg.inputs[0].type));

  ::testing::internal::CaptureStderr();
  client.AsyncModelInfer({tensor}, cfg);
  client.Shutdown();
  cq_thread.join();
  const std::string err = ::testing::internal::GetCapturedStderr();

  EXPECT_NE(err.find("Connection refused"), std::string::npos);
  SUCCEED();
}

TEST(InferenceClient, ServerIsLiveReturnsTrue)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> reference_outputs;
  auto server = starpu_server::start_test_grpc_server(
      queue, reference_outputs, {at::kFloat}, 0,
      starpu_server::VerbosityLevel::Silent);

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
  auto server = starpu_server::start_test_grpc_server(
      queue, reference_outputs, {at::kFloat}, 0,
      starpu_server::VerbosityLevel::Silent);

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

TEST(InferenceClient, DISABLED_AsyncCompleteRpcSuccess)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> reference_outputs = {torch::zeros({2, 2})};
  auto server = starpu_server::start_test_grpc_server(
      queue, reference_outputs, {at::kFloat}, 0,
      starpu_server::VerbosityLevel::Silent);

  constexpr float kVal1 = 10.0F;
  constexpr float kVal2 = 20.0F;
  constexpr float kVal3 = 30.0F;
  constexpr float kVal4 = 40.0F;
  std::vector<torch::Tensor> worker_outputs = {
      torch::tensor({kVal1, kVal2, kVal3, kVal4}).view({2, 2})};
  std::promise<void> response_sent_promise;
  auto response_sent = response_sent_promise.get_future();
  auto worker = starpu_server::run_single_job(
      queue, worker_outputs, 0.0,
      [&response_sent_promise](starpu_server::InferenceJob& job) {
        auto on_complete = job.completion().get_on_complete();
        job.completion().set_on_complete(
            [on_complete = std::move(on_complete),
             promise = &response_sent_promise](
                const std::vector<torch::Tensor>& outputs,
                double latency) mutable {
              on_complete(outputs, latency);
              promise->set_value();
            });
      });

  auto channel = grpc::CreateChannel(
      "127.0.0.1:" + std::to_string(server.port),
      grpc::InsecureChannelCredentials());
  starpu_server::InferenceClient client(
      channel, starpu_server::VerbosityLevel::Stats);
  ASSERT_TRUE(client.ServerIsReady());

  testing::internal::CaptureStdout();
  std::jthread cq_thread(
      &starpu_server::InferenceClient::AsyncCompleteRpc, &client);

  starpu_server::ClientConfig cfg;
  cfg.inputs = {{"input0", {2, 2}, at::kFloat}};
  cfg.model_name = "model";
  cfg.model_version = "1";
  torch::Tensor tensor = torch::zeros(
      cfg.inputs[0].shape, torch::TensorOptions().dtype(cfg.inputs[0].type));

  client.AsyncModelInfer({tensor}, cfg);

  const auto response_status = response_sent.wait_for(std::chrono::seconds(1));
  if (response_status != std::future_status::ready) {
    queue.shutdown();
    client.Shutdown();
    cq_thread.join();
    worker.join();
    starpu_server::StopServer(server.server.get());
    server.thread.join();
    FAIL() << "Timed out waiting for the server to produce a response";
  }

  worker.join();

  const auto completion_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(1);
  while (starpu_server::InferenceClientTestAccess::handled_requests(client) ==
             0U &&
         std::chrono::steady_clock::now() < completion_deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  if (starpu_server::InferenceClientTestAccess::handled_requests(client) ==
      0U) {
    queue.shutdown();
    client.Shutdown();
    cq_thread.join();
    starpu_server::StopServer(server.server.get());
    server.thread.join();
    FAIL() << "Timed out waiting for the client to observe the async response";
  }

  client.Shutdown();
  cq_thread.join();
  auto logs = testing::internal::GetCapturedStdout();
  EXPECT_NE(logs.find("Request ID 0"), std::string::npos);
  EXPECT_NE(logs.find("latency"), std::string::npos);
  EXPECT_NE(logs.find("request_latency"), std::string::npos);
  EXPECT_NE(logs.find("response_latency"), std::string::npos);

  queue.shutdown();
  starpu_server::StopServer(server.server.get());
  server.thread.join();
  EXPECT_EQ(server.server, nullptr);
}
