#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <memory>
#include <thread>

#include "../test_helpers.hpp"
#include "grpc/client/inference_client.hpp"
#include "grpc/server/inference_service.hpp"

TEST(GrpcClientServer, EndToEndInference)
{
  starpu_server::InferenceQueue queue;
  // Minimal reference outputs to allocate server-side buffers
  std::vector<torch::Tensor> reference_outputs = {torch::zeros({2, 2})};

  std::unique_ptr<grpc::Server> server;
  std::thread server_thread([&]() {
    starpu_server::RunGrpcServer(
        queue, reference_outputs, "127.0.0.1:50051", 1 << 20, server);
  });
  while (!server) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  // Worker thread producing expected outputs
  std::vector<torch::Tensor> expected_outputs = {
      torch::tensor({10.0f, 20.0f, 30.0f, 40.0f}).view({2, 2})};
  auto worker = starpu_server::run_single_job(queue, expected_outputs);

  auto channel = grpc::CreateChannel(
      "127.0.0.1:50051", grpc::InsecureChannelCredentials());
  // Instantiate client to connect to the server
  starpu_server::InferenceClient client(
      channel, starpu_server::VerbosityLevel::Silent);

  auto request = starpu_server::make_valid_request();
  request.MergeFrom(starpu_server::make_model_request("model", "1"));

  auto stub = inference::GRPCInferenceService::NewStub(channel);
  inference::ModelInferResponse response;
  grpc::ClientContext context;
  auto status = stub->ModelInfer(&context, request, &response);
  ASSERT_TRUE(status.ok());

  EXPECT_GT(response.server_receive_ms(), 0);
  EXPECT_GT(response.server_send_ms(), 0);
  starpu_server::verify_populate_response(
      request, response, expected_outputs, response.server_receive_ms(),
      response.server_send_ms());

  starpu_server::StopServer(server);
  server_thread.join();
  EXPECT_EQ(server, nullptr);
}
