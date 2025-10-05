#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <memory>
#include <thread>

#include "grpc/server/inference_service.hpp"
#include "test_helpers.hpp"

TEST(GrpcClientServer, EndToEndInference)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> reference_outputs = {torch::zeros({2, 2})};

  auto server = starpu_server::start_test_grpc_server(queue, reference_outputs);

  constexpr float kVal1 = 10.0F;
  constexpr float kVal2 = 20.0F;
  constexpr float kVal3 = 30.0F;
  constexpr float kVal4 = 40.0F;
  std::vector<torch::Tensor> expected_outputs = {
      torch::tensor({kVal1, kVal2, kVal3, kVal4}).view({2, 2})};
  auto worker = starpu_server::run_single_job(queue, expected_outputs);

  auto channel = grpc::CreateChannel(
      "127.0.0.1:" + std::to_string(server.port),
      grpc::InsecureChannelCredentials());

  auto request = starpu_server::make_valid_request();
  request.MergeFrom(starpu_server::make_model_request("model", "1"));

  auto stub = inference::GRPCInferenceService::NewStub(channel);
  inference::ModelInferResponse response;
  grpc::ClientContext context;
  auto status = stub->ModelInfer(&context, request, &response);
  ASSERT_TRUE(status.ok());

  EXPECT_GT(response.server_receive_ms(), 0);
  EXPECT_GT(response.server_send_ms(), 0);
  auto response_breakdown = starpu_server::make_latency_breakdown(response);
  starpu_server::verify_populate_response(
      request, response, expected_outputs, response.server_receive_ms(),
      response.server_send_ms(), response_breakdown);
  EXPECT_GE(response.server_total_ms(), 0.0);

  starpu_server::StopServer(server.server);
  server.thread.join();
  EXPECT_EQ(server.server, nullptr);
}
