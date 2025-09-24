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

  std::vector<torch::Tensor> expected_outputs = {
      torch::tensor({10.0f, 20.0f, 30.0f, 40.0f}).view({2, 2})};
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
  starpu_server::InferenceServiceImpl::LatencyBreakdown response_breakdown;
  response_breakdown.preprocess_ms = response.server_preprocess_ms();
  response_breakdown.queue_ms = response.server_queue_ms();
  response_breakdown.submit_ms = response.server_submit_ms();
  response_breakdown.scheduling_ms = response.server_scheduling_ms();
  response_breakdown.codelet_ms = response.server_codelet_ms();
  response_breakdown.inference_ms = response.server_inference_ms();
  response_breakdown.callback_ms = response.server_callback_ms();
  response_breakdown.postprocess_ms = response.server_postprocess_ms();
  response_breakdown.total_ms = response.server_total_ms();
  response_breakdown.overall_ms = response.server_overall_ms();
  starpu_server::verify_populate_response(
      request, response, expected_outputs, response.server_receive_ms(),
      response.server_send_ms(), response_breakdown);
  EXPECT_GE(response.server_total_ms(), 0.0);

  starpu_server::StopServer(server.server);
  server.thread.join();
  EXPECT_EQ(server.server, nullptr);
}
