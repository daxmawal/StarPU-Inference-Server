#include "test_inference_service.hpp"

TEST_F(InferenceServiceTest, ModelInferPropagatesSubmitError)
{
  auto req = starpu_server::make_valid_request();
  req.MergeFrom(starpu_server::make_model_request("m", "1"));
  auto worker = prepare_job({torch::zeros({1})});
  auto status = service->ModelInfer(&ctx, &req, &reply);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INTERNAL);
  expect_empty_infer_response(reply);
}

TEST_F(InferenceServiceTest, ModelInferReturnsOutputs)
{
  auto req = starpu_server::make_valid_request();
  req.MergeFrom(starpu_server::make_model_request("m", "1"));
  std::vector<torch::Tensor> outs = {
      torch::tensor({10.0f, 20.0f, 30.0f, 40.0f}).view({2, 2})};
  auto worker = prepare_job({torch::zeros({2, 2})}, outs);
  auto status = service->ModelInfer(&ctx, &req, &reply);
  ASSERT_TRUE(status.ok());
  EXPECT_GT(reply.server_receive_ms(), 0);
  EXPECT_GT(reply.server_send_ms(), 0);
  starpu_server::verify_populate_response(
      req, reply, outs, reply.server_receive_ms(), reply.server_send_ms());
}

TEST_F(InferenceServiceTest, ModelInferDetectsInputSizeMismatch)
{
  auto req = starpu_server::make_valid_request();
  req.MergeFrom(starpu_server::make_model_request("m", "1"));
  req.mutable_raw_input_contents(0)->append("0", 1);
  auto status = service->ModelInfer(&ctx, &req, &reply);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  expect_empty_infer_response(reply);
}

TEST_F(InferenceServiceTest, BasicLivenessAndReadiness)
{
  inference::ServerLiveRequest live_req;
  inference::ServerLiveResponse live_resp;
  auto status = service->ServerLive(&ctx, &live_req, &live_resp);
  ASSERT_TRUE(status.ok());
  EXPECT_TRUE(live_resp.live());
  inference::ServerReadyRequest ready_req;
  inference::ServerReadyResponse ready_resp;
  status = service->ServerReady(&ctx, &ready_req, &ready_resp);
  ASSERT_TRUE(status.ok());
  EXPECT_TRUE(ready_resp.ready());
  inference::ModelReadyRequest model_req;
  inference::ModelReadyResponse model_resp;
  status = service->ModelReady(&ctx, &model_req, &model_resp);
  ASSERT_TRUE(status.ok());
  EXPECT_TRUE(model_resp.ready());
}

TEST_P(SubmitJobAndWaitTest, ReturnsExpectedStatus)
{
  std::vector<torch::Tensor> inputs = {torch::tensor({1})};
  std::vector<torch::Tensor> outputs;
  auto worker = prepare_job(GetParam().ref_outputs, GetParam().worker_outputs);
  auto status = service->submit_job_and_wait(inputs, outputs);
  EXPECT_EQ(status.error_code(), GetParam().expected_status);
  if (status.ok()) {
    ASSERT_EQ(outputs.size(), GetParam().worker_outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
      EXPECT_TRUE(torch::equal(outputs[i], GetParam().worker_outputs[i]));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    SubmitJobAndWaitScenarios, SubmitJobAndWaitTest,
    ::testing::Values(
        SubmitJobAndWaitCase{
            {torch::zeros({1})}, {}, grpc::StatusCode::INTERNAL},
        SubmitJobAndWaitCase{
            {torch::zeros({1})}, {torch::tensor({42})}, grpc::StatusCode::OK}));

TEST(GrpcServer, RunGrpcServer_StartsAndResetsServer)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> reference_outputs;
  std::unique_ptr<grpc::Server> server;
  std::jthread thread([&]() {
    starpu_server::RunGrpcServer(
        queue, reference_outputs, {at::kFloat}, "127.0.0.1:0", 32 * 1024 * 1024,
        starpu_server::VerbosityLevel::Info, server);
  });
  while (!server) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  starpu_server::StopServer(server);
  thread.join();
  EXPECT_EQ(server, nullptr);
}

TEST(GrpcServer, StartAndStop)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> reference_outputs;
  auto server = starpu_server::start_test_grpc_server(queue, reference_outputs);
  starpu_server::StopServer(server.server);
  server.thread.join();
  EXPECT_EQ(server.server, nullptr);
}
