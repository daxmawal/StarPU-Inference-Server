#include <gtest/gtest.h>

#include "../test_helpers.hpp"
#include "inference_service_test.hpp"

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

TEST_F(InferenceServiceTest, SubmitJobAndWaitReturnsOutputs)
{
  std::vector<torch::Tensor> inputs = {torch::tensor({1})};
  std::vector<torch::Tensor> expected = {torch::tensor({42})};
  std::vector<torch::Tensor> outputs;
  auto worker = prepare_job({torch::zeros({1})}, expected);
  auto status = service->submit_job_and_wait(inputs, outputs);
  ASSERT_TRUE(status.ok());
  ASSERT_EQ(outputs.size(), expected.size());
  EXPECT_TRUE(torch::equal(outputs[0], expected[0]));
}

TEST(InferenceServiceImpl, PopulateResponseFillsFields)
{
  auto req = starpu_server::make_model_request("mymodel", "1");
  std::vector<torch::Tensor> outs = {torch::tensor({3.0f, 4.0f})};
  inference::ModelInferResponse resp;
  int64_t recv_ms = 10;
  int64_t send_ms = 20;
  starpu_server::InferenceServiceImpl::populate_response(
      &req, &resp, outs, recv_ms, send_ms);
  starpu_server::verify_populate_response(req, resp, outs, recv_ms, send_ms);
}
