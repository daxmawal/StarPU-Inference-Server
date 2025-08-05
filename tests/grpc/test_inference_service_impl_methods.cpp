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
