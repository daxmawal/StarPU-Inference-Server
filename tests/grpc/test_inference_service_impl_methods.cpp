#include <gtest/gtest.h>

#include <thread>

#include "inference_service_test.hpp"

TEST_F(InferenceServiceTest, BasicLivenessAndReadiness)
{
  grpc::ServerContext ctx;
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
  ref_outputs = {torch::zeros({1})};

  std::vector<torch::Tensor> inputs = {torch::tensor({1})};
  std::vector<torch::Tensor> expected = {torch::tensor({42})};

  std::vector<torch::Tensor> outputs;
  std::thread worker([&] {
    std::shared_ptr<starpu_server::InferenceJob> job;
    queue.wait_and_pop(job);
    job->get_on_complete()(expected, 0.0);
  });

  auto status = service->submit_job_and_wait(inputs, outputs);
  worker.join();

  ASSERT_TRUE(status.ok());
  ASSERT_EQ(outputs.size(), expected.size());
  EXPECT_TRUE(torch::equal(outputs[0], expected[0]));
}

TEST(InferenceServiceImpl, PopulateResponseFillsFields)
{
  inference::ModelInferRequest req;
  req.set_model_name("mymodel");
  req.set_model_version("1");

  std::vector<torch::Tensor> outs = {torch::tensor({3.0f, 4.0f})};
  inference::ModelInferResponse resp;
  starpu_server::InferenceServiceImpl::populate_response(
      &req, &resp, outs, 10, 20);

  EXPECT_EQ(resp.model_name(), "mymodel");
  EXPECT_EQ(resp.model_version(), "1");
  EXPECT_EQ(resp.server_receive_ms(), 10);
  EXPECT_EQ(resp.server_send_ms(), 20);
  ASSERT_EQ(resp.outputs_size(), 1);
  const auto& out_tensor = resp.outputs(0);
  EXPECT_EQ(out_tensor.name(), "output0");
  EXPECT_EQ(out_tensor.datatype(), "FP32");
  ASSERT_EQ(out_tensor.shape_size(), 1);
  EXPECT_EQ(out_tensor.shape(0), 2);
  ASSERT_EQ(resp.raw_output_contents_size(), 1);
  auto raw = resp.raw_output_contents(0);
  const float* data = reinterpret_cast<const float*>(raw.data());
  EXPECT_FLOAT_EQ(data[0], 3.0f);
  EXPECT_FLOAT_EQ(data[1], 4.0f);
}
