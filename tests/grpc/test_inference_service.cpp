#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstring>
#include <memory>
#include <vector>

#include "grpc/server/inference_service.hpp"
#include "test_helpers.hpp"

inline void
expect_empty_infer_response(const inference::ModelInferResponse& resp)
{
  EXPECT_EQ(resp.model_name(), "");
  EXPECT_EQ(resp.model_version(), "");
  EXPECT_EQ(resp.outputs_size(), 0);
  EXPECT_EQ(resp.raw_output_contents_size(), 0);
  EXPECT_EQ(resp.server_receive_ms(), 0);
  EXPECT_EQ(resp.server_send_ms(), 0);
}

class InferenceServiceTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    service = std::make_unique<starpu_server::InferenceServiceImpl>(
        &queue, &ref_outputs);
  }
  auto prepare_job(
      std::vector<torch::Tensor> ref_outs,
      std::vector<torch::Tensor> worker_outs = {}) -> std::jthread
  {
    ref_outputs = std::move(ref_outs);
    return starpu_server::run_single_job(queue, std::move(worker_outs));
  }
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  std::unique_ptr<starpu_server::InferenceServiceImpl> service;
  grpc::ServerContext ctx;
  inference::ModelInferResponse reply;
};

struct SubmitJobAndWaitCase {
  std::vector<torch::Tensor> ref_outputs;
  std::vector<torch::Tensor> worker_outputs;
  grpc::StatusCode expected_status;
};

class SubmitJobAndWaitTest
    : public InferenceServiceTest,
      public ::testing::WithParamInterface<SubmitJobAndWaitCase> {};

TEST(InferenceService, ValidateInputsSuccess)
{
  auto req = starpu_server::make_valid_request();
  std::vector<torch::Tensor> inputs;
  auto status =
      starpu_server::InferenceServiceImpl::validate_and_convert_inputs(
          &req, inputs);
  ASSERT_TRUE(status.ok());
  ASSERT_EQ(inputs.size(), 1u);
  EXPECT_EQ(inputs[0].sizes(), (torch::IntArrayRef{2, 2}));
  EXPECT_EQ(inputs[0].scalar_type(), at::kFloat);
  EXPECT_FLOAT_EQ(inputs[0][0][0].item<float>(), 1.0f);
}

TEST(InferenceService, ValidateInputsMultipleDtypes)
{
  std::vector<float> data0 = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int64_t> data1 = {10, 20, 30};
  auto req = starpu_server::make_model_infer_request({
      {{2, 2}, at::kFloat, starpu_server::to_raw_data(data0)},
      {{3}, at::kLong, starpu_server::to_raw_data(data1)},
  });
  std::vector<torch::Tensor> inputs;
  auto status =
      starpu_server::InferenceServiceImpl::validate_and_convert_inputs(
          &req, inputs);
  ASSERT_TRUE(status.ok());
  ASSERT_EQ(inputs.size(), 2u);
  EXPECT_EQ(inputs[0].sizes(), (torch::IntArrayRef{2, 2}));
  EXPECT_EQ(inputs[0].scalar_type(), at::kFloat);
  EXPECT_FLOAT_EQ(inputs[0][0][0].item<float>(), 1.0f);
  EXPECT_EQ(inputs[1].sizes(), (torch::IntArrayRef{3}));
  EXPECT_EQ(inputs[1].scalar_type(), at::kLong);
  EXPECT_EQ(inputs[1][0].item<int64_t>(), 10);
}

TEST_F(InferenceServiceTest, ModelInferReturnsValidationError)
{
  auto req = starpu_server::make_valid_request();
  req.add_raw_input_contents()->assign("", 0);
  req.MergeFrom(starpu_server::make_model_request("m", "1"));
  auto status = service->ModelInfer(&ctx, &req, &reply);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  expect_empty_infer_response(reply);
}

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

TEST(GrpcServer, StartAndStop)
{
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> reference_outputs;
  auto server = starpu_server::start_test_grpc_server(
      queue, reference_outputs, "127.0.0.1:0");
  starpu_server::StopServer(server.server);
  server.thread.join();
  EXPECT_EQ(server.server, nullptr);
}

TEST(GrpcServer, StopServerNullptr)
{
  std::unique_ptr<grpc::Server> server;
  EXPECT_NO_THROW(starpu_server::StopServer(server));
  EXPECT_EQ(server, nullptr);
}

TEST(InferenceServiceImpl, PopulateResponsePopulatesFieldsAndTimes)
{
  auto req = starpu_server::make_model_request("model", "1");
  std::vector<torch::Tensor> outputs = {
      torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kInt))};
  inference::ModelInferResponse reply;
  int64_t recv_ms = 10;
  int64_t send_ms = 20;
  starpu_server::InferenceServiceImpl::populate_response(
      &req, &reply, outputs, recv_ms, send_ms);
  starpu_server::verify_populate_response(
      req, reply, outputs, recv_ms, send_ms);
}

struct InvalidRequestCase {
  const char* name;
  inference::ModelInferRequest request;
};

class InferenceServiceValidation
    : public ::testing::TestWithParam<InvalidRequestCase> {};

TEST_P(InferenceServiceValidation, InvalidRequests)
{
  auto req = GetParam().request;
  std::vector<torch::Tensor> inputs;
  auto status =
      starpu_server::InferenceServiceImpl::validate_and_convert_inputs(
          &req, inputs);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

INSTANTIATE_TEST_SUITE_P(
    , InferenceServiceValidation,
    ::testing::Values(
        InvalidRequestCase{
            "RawInputCountMismatch",
            [] {
              std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
              auto req = starpu_server::make_model_infer_request(
                  {{{2, 2}, at::kFloat, starpu_server::to_raw_data(data)}});
              req.add_raw_input_contents()->assign("", 0);
              return req;
            }()},
        InvalidRequestCase{
            "RawContentSizeMismatch",
            [] {
              std::vector<float> data = {1.0f, 2.0f, 3.0f};
              return starpu_server::make_model_infer_request(
                  {{{2, 2}, at::kFloat, starpu_server::to_raw_data(data)}});
            }()}),
    [](const testing::TestParamInfo<InvalidRequestCase>& info) {
      return info.param.name;
    });