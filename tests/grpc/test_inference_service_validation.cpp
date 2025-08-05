#include <gtest/gtest.h>

#include "../test_helpers.hpp"
#include "inference_service_test.hpp"

TEST(InferenceServiceValidation, RawInputCountMismatch)
{
  auto req = starpu_server::make_valid_request();
  req.add_raw_input_contents()->assign("", 0);
  std::vector<torch::Tensor> inputs;
  auto status =
      starpu_server::InferenceServiceImpl::validate_and_convert_inputs(
          &req, inputs);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(InferenceServiceValidation, RawContentSizeMismatch)
{
  inference::ModelInferRequest req;
  auto* input = req.add_inputs();
  input->set_name("input0");
  input->set_datatype("FP32");
  input->add_shape(2);
  input->add_shape(2);
  std::vector<float> data = {1.0f, 2.0f, 3.0f};
  req.add_raw_input_contents()->assign(
      reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
  std::vector<torch::Tensor> inputs;
  auto status =
      starpu_server::InferenceServiceImpl::validate_and_convert_inputs(
          &req, inputs);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST_F(InferenceServiceTest, SubmitJobAndWaitInternalError)
{
  ref_outputs = {torch::zeros({1})};

  std::vector<torch::Tensor> inputs = {torch::tensor({1})};
  std::vector<torch::Tensor> outputs;

  auto worker = starpu_server::run_single_job(queue);

  auto status = service->submit_job_and_wait(inputs, outputs);

  EXPECT_EQ(status.error_code(), grpc::StatusCode::INTERNAL);
}

TEST(InferenceServiceValidation, PopulateResponseSetsServerTimes)
{
  inference::ModelInferRequest req;
  req.set_model_name("model");
  req.set_model_version("1");

  std::vector<torch::Tensor> outputs = {
      torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kInt))};
  inference::ModelInferResponse reply;
  int64_t recv_ms = 10;
  int64_t send_ms = 20;

  starpu_server::InferenceServiceImpl::populate_response(
      &req, &reply, outputs, recv_ms, send_ms);

  EXPECT_EQ(reply.server_receive_ms(), recv_ms);
  EXPECT_EQ(reply.server_send_ms(), send_ms);
}
