#include <gtest/gtest.h>

#include <cstring>

#include "grpc/server/inference_service.hpp"

using namespace starpu_server;

static inference::ModelInferRequest
make_valid_request()
{
  inference::ModelInferRequest req;
  auto* input = req.add_inputs();
  input->set_name("input0");
  input->set_datatype("FP32");
  input->add_shape(2);
  input->add_shape(2);

  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  req.add_raw_input_contents()->assign(
      reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
  return req;
}

TEST(InferenceService, ValidateInputsSuccess)
{
  auto req = make_valid_request();
  std::vector<torch::Tensor> inputs;
  auto status = InferenceServiceImpl::validate_and_convert_inputs(&req, inputs);
  ASSERT_TRUE(status.ok());
  ASSERT_EQ(inputs.size(), 1u);
  EXPECT_EQ(inputs[0].sizes(), (torch::IntArrayRef{2, 2}));
  EXPECT_EQ(inputs[0].scalar_type(), at::kFloat);
  EXPECT_FLOAT_EQ(inputs[0][0][0].item<float>(), 1.0f);
}

TEST(InferenceService, RawInputCountMismatch)
{
  auto req = make_valid_request();
  // add an extra raw input to cause mismatch
  req.add_raw_input_contents()->assign("", 0);
  std::vector<torch::Tensor> inputs;
  auto status = InferenceServiceImpl::validate_and_convert_inputs(&req, inputs);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(InferenceService, RawContentSizeMismatch)
{
  inference::ModelInferRequest req;
  auto* input = req.add_inputs();
  input->set_name("input0");
  input->set_datatype("FP32");
  input->add_shape(2);
  input->add_shape(2);
  // only 3 floats provided but shape expects 4
  std::vector<float> data = {1.0f, 2.0f, 3.0f};
  req.add_raw_input_contents()->assign(
      reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
  std::vector<torch::Tensor> inputs;
  auto status = InferenceServiceImpl::validate_and_convert_inputs(&req, inputs);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(InferenceService, PopulateResponseFillsFields)
{
  inference::ModelInferRequest req;
  req.set_model_name("model");
  req.set_model_version("1");

  std::vector<torch::Tensor> outputs = {
      torch::tensor({1, 2, 3}, torch::TensorOptions().dtype(at::kInt))};
  inference::ModelInferResponse reply;
  int64_t recv_ms = 10;
  int64_t send_ms = 20;

  InferenceServiceImpl::populate_response(
      &req, &reply, outputs, recv_ms, send_ms);

  EXPECT_EQ(reply.model_name(), "model");
  EXPECT_EQ(reply.model_version(), "1");
  EXPECT_EQ(reply.server_receive_ms(), recv_ms);
  EXPECT_EQ(reply.server_send_ms(), send_ms);

  ASSERT_EQ(reply.outputs_size(), 1);
  ASSERT_EQ(reply.raw_output_contents_size(), 1);

  const auto& out = reply.outputs(0);
  EXPECT_EQ(out.name(), "output0");
  EXPECT_EQ(out.datatype(), "INT32");
  ASSERT_EQ(out.shape_size(), 1);
  EXPECT_EQ(out.shape(0), 3);

  auto flat = outputs[0].view({-1});
  const auto& raw = reply.raw_output_contents(0);
  ASSERT_EQ(raw.size(), flat.numel() * flat.element_size());
  EXPECT_EQ(0, std::memcmp(raw.data(), flat.data_ptr(), raw.size()));
}