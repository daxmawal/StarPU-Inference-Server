#include <gtest/gtest.h>

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