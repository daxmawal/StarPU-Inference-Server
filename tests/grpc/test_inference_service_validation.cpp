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
