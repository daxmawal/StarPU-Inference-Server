#include "test_inference_service.hpp"

TEST(InferenceService, ValidateInputsSuccess)
{
  auto req = starpu_server::make_valid_request();
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  std::vector<at::ScalarType> expected_types = {at::kFloat};
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, expected_types);
  std::vector<torch::Tensor> inputs;
  auto status = service.validate_and_convert_inputs(&req, inputs);
  ASSERT_TRUE(status.ok());
  ASSERT_EQ(inputs.size(), 1U);
  EXPECT_EQ(inputs[0].sizes(), (torch::IntArrayRef{2, 2}));
  EXPECT_EQ(inputs[0].scalar_type(), at::kFloat);
  EXPECT_FLOAT_EQ(inputs[0][0][0].item<float>(), 1.0F);
}

TEST(InferenceService, ValidateInputsMultipleDtypes)
{
  std::vector<float> data0 = {1.0F, 2.0F, 3.0F, 4.0F};
  std::vector<int64_t> data1 = {10, 20, 30};
  auto req = starpu_server::make_model_infer_request({
      {{2, 2}, at::kFloat, starpu_server::to_raw_data(data0)},
      {{3}, at::kLong, starpu_server::to_raw_data(data1)},
  });
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  std::vector<at::ScalarType> expected_types = {at::kFloat, at::kLong};
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, expected_types);
  std::vector<torch::Tensor> inputs;
  auto status = service.validate_and_convert_inputs(&req, inputs);
  ASSERT_TRUE(status.ok());
  ASSERT_EQ(inputs.size(), 2U);
  EXPECT_EQ(inputs[0].sizes(), (torch::IntArrayRef{2, 2}));
  EXPECT_EQ(inputs[0].scalar_type(), at::kFloat);
  EXPECT_FLOAT_EQ(inputs[0][0][0].item<float>(), 1.0F);
  EXPECT_EQ(inputs[1].sizes(), (torch::IntArrayRef{3}));
  EXPECT_EQ(inputs[1].scalar_type(), at::kLong);
  EXPECT_EQ(inputs[1][0].item<int64_t>(), 10);
}

TEST(InferenceService, ValidateInputsMismatchedCount)
{
  std::vector<float> data0 = {1.0F, 2.0F, 3.0F, 4.0F};
  std::vector<int64_t> data1 = {10, 20, 30};
  auto req = starpu_server::make_model_infer_request({
      {{2, 2}, at::kFloat, starpu_server::to_raw_data(data0)},
      {{3}, at::kLong, starpu_server::to_raw_data(data1)},
  });
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  std::vector<at::ScalarType> expected_types = {at::kFloat};
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, expected_types);
  std::vector<torch::Tensor> inputs;
  auto status = service.validate_and_convert_inputs(&req, inputs);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
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
