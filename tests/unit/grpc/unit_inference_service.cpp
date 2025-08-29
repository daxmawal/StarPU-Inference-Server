#include <limits>

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

TEST(InferenceService, ValidateInputsNonContiguous)
{
  auto base = torch::tensor({{1.0F, 2.0F}, {3.0F, 4.0F}});
  auto noncontig = base.transpose(0, 1);
  auto contig = noncontig.contiguous();
  std::vector<float> data(
      contig.data_ptr<float>(), contig.data_ptr<float>() + contig.numel());
  auto req = starpu_server::make_model_infer_request({
      {{2, 2}, at::kFloat, starpu_server::to_raw_data(data)},
  });
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  std::vector<at::ScalarType> expected_types = {at::kFloat};
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, expected_types);
  std::vector<torch::Tensor> inputs;
  auto status = service.validate_and_convert_inputs(&req, inputs);
  ASSERT_TRUE(status.ok());
  ASSERT_EQ(inputs.size(), 1U);
  EXPECT_TRUE(inputs[0].is_contiguous());
  EXPECT_TRUE(torch::allclose(inputs[0], contig));
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
  uint64_t recv_ms = 10;
  uint64_t send_ms = 20;
  auto status = starpu_server::InferenceServiceImpl::populate_response(
      &req, &reply, outputs, recv_ms, send_ms);
  ASSERT_TRUE(status.ok());
  starpu_server::verify_populate_response(
      req, reply, outputs, recv_ms, send_ms);
}

TEST(InferenceServiceImpl, PopulateResponseHandlesNonContiguousOutputs)
{
  auto req = starpu_server::make_model_request("model", "1");
  auto base = torch::tensor({{1, 2}, {3, 4}});
  auto noncontig = base.transpose(0, 1);
  ASSERT_FALSE(noncontig.is_contiguous());
  std::vector<torch::Tensor> outputs = {noncontig};
  inference::ModelInferResponse reply;
  uint64_t recv_ms = 10;
  uint64_t send_ms = 20;
  auto status = starpu_server::InferenceServiceImpl::populate_response(
      &req, &reply, outputs, recv_ms, send_ms);
  ASSERT_TRUE(status.ok());
  auto contig = noncontig.contiguous();
  starpu_server::verify_populate_response(
      req, reply, {contig}, recv_ms, send_ms);
}

TEST(InferenceServiceImpl, PopulateResponseDetectsOverflow)
{
  auto req = starpu_server::make_model_request("model", "1");
  static float dummy;
  const size_t huge_elems =
      std::numeric_limits<size_t>::max() / sizeof(float) + 1;
  auto huge_tensor = torch::from_blob(
      &dummy, {static_cast<int64_t>(huge_elems)},
      torch::TensorOptions().dtype(at::kFloat));
  inference::ModelInferResponse reply;
  int64_t recv_ms = 0;
  int64_t send_ms = 0;
  auto status = starpu_server::InferenceServiceImpl::populate_response(
      &req, &reply, {huge_tensor}, recv_ms, send_ms);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(InferenceService, ValidateInputsMismatchedRawContents)
{
  auto req = starpu_server::make_valid_request();
  req.add_raw_input_contents("extra");
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

TEST(InferenceService, ValidateInputsDatatypeMismatch)
{
  auto req = starpu_server::make_valid_request();
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  std::vector<at::ScalarType> expected_types = {at::kLong};
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, expected_types);
  std::vector<torch::Tensor> inputs;
  auto status = service.validate_and_convert_inputs(&req, inputs);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(InferenceService, ValidateInputsShapeOverflow)
{
  starpu_server::InputSpec spec{
      {std::numeric_limits<int64_t>::max()},
      at::kFloat,
      starpu_server::to_raw_data<float>({1.0F})};
  auto req = starpu_server::make_model_infer_request({spec});
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
