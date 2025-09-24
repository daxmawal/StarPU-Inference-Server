#include <cstdint>
#include <limits>
#include <span>

#include "test_inference_service.hpp"

namespace {
constexpr float kF1 = 1.0F;
constexpr float kF2 = 2.0F;
constexpr float kF3 = 3.0F;
constexpr float kF4 = 4.0F;
constexpr int64_t kI10 = 10;
constexpr int64_t kI20 = 20;
constexpr int64_t kI30 = 30;
}  // namespace

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
  EXPECT_FLOAT_EQ(inputs[0][0][0].item<float>(), kF1);
}

TEST(InferenceService, ValidateInputsNonContiguous)
{
  auto base = torch::tensor({{kF1, kF2}, {kF3, kF4}});
  auto noncontig = base.transpose(0, 1);
  auto contig = noncontig.contiguous();
  std::span<const float> span{
      contig.data_ptr<float>(), static_cast<size_t>(contig.numel())};
  std::vector<float> data(span.begin(), span.end());
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
  std::vector<float> data0 = {kF1, kF2, kF3, kF4};
  std::vector<int64_t> data1 = {kI10, kI20, kI30};
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
  EXPECT_FLOAT_EQ(inputs[0][0][0].item<float>(), kF1);
  EXPECT_EQ(inputs[1].sizes(), (torch::IntArrayRef{3}));
  EXPECT_EQ(inputs[1].scalar_type(), at::kLong);
  EXPECT_EQ(inputs[1][0].item<int64_t>(), kI10);
}

TEST(InferenceService, ValidateInputsMismatchedCount)
{
  std::vector<float> data0 = {kF1, kF2, kF3, kF4};
  std::vector<int64_t> data1 = {kI10, kI20, kI30};
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
  int64_t recv_ms = kI10;
  int64_t send_ms = kI20;
  starpu_server::InferenceServiceImpl::LatencyBreakdown breakdown;
  breakdown.queue_ms = 1.0;
  breakdown.submit_ms = 2.0;
  breakdown.scheduling_ms = 3.0;
  breakdown.codelet_ms = 4.0;
  breakdown.inference_ms = 5.0;
  breakdown.callback_ms = 6.0;
  breakdown.total_ms = 7.0;
  auto status = starpu_server::InferenceServiceImpl::populate_response(
      &req, &reply, outputs, recv_ms, send_ms, breakdown);
  ASSERT_TRUE(status.ok());
  starpu_server::verify_populate_response(
      req, reply, outputs, recv_ms, send_ms, breakdown);
}

TEST(InferenceServiceImpl, PopulateResponseHandlesNonContiguousOutputs)
{
  auto req = starpu_server::make_model_request("model", "1");
  auto base = torch::tensor({{1, 2}, {3, 4}});
  auto noncontig = base.transpose(0, 1);
  ASSERT_FALSE(noncontig.is_contiguous());
  std::vector<torch::Tensor> outputs = {noncontig};
  inference::ModelInferResponse reply;
  int64_t recv_ms = kI10;
  int64_t send_ms = kI20;
  starpu_server::InferenceServiceImpl::LatencyBreakdown breakdown;
  breakdown.queue_ms = 0.5;
  breakdown.submit_ms = 1.5;
  breakdown.scheduling_ms = 2.5;
  breakdown.codelet_ms = 3.5;
  breakdown.inference_ms = 4.5;
  breakdown.callback_ms = 5.5;
  breakdown.total_ms = 6.5;
  auto status = starpu_server::InferenceServiceImpl::populate_response(
      &req, &reply, outputs, recv_ms, send_ms, breakdown);
  ASSERT_TRUE(status.ok());
  auto contig = noncontig.contiguous();
  starpu_server::verify_populate_response(
      req, reply, {contig}, recv_ms, send_ms, breakdown);
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
  starpu_server::InferenceServiceImpl::LatencyBreakdown breakdown;
  auto status = starpu_server::InferenceServiceImpl::populate_response(
      &req, &reply, {huge_tensor}, recv_ms, send_ms, breakdown);
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

TEST(InferenceService, SubmitJobAndWaitReturnsUnavailableWhenQueueShutdown)
{
  starpu_server::InferenceQueue queue;
  queue.shutdown();
  std::vector<torch::Tensor> ref_outputs = {
      torch::zeros({1}, torch::TensorOptions().dtype(at::kFloat))};
  std::vector<at::ScalarType> expected_types = {at::kFloat};
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, expected_types);
  std::vector<torch::Tensor> inputs = {
      torch::tensor({kF1}, torch::TensorOptions().dtype(at::kFloat))};
  std::vector<torch::Tensor> outputs = {
      torch::tensor({kF2}, torch::TensorOptions().dtype(at::kFloat))};

  starpu_server::InferenceServiceImpl::LatencyBreakdown breakdown;
  auto status = service.submit_job_and_wait(inputs, outputs, breakdown);

  EXPECT_EQ(status.error_code(), grpc::StatusCode::UNAVAILABLE);
  EXPECT_TRUE(outputs.empty());
}
