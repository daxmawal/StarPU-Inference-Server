#include <limits>

#include "test_inference_service.hpp"

TEST(InferenceService, ValidateInputsCountMismatch)
{
  auto req = starpu_server::make_valid_request();
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  std::vector<at::ScalarType> expected_types = {at::kFloat, at::kFloat};
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, expected_types);
  std::vector<torch::Tensor> inputs;
  auto status = service.validate_and_convert_inputs(&req, inputs);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(InferenceService, ValidateInputsSizeMismatch)
{
  auto req = starpu_server::make_valid_request();
  req.mutable_raw_input_contents(0)->append("0", 1);
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  std::vector<at::ScalarType> expected_types = {at::kFloat};
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, expected_types);
  std::vector<torch::Tensor> inputs;
  auto status = service.validate_and_convert_inputs(&req, inputs);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(InferenceService, ValidateInputsNegativeDimension)
{
  std::vector<float> data = {1.0F};
  auto req = starpu_server::make_model_infer_request({
      {{-1}, at::kFloat, starpu_server::to_raw_data(data)},
  });
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  std::vector<at::ScalarType> expected_types = {at::kFloat};
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, expected_types);
  std::vector<torch::Tensor> inputs;
  auto status = service.validate_and_convert_inputs(&req, inputs);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(InferenceService, ValidateInputsZeroDimension)
{
  std::vector<float> data = {};
  auto req = starpu_server::make_model_infer_request({
      {{0}, at::kFloat, starpu_server::to_raw_data(data)},
  });
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  std::vector<at::ScalarType> expected_types = {at::kFloat};
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, expected_types);
  std::vector<torch::Tensor> inputs;
  auto status = service.validate_and_convert_inputs(&req, inputs);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

TEST(InferenceService, ValidateInputsDimensionOverflow)
{
  const auto large_dim = std::numeric_limits<int64_t>::max();
  std::vector<float> data = {};
  auto req = starpu_server::make_model_infer_request({
      {{large_dim}, at::kFloat, starpu_server::to_raw_data(data)},
  });
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  std::vector<at::ScalarType> expected_types = {at::kFloat};
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, expected_types);
  std::vector<torch::Tensor> inputs;
  auto status = service.validate_and_convert_inputs(&req, inputs);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
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

TEST_F(InferenceServiceTest, InvalidDatatypeDoesNotShutdownServer)
{
  auto req = starpu_server::make_valid_request();
  req.mutable_inputs(0)->set_datatype("INT32");
  req.MergeFrom(starpu_server::make_model_request("m", "1"));
  auto status = service->ModelInfer(&ctx, &req, &reply);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  expect_empty_infer_response(reply);

  auto valid_req = starpu_server::make_valid_request();
  valid_req.MergeFrom(starpu_server::make_model_request("m", "1"));
  auto worker = prepare_job({torch::zeros({2, 2})}, {torch::zeros({2, 2})});
  status = service->ModelInfer(&ctx, &valid_req, &reply);
  EXPECT_TRUE(status.ok());
}

TEST(GrpcServer, StopServerNullptr)
{
  std::unique_ptr<grpc::Server> server;
  EXPECT_NO_THROW(starpu_server::StopServer(server));
  EXPECT_EQ(server, nullptr);
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
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  std::vector<at::ScalarType> expected_types = {at::kFloat};
  starpu_server::InferenceServiceImpl service(
      &queue, &ref_outputs, expected_types);
  std::vector<torch::Tensor> inputs;
  auto status = service.validate_and_convert_inputs(&req, inputs);
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
}

INSTANTIATE_TEST_SUITE_P(
    , InferenceServiceValidation,
    ::testing::Values(
        InvalidRequestCase{
            "RawInputCountMismatch",
            [] {
              std::vector<float> data = {1.0F, 2.0F, 3.0F, 4.0F};
              auto req = starpu_server::make_model_infer_request(
                  {{{2, 2}, at::kFloat, starpu_server::to_raw_data(data)}});
              req.add_raw_input_contents()->assign("", 0);
              return req;
            }()},
        InvalidRequestCase{
            "RawContentSizeMismatch",
            [] {
              std::vector<float> data = {1.0F, 2.0F, 3.0F};
              return starpu_server::make_model_infer_request(
                  {{{2, 2}, at::kFloat, starpu_server::to_raw_data(data)}});
            }()}),
    [](const testing::TestParamInfo<InvalidRequestCase>& info) {
      return info.param.name;
    });
