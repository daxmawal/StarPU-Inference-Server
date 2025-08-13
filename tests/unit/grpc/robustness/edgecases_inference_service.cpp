#include "test_inference_service.hpp"

TEST(InferenceService, ValidateInputsSizeMismatch)
{
  auto req = starpu_server::make_valid_request();
  req.mutable_raw_input_contents(0)->append("0", 1);
  std::vector<torch::Tensor> inputs;
  auto status =
      starpu_server::InferenceServiceImpl::validate_and_convert_inputs(
          &req, inputs);
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
