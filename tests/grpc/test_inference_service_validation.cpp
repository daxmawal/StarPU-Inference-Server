#include <gtest/gtest.h>

#include "../test_helpers.hpp"
#include "inference_service_test.hpp"

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