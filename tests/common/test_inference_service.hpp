#pragma once

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "core/inference_runner.hpp"
#include "grpc/server/inference_service.hpp"
#include "test_constants.hpp"
#include "test_helpers.hpp"

inline void
expect_empty_infer_response(const inference::ModelInferResponse& resp)
{
  EXPECT_EQ(resp.model_name(), "");
  EXPECT_EQ(resp.model_version(), "");
  EXPECT_EQ(resp.outputs_size(), 0);
  EXPECT_EQ(resp.raw_output_contents_size(), 0);
  EXPECT_EQ(resp.server_receive_ms(), 0);
  EXPECT_EQ(resp.server_send_ms(), 0);
  EXPECT_DOUBLE_EQ(resp.server_preprocess_ms(), 0.0);
  EXPECT_DOUBLE_EQ(resp.server_queue_ms(), 0.0);
  EXPECT_DOUBLE_EQ(resp.server_submit_ms(), 0.0);
  EXPECT_DOUBLE_EQ(resp.server_scheduling_ms(), 0.0);
  EXPECT_DOUBLE_EQ(resp.server_codelet_ms(), 0.0);
  EXPECT_DOUBLE_EQ(resp.server_inference_ms(), 0.0);
  EXPECT_DOUBLE_EQ(resp.server_callback_ms(), 0.0);
  EXPECT_DOUBLE_EQ(resp.server_postprocess_ms(), 0.0);
  EXPECT_DOUBLE_EQ(resp.server_total_ms(), 0.0);
  EXPECT_DOUBLE_EQ(resp.server_overall_ms(), 0.0);
}

namespace starpu_server {

inline auto
make_shape_request(
    const std::vector<int64_t>& shape,
    float fill_value = test_constants::kF1) -> inference::ModelInferRequest
{
  size_t total = 1;
  for (const auto dim : shape) {
    total *= static_cast<size_t>(dim);
  }
  std::vector<float> values(total, fill_value);
  InputSpec spec;
  spec.shape = shape;
  spec.dtype = at::kFloat;
  spec.raw_data = to_raw_data(values);
  return make_model_infer_request({spec});
}

}  // namespace starpu_server

class InferenceServiceTest : public ::testing::Test {
 protected:
  struct ServiceConfig {
    std::vector<at::ScalarType> expected_input_types = {at::kFloat};
    std::optional<std::vector<std::vector<int64_t>>> expected_input_dims;
    int max_batch_size = 0;
  };

  virtual auto make_service_config() const -> ServiceConfig { return {}; }

  void SetUp() override
  {
    auto config = make_service_config();
    if (config.expected_input_dims.has_value()) {
      service = std::make_unique<starpu_server::InferenceServiceImpl>(
          &queue, &ref_outputs, std::move(config.expected_input_types),
          *config.expected_input_dims, config.max_batch_size);
    } else {
      ASSERT_EQ(config.max_batch_size, 0)
          << "max_batch_size requires expected_input_dims";
      service = std::make_unique<starpu_server::InferenceServiceImpl>(
          &queue, &ref_outputs, std::move(config.expected_input_types));
    }
  }
  auto prepare_job(
      std::vector<torch::Tensor> ref_outs,
      std::vector<torch::Tensor> worker_outs = {},
      std::function<void(starpu_server::InferenceJob&)> job_mutator = {})
      -> std::jthread
  {
    ref_outputs = std::move(ref_outs);
    return starpu_server::run_single_job(
        queue, std::move(worker_outs), 0.0, std::move(job_mutator));
  }
  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  std::unique_ptr<starpu_server::InferenceServiceImpl> service;
  grpc::ServerContext ctx;
  inference::ModelInferResponse reply;
};

struct SubmitJobAndWaitCase {
  std::vector<torch::Tensor> ref_outputs;
  std::vector<torch::Tensor> worker_outputs;
  grpc::StatusCode expected_status;
};

class SubmitJobAndWaitTest
    : public InferenceServiceTest,
      public ::testing::WithParamInterface<SubmitJobAndWaitCase> {};
