#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstring>
#include <memory>
#include <vector>

#include "grpc/server/inference_service.hpp"
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
}

class InferenceServiceTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    service = std::make_unique<starpu_server::InferenceServiceImpl>(
        &queue, &ref_outputs);
  }
  auto prepare_job(
      std::vector<torch::Tensor> ref_outs,
      std::vector<torch::Tensor> worker_outs = {}) -> std::jthread
  {
    ref_outputs = std::move(ref_outs);
    return starpu_server::run_single_job(queue, std::move(worker_outs));
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
