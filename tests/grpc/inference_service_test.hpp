#pragma once

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <memory>
#include <vector>

#include "../test_helpers.hpp"
#include "grpc/server/inference_service.hpp"

// Test fixture providing a ready-to-use InferenceServiceImpl instance along
// with its required dependencies.
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
