#pragma once

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <memory>
#include <vector>

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

  starpu_server::InferenceQueue queue;
  std::vector<torch::Tensor> ref_outputs;
  std::unique_ptr<starpu_server::InferenceServiceImpl> service;
};
