#pragma once

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <starpu.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <chrono>
#include <climits>
#include <functional>
#include <memory>
#include <vector>

#include "core/inference_params.hpp"
#include "core/starpu_setup.hpp"
#include "core/tensor_builder.hpp"
#include "test_utils.hpp"
#include "utils/exceptions.hpp"

struct ExtractTensorsParam {
  c10::IValue input;
  std::vector<at::Tensor> expected;
};

class StarPUSetupExtractTensorsTest
    : public ::testing::TestWithParam<ExtractTensorsParam> {};
