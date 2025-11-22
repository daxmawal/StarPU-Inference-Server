#pragma once

#include <gtest/gtest.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <vector>

struct ExtractTensorsParam {
  c10::IValue input;
  std::vector<at::Tensor> expected;
};

class StarPUSetupExtractTensorsTest
    : public ::testing::TestWithParam<ExtractTensorsParam> {};
