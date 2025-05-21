#pragma once
#include <torch/script.h>

#include <array>

namespace InferLimits {
constexpr size_t MaxInputs = 16;
constexpr size_t MaxDims = 8;
}  // namespace InferLimits

struct InferenceParams {
  torch::jit::script::Module* modele_cpu = nullptr;
  torch::jit::script::Module* modele_gpu = nullptr;
  size_t num_inputs = 0;
  size_t num_outputs = 0;
  int64_t output_size = 0;
  int job_id = 0;

  std::array<std::array<int64_t, InferLimits::MaxDims>, InferLimits::MaxInputs>
      dims{};
  std::array<int64_t, InferLimits::MaxInputs> num_dims{};
  std::array<at::ScalarType, InferLimits::MaxInputs> input_types{};
};