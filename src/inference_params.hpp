#pragma once
#include <torch/script.h>

#include <array>

#include "device_type.hpp"

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
  int job_id = -1;
  int* device_id;
  DeviceType* executed_on = nullptr;
  std::chrono::high_resolution_clock::time_point* codelet_start_time;
  std::chrono::high_resolution_clock::time_point* codelet_end_time;
  std::chrono::high_resolution_clock::time_point* inference_start_time;

  std::array<std::array<int64_t, InferLimits::MaxDims>, InferLimits::MaxInputs>
      dims{};
  std::array<int64_t, InferLimits::MaxInputs> num_dims{};
  std::array<at::ScalarType, InferLimits::MaxInputs> input_types{};
};