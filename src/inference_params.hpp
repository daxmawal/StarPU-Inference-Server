#pragma once
#include <torch/script.h>

#include <array>

#include "device_type.hpp"
#include "logger.hpp"

namespace InferLimits {
constexpr size_t MaxInputs = 16;
constexpr size_t MaxDims = 8;
constexpr size_t MaxModelsGPU = 32;
}  // namespace InferLimits

struct InferenceParams {
  // === Models ===
  torch::jit::script::Module* model_cpu = nullptr;
  std::array<torch::jit::script::Module*, InferLimits::MaxModelsGPU>
      models_gpu{};
  size_t num_models_gpu = 0;

  // === Tensor properties ===
  size_t num_inputs = 0;
  size_t num_outputs = 0;
  int64_t output_size = 0;

  std::array<std::array<int64_t, InferLimits::MaxDims>, InferLimits::MaxInputs>
      dims{};
  std::array<int64_t, InferLimits::MaxInputs> num_dims{};
  std::array<at::ScalarType, InferLimits::MaxInputs> input_types{};

  // === Job metadata ===
  unsigned int job_id = 0;
  VerbosityLevel verbosity;

  // === Device information ===
  int* device_id = nullptr;
  DeviceType* executed_on = nullptr;

  // === Timing / Profiling information ===
  std::chrono::high_resolution_clock::time_point* codelet_start_time = nullptr;
  std::chrono::high_resolution_clock::time_point* codelet_end_time = nullptr;
  std::chrono::high_resolution_clock::time_point* inference_start_time =
      nullptr;
};