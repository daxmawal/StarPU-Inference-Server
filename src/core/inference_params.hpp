#pragma once

#include <torch/script.h>

#include <chrono>

#include "device_type.hpp"
#include "logger.hpp"

namespace starpu_server {
// =============================================================================
// Constants for inference limitations
// =============================================================================

namespace InferLimits {
inline constexpr size_t MaxInputs = 16;
inline constexpr size_t MaxDims = 8;
inline constexpr size_t MaxModelsGPU = 32;
}  // namespace InferLimits

namespace detail {
// =============================================================================
// Timing info for measuring durations
// =============================================================================

struct Timing {
  std::chrono::high_resolution_clock::time_point* codelet_start_time = nullptr;
  std::chrono::high_resolution_clock::time_point* codelet_end_time = nullptr;
  std::chrono::high_resolution_clock::time_point* inference_start_time =
      nullptr;
};

// =============================================================================
// Device-related info (on which device the inference ran)
// =============================================================================

struct DeviceInfo {
  int* device_id = nullptr;
  int* worker_id = nullptr;
  DeviceType* executed_on = nullptr;
};

// =============================================================================
// Model pointer container (CPU + replicated GPU modules)
// =============================================================================

struct ModelPointers {
  torch::jit::script::Module* model_cpu = nullptr;
  std::vector<torch::jit::script::Module*> models_gpu;
  size_t num_models_gpu = 0;
};

// =============================================================================
// Describes the expected input layout
// =============================================================================

struct TensorLayout {
  std::vector<std::vector<int64_t>> dims;
  std::vector<int64_t> num_dims;
  std::vector<at::ScalarType> input_types;
};

// =============================================================================
// Limit values carried with inference parameters
// =============================================================================

struct Limits {
  size_t max_inputs = 0;
  size_t max_dims = 0;
  size_t max_models_gpu = 0;
};

}  // namespace detail

// =============================================================================
// Parameters passed to an inference task
// =============================================================================

struct InferenceParams {
  detail::ModelPointers models;
  detail::TensorLayout layout;
  detail::DeviceInfo device;
  detail::Timing timing;
  detail::Limits limits{};
  size_t num_inputs = 0;
  size_t num_outputs = 0;
  int64_t batch_size = 1;
  int request_id = 0;
  VerbosityLevel verbosity = VerbosityLevel::Silent;
};

}  // namespace starpu_server
