#pragma once

#include <torch/script.h>

#include <chrono>

#include "device_type.hpp"
#include "logger.hpp"

namespace starpu_server {
// =============================================================================
// Constants for inference limitations
// =============================================================================

#ifndef STARPU_SERVER_MAX_INPUTS
#define STARPU_SERVER_MAX_INPUTS 16
#endif

#ifndef STARPU_SERVER_MAX_DIMS
#define STARPU_SERVER_MAX_DIMS 8
#endif

#ifndef STARPU_SERVER_MAX_MODELS_GPU
#define STARPU_SERVER_MAX_MODELS_GPU 32
#endif

namespace InferLimits {
constexpr size_t MaxInputs = STARPU_SERVER_MAX_INPUTS;
constexpr size_t MaxDims = STARPU_SERVER_MAX_DIMS;
constexpr size_t MaxModelsGPU = STARPU_SERVER_MAX_MODELS_GPU;
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
  std::vector<torch::jit::script::Module*> models_gpu{};
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
  int job_id = 0;
  VerbosityLevel verbosity = VerbosityLevel::Silent;
};

}  // namespace starpu_server
