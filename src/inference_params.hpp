#pragma once

#include <torch/script.h>

#include <array>
#include <chrono>

#include "device_type.hpp"
#include "logger.hpp"

// =============================================================================
// Constants for inference limitations
// =============================================================================
namespace InferLimits {
constexpr size_t MaxInputs = 16;     // Max number of input tensors
constexpr size_t MaxDims = 8;        // Max number of dimensions per tensor
constexpr size_t MaxModelsGPU = 32;  // Max number of GPU
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
  int* device_id =
      nullptr;  // Where the task was executed (CPU/GPU, which device)
  int* worker_id = nullptr;  // Where the task was executed, wich StarPU worker
  DeviceType* executed_on = nullptr;
};

// =============================================================================
// Model pointer container (CPU + replicated GPU modules)
// =============================================================================
struct ModelPointers {
  torch::jit::script::Module* model_cpu = nullptr;
  std::array<torch::jit::script::Module*, InferLimits::MaxModelsGPU>
      models_gpu{};
  size_t num_models_gpu = 0;
};

// =============================================================================
// Describes the expected input layout
// =============================================================================
struct TensorLayout {
  std::array<std::array<int64_t, InferLimits::MaxDims>, InferLimits::MaxInputs>
      dims{};
  std::array<int64_t, InferLimits::MaxInputs> num_dims{};
  std::array<at::ScalarType, InferLimits::MaxInputs> input_types{};
};

}  // namespace detail

// =============================================================================
// Parameters passed to an inference task
// =============================================================================
struct InferenceParams {
  detail::ModelPointers models;  // Model handles (CPU + GPU replicas)
  detail::TensorLayout layout;   // Tensor dimensions and types
  size_t num_inputs = 0;         // Number of input tensors
  size_t num_outputs = 0;        // Number of output tensors
  unsigned int job_id = 0;       // Job identifier for logging/debugging
  detail::DeviceInfo device;
  detail::Timing timing;  // Timing data (for benchmarking)
  VerbosityLevel verbosity = VerbosityLevel::Silent;  // Logging verbosity
};
