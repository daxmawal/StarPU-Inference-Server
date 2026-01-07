#pragma once

#include <torch/script.h>

#include <chrono>

#include "device_type.hpp"
#include "inference_limits.hpp"
#include "logger.hpp"
#include "utils/monotonic_clock.hpp"

namespace starpu_server {

namespace detail {
// =============================================================================
// Timing info for measuring durations
// =============================================================================

struct Timing {
  MonotonicClock::time_point* codelet_start_time = nullptr;
  MonotonicClock::time_point* codelet_end_time = nullptr;
  MonotonicClock::time_point* inference_start_time = nullptr;
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
  std::vector<int> device_ids;
  std::vector<torch::jit::script::Module*> models_gpu;
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

  // Cached views reused for each codelet invocation to avoid re-allocations.
  std::vector<torch::Tensor> cached_input_tensors;
  std::vector<c10::IValue> cached_input_ivalues;
};

}  // namespace starpu_server
