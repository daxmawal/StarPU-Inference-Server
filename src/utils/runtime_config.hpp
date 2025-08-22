#pragma once
#include <ATen/core/ScalarType.h>

#include <cstdint>
#include <string>
#include <vector>

#include "datatype_utils.hpp"
#include "logger.hpp"

namespace starpu_server {
// =============================================================================
// TensorConfig
// -----------------------------------------------------------------------------
// Configuration for a single tensor.
// =============================================================================
struct TensorConfig {
  std::string name;
  std::vector<int64_t> dims;
  at::ScalarType type = at::ScalarType::Undefined;
};

// =============================================================================
// RuntimeConfig
// -----------------------------------------------------------------------------
// Global configuration structure for inference runtime.
//
// Contains:
//   - General settings (model, scheduler, etc.)
//   - Device configuration (CPU, CUDA, GPU IDs)
//   - Logging level
//   - Model input/output layout
// =============================================================================
struct RuntimeConfig {
  std::string scheduler = "lws";
  std::string model_path;
  std::string config_path;
  std::string server_address = "0.0.0.0:50051";
  int metrics_port = 9090;

  std::vector<int> device_ids;
  std::vector<TensorConfig> inputs;
  std::vector<TensorConfig> outputs;

  VerbosityLevel verbosity = VerbosityLevel::Info;
  int iterations = 1;
  int delay_ms = 0;
  int max_batch_size = 1;
  int max_message_bytes = 32 * 1024 * 1024;

  bool synchronous = false;
  bool show_help = false;
  bool valid = true;
  bool use_cpu = true;
  bool use_cuda = false;
};

inline auto
compute_max_message_bytes(
    int max_batch_size, const std::vector<TensorConfig>& tensors) -> int
{
  size_t per_sample_bytes = 0;
  for (const auto& t : tensors) {
    size_t numel = 1;
    for (int64_t d : t.dims) {
      numel *= static_cast<size_t>(d);
    }
    per_sample_bytes += numel * element_size(t.type);
  }
  return static_cast<int>(
      per_sample_bytes * static_cast<size_t>(max_batch_size));
}
}  // namespace starpu_server
