#pragma once
#include <ATen/core/ScalarType.h>

#include <cstdint>
#include <string>
#include <vector>

#include "datatype_utils.hpp"
#include "logger.hpp"

namespace starpu_server {
// =============================================================================
// RuntimeConfig
// -----------------------------------------------------------------------------
// Global configuration structure for inference runtime.
//
// Contains:
//   - General settings (model, scheduler, etc.)
//   - Device configuration (CPU, CUDA, GPU IDs)
//   - Logging level
//   - Model input layout (dims and types)
// =============================================================================
struct RuntimeConfig {
  std::string scheduler = "lws";
  std::string model_path;
  std::string config_path;
  std::string server_address = "0.0.0.0:50051";
  int metrics_port = 9090;

  std::vector<int> device_ids;
  std::vector<std::vector<int64_t>> input_dims;
  std::vector<at::ScalarType> input_types;

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
    int max_batch_size, const std::vector<std::vector<int64_t>>& dims,
    const std::vector<at::ScalarType>& types) -> int
{
  size_t per_sample_bytes = 0;
  const size_t n = std::min(dims.size(), types.size());
  for (size_t i = 0; i < n; ++i) {
    size_t numel = 1;
    for (int64_t dim : dims[i]) {
      numel *= static_cast<size_t>(dim);
    }
    per_sample_bytes += numel * element_size(types[i]);
  }
  return static_cast<int>(
      per_sample_bytes * static_cast<size_t>(max_batch_size));
}
}  // namespace starpu_server
