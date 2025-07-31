#pragma once
#include <ATen/core/ScalarType.h>

#include <cstdint>
#include <string>
#include <vector>

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
//   - Model input layout (shapes and types)
// =============================================================================
struct RuntimeConfig {
  std::string scheduler = "lws";
  std::string model_path;
  std::string server_address = "0.0.0.0:50051";

  std::vector<int> device_ids;
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<at::ScalarType> input_types;

  VerbosityLevel verbosity = VerbosityLevel::Info;
  int iterations = 1;
  int delay_ms = 0;
  int max_message_bytes = 32 * 1024 * 1024;

  bool synchronous = false;
  bool show_help = false;
  bool valid = true;
  bool use_cpu = true;
  bool use_cuda = false;
};
}  // namespace starpu_server
