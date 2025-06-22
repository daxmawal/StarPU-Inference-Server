#pragma once
#include <ATen/core/ScalarType.h>

#include <cstdint>
#include <string>
#include <vector>

#include "logger.hpp"

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
  // General configuration
  std::string scheduler = "lws";  // Scheduling policy
  std::string model_path;         // Path to ONNX or TorchScript model
  int iterations = 1;             // Number of inference iterations
  bool synchronous = false;       // Run synchronously or asynchronously
  int delay_ms = 0;               // Artificial delay between inferences (ms)
  bool show_help = false;         // Show help message and exit
  bool valid = true;              // Was parsing successful

  // Device configuration
  bool use_cpu = true;          // Use CPU for inference
  bool use_cuda = false;        // Use CUDA-enabled GPUs
  std::vector<int> device_ids;  // Given available GPU device IDs

  // Logging
  VerbosityLevel verbosity = VerbosityLevel::Info;

  // Server configuration
  std::string server_address = "0.0.0.0:50051";
  int max_message_bytes = 32 * 1024 * 1024;

  // Model input specifications
  std::vector<std::vector<int64_t>> input_shapes;  // Expected input shapes
  std::vector<at::ScalarType> input_types;  // Corresponding input data types
};