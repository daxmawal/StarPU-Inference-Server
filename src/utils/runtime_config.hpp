#pragma once
#include <ATen/core/ScalarType.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "datatype_utils.hpp"
#include "exceptions.hpp"
#include "logger.hpp"

namespace starpu_server {
// =============================================================================
// Compile-time defaults for inference limits
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
  std::size_t max_message_bytes = 32 * 1024 * 1024;
  size_t pregen_inputs = 10;
  size_t warmup_pregen_inputs = 2;
  int warmup_iterations = 2;

  size_t max_inputs = STARPU_SERVER_MAX_INPUTS;
  size_t max_dims = STARPU_SERVER_MAX_DIMS;
  size_t max_models_gpu = STARPU_SERVER_MAX_MODELS_GPU;

  std::optional<uint64_t> seed{};

  bool synchronous = false;
  bool show_help = false;
  bool valid = true;
  bool use_cpu = true;
  bool use_cuda = false;
};

inline auto
compute_max_message_bytes(
    int max_batch_size, const std::vector<TensorConfig>& inputs,
    const std::vector<TensorConfig>& outputs,
    std::size_t min_message_bytes = 32 * 1024 * 1024) -> std::size_t
{
  if (max_batch_size <= 0) {
    throw InvalidDimensionException("max_batch_size must be > 0");
  }
  size_t per_sample_bytes = 0;
  const auto accumulate_bytes = [&](const std::vector<TensorConfig>& tensors) {
    for (const auto& t : tensors) {
      size_t numel = 1;
      for (int64_t d : t.dims) {
        if (d < 0) {
          throw InvalidDimensionException(
              "dimension size must be non-negative");
        }
        const auto d_size = static_cast<size_t>(d);
        if (d_size != 0 &&
            numel > std::numeric_limits<size_t>::max() / d_size) {
          throw MessageSizeOverflowException(
              "numel * dimension size would overflow size_t");
        }
        numel *= d_size;
      }
      const size_t type_size = element_size(t.type);
      if (numel > std::numeric_limits<size_t>::max() / type_size) {
        throw MessageSizeOverflowException(
            "numel * element size would overflow size_t");
      }
      const size_t tensor_bytes = numel * type_size;
      if (per_sample_bytes >
          std::numeric_limits<size_t>::max() - tensor_bytes) {
        throw MessageSizeOverflowException(
            "per_sample_bytes + tensor_bytes would overflow size_t");
      }
      per_sample_bytes += tensor_bytes;
    }
  };

  accumulate_bytes(inputs);
  accumulate_bytes(outputs);

  if (per_sample_bytes > std::numeric_limits<size_t>::max() /
                             static_cast<size_t>(max_batch_size)) {
    throw MessageSizeOverflowException(
        "per_sample_bytes * max_batch_size overflows size_t");
  }

  const size_t total = per_sample_bytes * static_cast<size_t>(max_batch_size);
  return std::max(total, min_message_bytes);
}
}  // namespace starpu_server
