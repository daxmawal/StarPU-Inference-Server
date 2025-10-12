#pragma once
#include <ATen/core/ScalarType.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "datatype_utils.hpp"
#include "exceptions.hpp"
#include "logger.hpp"
#include "transparent_hash.hpp"

namespace starpu_server {
// =============================================================================
// Compile-time defaults for inference limits
// =============================================================================
inline constexpr std::size_t kMaxInputs = 16;
inline constexpr std::size_t kMaxDims = 8;
inline constexpr std::size_t kMaxModelsGpu = 32;

inline const std::unordered_set<std::string, TransparentHash, std::equal_to<>>
    kAllowedSchedulers = {"lws",  "dmda",   "dmdas", "ws",   "eager", "random",
                          "prio", "peager", "pheft", "heft", "fcfs"};

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
// ModelConfig
// -----------------------------------------------------------------------------
// Configuration for a single model, including its path and I/O tensors.
// =============================================================================
struct ModelConfig {
  std::string name;
  std::string path;
  std::vector<TensorConfig> inputs;
  std::vector<TensorConfig> outputs;
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
  std::string config_path;
  std::string server_address = "0.0.0.0:50051";
  int metrics_port = 9090;

  std::vector<int> device_ids;
  std::vector<ModelConfig> models;

  VerbosityLevel verbosity = VerbosityLevel::Info;
  int request_nb = 1;
  int delay_ms = 0;
  int batch_coalesce_timeout_ms = 0;
  int max_batch_size = 1;
  int input_slots = 0;
  std::size_t max_message_bytes = 32 * 1024 * 1024;
  size_t pregen_inputs = 10;
  size_t warmup_pregen_inputs = 2;
  int warmup_request_nb = 2;
  double rtol = 1e-3;
  double atol = 1e-5;
  bool validate_results = true;

  size_t max_inputs = kMaxInputs;
  size_t max_dims = kMaxDims;
  size_t max_models_gpu = kMaxModelsGpu;

  std::optional<uint64_t> seed{};

  bool synchronous = false;
  bool show_help = false;
  bool valid = true;
  bool use_cpu = true;
  bool use_cuda = false;
  bool dynamic_batching = false;
  bool seen_combined_input = false;
};

inline auto
compute_model_message_bytes(
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
      size_t type_size = 0;
      try {
        type_size = element_size(t.type);
      }
      catch (const std::invalid_argument& e) {
        throw UnsupportedDtypeException(e.what());
      }
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

inline auto
compute_max_message_bytes(
    int max_batch_size, const std::vector<ModelConfig>& models,
    std::size_t min_message_bytes = 32 * 1024 * 1024) -> std::size_t
{
  size_t max_bytes = min_message_bytes;
  for (const auto& model : models) {
    const auto bytes = compute_model_message_bytes(
        max_batch_size, model.inputs, model.outputs, min_message_bytes);
    max_bytes = std::max(max_bytes, bytes);
  }
  return max_bytes;
}
}  // namespace starpu_server
