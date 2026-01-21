#pragma once
#include <ATen/core/ScalarType.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "datatype_utils.hpp"
#include "exceptions.hpp"
#include "inference_limits.hpp"
#include "logger.hpp"

namespace starpu_server {
// =============================================================================
// Compile-time defaults for inference limits
// =============================================================================
inline constexpr std::size_t kBytesPerKiB = 1024ULL;
inline constexpr std::size_t kBytesPerMiB = kBytesPerKiB * 1024ULL;
inline constexpr std::size_t kDefaultMessageSizeMiB = 32ULL;
inline constexpr std::size_t kDefaultMinMessageBytes =
    kDefaultMessageSizeMiB * kBytesPerMiB;
inline constexpr std::size_t kDefaultMaxQueueSize = 100ULL;
inline constexpr std::string_view kDefaultTraceFileName = "perfetto_trace.json";
inline constexpr int kDefaultMetricsPort = 9090;
inline constexpr std::string_view kDefaultStarpuScheduler = "lws";
inline constexpr std::string_view kStarpuSchedulerEnvVar = "STARPU_SCHED";
// =============================================================================
// Compile-time defaults for congestion detection
// =============================================================================
inline constexpr double kDefaultCongestionQueueLatencyBudgetRatio = 0.30;
inline constexpr double kDefaultCongestionE2EWarnRatio = 0.90;
inline constexpr double kDefaultCongestionE2EOkRatio = 0.80;
inline constexpr double kDefaultCongestionFillHigh = 0.80;
inline constexpr double kDefaultCongestionFillLow = 0.60;
inline constexpr double kDefaultCongestionRhoHigh = 1.05;
inline constexpr double kDefaultCongestionRhoLow = 0.90;
inline constexpr double kDefaultCongestionEwmaAlpha = 0.2;
inline constexpr int kDefaultCongestionEntryHorizonMs = 1000;
inline constexpr int kDefaultCongestionExitHorizonMs = 2000;
inline constexpr int kDefaultCongestionTickIntervalMs = 1000;

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
//   - General settings (model, logging, etc.)
//   - Device configuration (CPU, CUDA, GPU IDs)
//   - Logging level
//   - Model input/output layout
// =============================================================================
struct RuntimeConfig {
  struct DeviceSettings {
    std::vector<int> ids;
    bool use_cpu = true;
    bool use_cuda = false;
    bool group_cpu_by_numa = false;
  };

  struct BatchingSettings {
    int batch_coalesce_timeout_ms = 0;
    int max_batch_size = 1;
    int pool_size = 0;
    std::size_t max_message_bytes = kDefaultMinMessageBytes;
    std::size_t max_queue_size = kDefaultMaxQueueSize;
    std::size_t max_inflight_tasks = 0;
    size_t warmup_pregen_inputs = 2;
    int warmup_request_nb = 2;
    int warmup_batches_per_worker = 1;
    bool synchronous = false;
    bool dynamic_batching = true;
    bool trace_enabled = false;
    std::string trace_output_path = std::string(kDefaultTraceFileName);
  };

  struct CongestionSettings {
    bool enabled = true;          // Enable congestion detection.
    double latency_slo_ms = 0.0;  // End-to-end SLO (ms); 0 disables SLO.
    double queue_latency_budget_ms =
        0.0;  // Queue latency budget (ms); 0 uses ratio.
    double queue_latency_budget_ratio =
        kDefaultCongestionQueueLatencyBudgetRatio;  // Fraction of SLO reserved
                                                    // for queue.
    double e2e_warn_ratio =
        kDefaultCongestionE2EWarnRatio;  // SLO ratio to trigger congestion.
    double e2e_ok_ratio =
        kDefaultCongestionE2EOkRatio;  // SLO ratio to clear congestion.
    double fill_high =
        kDefaultCongestionFillHigh;  // Queue fill ratio to enter congestion.
    double fill_low =
        kDefaultCongestionFillLow;  // Queue fill ratio to exit congestion.
    double rho_high =
        kDefaultCongestionRhoHigh;  // Arrival/processing ratio to enter.
    double rho_low =
        kDefaultCongestionRhoLow;  // Arrival/processing ratio to exit.
    double alpha = kDefaultCongestionEwmaAlpha;  // EWMA smoothing factor.
    int entry_horizon_ms =
        kDefaultCongestionEntryHorizonMs;  // Time in condition before entering
                                           // (ms).
    int exit_horizon_ms =
        kDefaultCongestionExitHorizonMs;  // Time in condition before exiting
                                          // (ms).
    int tick_interval_ms =
        kDefaultCongestionTickIntervalMs;  // Sampling interval for congestion.
  };

  struct Limits {
    size_t max_inputs = InferLimits::MaxInputs;
    size_t max_dims = InferLimits::MaxDims;
    size_t max_models_gpu = InferLimits::MaxModelsGPU;
  };

  std::string name;
  std::string server_address = "127.0.0.1:50051";
  int metrics_port = kDefaultMetricsPort;

  std::map<std::string, std::string, std::less<>> starpu_env;
  std::optional<ModelConfig> model;
  VerbosityLevel verbosity = VerbosityLevel::Silent;
  DeviceSettings devices{};
  BatchingSettings batching{};
  Limits limits{};
  std::optional<uint64_t> seed;
  CongestionSettings congestion{};
  bool valid = true;
};

inline auto
compute_model_message_bytes(
    int max_batch_size, const std::vector<TensorConfig>& inputs,
    const std::vector<TensorConfig>& outputs,
    std::size_t min_message_bytes = kDefaultMinMessageBytes) -> std::size_t
{
  if (max_batch_size <= 0) {
    throw InvalidDimensionException("max_batch_size must be > 0");
  }
  size_t per_sample_bytes = 0;
  const auto compute_numel = [](const TensorConfig& tensor_config) {
    size_t numel = 1;
    for (int64_t dim : tensor_config.dims) {
      if (dim < 0) {
        throw InvalidDimensionException("dimension size must be non-negative");
      }
      const auto dim_size = static_cast<size_t>(dim);
      if (dim_size != 0 &&
          numel > std::numeric_limits<size_t>::max() / dim_size) {
        throw MessageSizeOverflowException(
            "numel * dimension size would overflow size_t");
      }
      numel *= dim_size;
    }
    return numel;
  };

  const auto tensor_bytes = [&](const TensorConfig& tensor_config) {
    const size_t numel = compute_numel(tensor_config);
    size_t type_size = 0;
    try {
      type_size = element_size(tensor_config.type);
    }
    catch (const std::invalid_argument& e) {
      throw UnsupportedDtypeException(e.what());
    }
    if (numel > std::numeric_limits<size_t>::max() / type_size) {
      throw MessageSizeOverflowException(
          "numel * element size would overflow size_t");
    }
    return numel * type_size;
  };

  const auto accumulate_bytes = [&](const std::vector<TensorConfig>& tensors) {
    for (const auto& tensor : tensors) {
      const size_t current_tensor_bytes = tensor_bytes(tensor);
      if (per_sample_bytes >
          std::numeric_limits<size_t>::max() - current_tensor_bytes) {
        throw MessageSizeOverflowException(
            "per_sample_bytes + tensor_bytes would overflow size_t");
      }
      per_sample_bytes += current_tensor_bytes;
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
    int max_batch_size, const std::optional<ModelConfig>& model,
    std::size_t min_message_bytes = kDefaultMinMessageBytes) -> std::size_t
{
  if (!model.has_value()) {
    return min_message_bytes;
  }

  return compute_model_message_bytes(
      max_batch_size, model->inputs, model->outputs, min_message_bytes);
}
}  // namespace starpu_server
