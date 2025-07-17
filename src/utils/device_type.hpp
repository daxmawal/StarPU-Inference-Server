#pragma once

#include <cstdint>

namespace starpu_server {
// =============================================================================
// DeviceType enum defines where an inference task is executed.
// =============================================================================

enum class DeviceType : uint8_t {
  CPU,     // Inference runs on CPU
  CUDA,    // Inference runs on CUDA-capable GPU
  Unknown  // Fallback or undefined execution context
};

// Utility for converting DeviceType to string (for logs/debug)
inline auto
to_string(const DeviceType& type) -> const char*
{
  using enum DeviceType;
  switch (type) {
    case CPU:
      return "CPU";
    case CUDA:
      return "CUDA";
    case Unknown:
      return "Unknown";
    default:
      return "InvalidDeviceType";
  }
}
}  // namespace starpu_server
