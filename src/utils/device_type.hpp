#pragma once

#include <cstdint>

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
  switch (type) {
    case DeviceType::CPU:
      return "CPU";
    case DeviceType::CUDA:
      return "CUDA";
    case DeviceType::Unknown:
      return "Unknown";
    default:
      return "InvalidDeviceType";
  }
}