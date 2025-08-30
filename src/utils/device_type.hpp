#pragma once

#include <cstdint>

namespace starpu_server {
// =============================================================================
// DeviceType enum defines where an inference task is executed.
// =============================================================================

enum class DeviceType : uint8_t { CPU, CUDA, Unknown };

inline auto
to_string(DeviceType type) -> const char*
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
