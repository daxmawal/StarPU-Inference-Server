#pragma once

#include <c10/core/Device.h>

#include <optional>

namespace starpu_server::testing {

void set_torch_cuda_device_count_override(
    std::optional<c10::DeviceIndex> override);
void reset_torch_cuda_device_count_override();

class TorchCudaDeviceCountOverrideGuard {
 public:
  explicit TorchCudaDeviceCountOverrideGuard(
      std::optional<c10::DeviceIndex> override);
  explicit TorchCudaDeviceCountOverrideGuard(c10::DeviceIndex override)
      : TorchCudaDeviceCountOverrideGuard(
            std::optional<c10::DeviceIndex>(override))
  {
  }
  ~TorchCudaDeviceCountOverrideGuard();

  TorchCudaDeviceCountOverrideGuard(const TorchCudaDeviceCountOverrideGuard&) =
      delete;
  auto operator=(const TorchCudaDeviceCountOverrideGuard&)
      -> TorchCudaDeviceCountOverrideGuard& = delete;
  TorchCudaDeviceCountOverrideGuard(TorchCudaDeviceCountOverrideGuard&&) =
      delete;
  auto operator=(TorchCudaDeviceCountOverrideGuard&&)
      -> TorchCudaDeviceCountOverrideGuard& = delete;

 private:
  bool active_ = false;
  std::optional<c10::DeviceIndex> previous_;
};

}  // namespace starpu_server::testing
