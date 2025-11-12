#include "torch_cuda_device_count_override.hpp"

#include <c10/cuda/CUDAFunctions.h>

#include <mutex>
#include <optional>

namespace {

auto
override_mutex() -> std::mutex&
{
  static std::mutex mutex;
  return mutex;
}

auto
override_storage() -> std::optional<c10::DeviceIndex>&
{
  static std::optional<c10::DeviceIndex> value;
  return value;
}

auto
set_override_and_get_previous(std::optional<c10::DeviceIndex> value)
    -> std::optional<c10::DeviceIndex>
{
  auto& mutex = override_mutex();
  std::lock_guard<std::mutex> lock(mutex);
  auto previous = override_storage();
  override_storage() = value;
  return previous;
}

auto
read_override() -> std::optional<c10::DeviceIndex>
{
  auto& mutex = override_mutex();
  std::lock_guard<std::mutex> lock(mutex);
  return override_storage();
}

}  // namespace

namespace starpu_server::testing {

void
set_torch_cuda_device_count_override(std::optional<c10::DeviceIndex> override)
{
  (void)set_override_and_get_previous(override);
}

void
reset_torch_cuda_device_count_override()
{
  (void)set_override_and_get_previous(std::nullopt);
}

TorchCudaDeviceCountOverrideGuard::TorchCudaDeviceCountOverrideGuard(
    std::optional<c10::DeviceIndex> override)
{
  if (override.has_value()) {
    previous_ = set_override_and_get_previous(override);
    active_ = true;
  }
}

TorchCudaDeviceCountOverrideGuard::~TorchCudaDeviceCountOverrideGuard()
{
  if (active_) {
    (void)set_override_and_get_previous(previous_);
  }
}

}  // namespace starpu_server::testing

namespace torch::cuda {

c10::DeviceIndex
device_count()
{
  if (const auto override = read_override(); override.has_value()) {
    return *override;
  }
  return static_cast<c10::DeviceIndex>(c10::cuda::device_count());
}

}  // namespace torch::cuda
