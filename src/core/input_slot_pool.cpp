#include "input_slot_pool.hpp"

#include <cuda_runtime_api.h>
#include <starpu.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdlib>
#include <format>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>

#include "slot_pool_buffer_utils.hpp"
#include "utils/logger.hpp"

namespace starpu_server {

namespace detail {

using StarpuVectorRegisterFn = decltype(&starpu_vector_data_register);
using RegisterFailureObserverFn =
    std::function<void(const InputSlotPool::SlotInfo& slot)>;

auto
starpu_vector_register_hook() -> StarpuVectorRegisterFn&
{
  static StarpuVectorRegisterFn hook = &starpu_vector_data_register;
  return hook;
}

auto
starpu_register_failure_observer() -> RegisterFailureObserverFn&
{
  static RegisterFailureObserverFn observer;
  return observer;
}

}  // namespace detail

namespace {

constexpr auto kHostBufferAlignment = std::align_val_t{64};

using detail::RegisterFailureObserverFn;
using detail::starpu_register_failure_observer;
using detail::starpu_vector_register_hook;
using detail::StarpuVectorRegisterFn;

struct ComputedInputSizes {
  size_t total_bytes = 0;
  size_t total_numel = 0;
};

struct AllocatedHostBuffer {
  std::byte* ptr = nullptr;
  InputSlotPool::HostBufferInfo info;
};

auto
alloc_host_buffer(size_t bytes, bool use_pinned, bool& cuda_pinned_out)
    -> std::byte*
{
  return detail::allocate_host_buffer_with_optional_cuda_pinning(
      bytes, use_pinned, cuda_pinned_out,
      [](size_t requested_bytes) -> std::byte* {
        void* cuda_ptr = nullptr;
        const auto err =
            cudaHostAlloc(&cuda_ptr, requested_bytes, cudaHostAllocPortable);
        if (err == cudaSuccess && cuda_ptr != nullptr) {
          return static_cast<std::byte*>(cuda_ptr);
        }
        return nullptr;
      },
      [](size_t requested_bytes) -> std::byte* {
        return static_cast<std::byte*>(
            ::operator new(requested_bytes, kHostBufferAlignment));
      });
}

struct HostBufferDeleter {
  InputSlotPool::HostBufferInfo info;

  void operator()(std::byte* ptr) const noexcept
  {
    if (ptr == nullptr) {
      return;
    }
    if (info.cuda_pinned) {
      cudaFreeHost(static_cast<void*>(ptr));
      return;
    }
    ::operator delete(static_cast<void*>(ptr), kHostBufferAlignment);
  }
};

void
free_host_buffer(
    std::byte* ptr, const InputSlotPool::HostBufferInfo& buffer_info)
{
  if (ptr == nullptr) {
    return;
  }
  detail::try_unpin_host_buffer_for_starpu(
      ptr, buffer_info, [](int result_code) {
        log_warning(std::format(
            "starpu_memory_unpin failed for input buffer: rc={}",
            std::to_string(result_code)));
      });
  HostBufferDeleter deleter{buffer_info};
  std::unique_ptr<std::byte, HostBufferDeleter> owner(ptr, deleter);
}

auto
compute_input_sizes(
    std::size_t per_sample_bytes, std::size_t per_sample_numel,
    std::size_t batch_size, std::size_t input_index) -> ComputedInputSizes
{
  return {
      .total_bytes = detail::checked_size_product(
          per_sample_bytes, batch_size,
          [input_index](
              std::size_t checked_per_sample_bytes,
              std::size_t checked_batch_size) {
            return std::format(
                "InputSlotPool: per-sample bytes ({}) times batch size ({}) "
                "exceeds size_t range for input {}",
                checked_per_sample_bytes, checked_batch_size, input_index);
          }),
      .total_numel = detail::checked_size_product(
          per_sample_numel, batch_size,
          [input_index](
              std::size_t checked_per_sample_numel,
              std::size_t checked_batch_size) {
            return std::format(
                "InputSlotPool: per-sample numel ({}) times batch size ({}) "
                "exceeds size_t range for input {}",
                checked_per_sample_numel, checked_batch_size, input_index);
          }),
  };
}

auto
allocate_and_pin_buffer(
    std::size_t bytes, bool want_pinned, int slot_id,
    std::size_t input_index) -> AllocatedHostBuffer
{
  bool cuda_pinned = false;
  std::byte* ptr = alloc_host_buffer(bytes, want_pinned, cuda_pinned);

  AllocatedHostBuffer allocation;
  allocation.ptr = ptr;
  allocation.info.cuda_pinned = cuda_pinned;
  allocation.info.bytes = bytes;

  const bool should_starpu_pin = want_pinned && !cuda_pinned;
  detail::try_pin_host_buffer_for_starpu(
      ptr, should_starpu_pin, allocation.info, &starpu_memory_pin,
      [slot_id, input_index](int pin_result) {
        log_warning(std::format(
            "starpu_memory_pin failed for input slot {}, index {}: rc={}",
            slot_id, input_index, pin_result));
      });

  return allocation;
}

void
cleanup_slot_allocations(
    InputSlotPool::SlotInfo& slot,
    std::vector<InputSlotPool::HostBufferInfo>& buffer_infos, size_t count)
{
  detail::cleanup_slot_resources(
      slot, buffer_infos, count, free_host_buffer,
      detail::SlotCleanupOrder::UnregisterThenFree,
      /*reset_buffer_info=*/false);
}

auto
register_starpu_handle_or_throw(
    std::byte* ptr, const ComputedInputSizes& sizes, at::ScalarType dtype,
    size_t input_index, InputSlotPool::SlotInfo& slot,
    std::vector<InputSlotPool::HostBufferInfo>& buffer_infos)
    -> starpu_data_handle_t
{
  starpu_data_handle_t starpu_handle = nullptr;
  starpu_vector_register_hook()(
      &starpu_handle, STARPU_MAIN_RAM, std::bit_cast<uintptr_t>(ptr),
      sizes.total_numel, element_size(dtype));
  if (starpu_handle == nullptr) {
    cleanup_slot_allocations(slot, buffer_infos, input_index + 1);
    auto& failure_observer = starpu_register_failure_observer();
    if (failure_observer) {
      failure_observer(slot);
    }
    throw StarPURegistrationException(
        "Failed to register StarPU vector handle");
  }
  return starpu_handle;
}

}  // namespace

InputSlotPool::InputSlotPool(const RuntimeConfig& opts, int slots)
{
  bmax_ = std::max(1, opts.batching.max_batch_size);

  if (!opts.model.has_value()) {
    throw std::invalid_argument("No model config provided for InputSlotPool");
  }
  const auto& inputs = opts.model->inputs;
  input_types_.reserve(inputs.size());
  per_input_numel_single_.reserve(inputs.size());
  per_input_bytes_single_.reserve(inputs.size());

  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& input_spec = inputs[i];
    input_types_.push_back(input_spec.type);
    if (input_spec.dims.size() >= 2) {
      const int64_t batch_dim = input_spec.dims[0];
      if (batch_dim <= 0) {
        throw std::invalid_argument("dims[0] (batch) must be positive");
      }
      if (batch_dim > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("dims[0] (batch) exceeds int max");
      }
      bmax_ = std::max(bmax_, static_cast<int>(batch_dim));
    }

    const size_t numel = product_dims(input_spec.dims);
    per_input_numel_single_.push_back(numel);
    const size_t elsize = element_size(input_spec.type);
    if (elsize != 0 && numel > std::numeric_limits<size_t>::max() / elsize) {
      throw std::overflow_error(std::format(
          "InputSlotPool: per-sample bytes overflow for input {}", i));
    }
    per_input_bytes_single_.push_back(numel * elsize);
  }

  allocate_pool(opts, slots);
}

InputSlotPool::~InputSlotPool()
{
  auto& slots_storage = slots();
  detail::cleanup_slots(
      slots_storage, host_buffer_infos_, free_host_buffer,
      detail::SlotCleanupOrder::UnregisterThenFree,
      /*reset_buffer_info=*/false);
}

void
InputSlotPool::allocate_pool(const RuntimeConfig& opts, int slots)
{
  auto& slots_storage = this->slots();
  auto& free_ids_storage = this->free_ids();
  detail::init_slots(
      slots, slots_storage, host_buffer_infos_, free_ids_storage,
      [this, &opts](int slot_id) {
        allocate_slot_buffers_and_register(slot_id, opts);
      });
}

void
InputSlotPool::allocate_slot_buffers_and_register(
    int slot_id, const RuntimeConfig& opts)
{
  const size_t n_in = per_input_numel_single_.size();
  auto& slot = slots().at(static_cast<size_t>(slot_id));
  slot.base_ptrs.resize(n_in);
  slot.handles.resize(n_in);
  auto& buffer_infos = host_buffer_infos_[static_cast<size_t>(slot_id)];
  buffer_infos.resize(n_in);

  const bool want_pinned = opts.devices.use_cuda;
  const auto batch_size = static_cast<size_t>(bmax_);

  try {
    for (size_t i = 0; i < n_in; ++i) {
      const auto sizes = compute_input_sizes(
          per_input_bytes_single_[i], per_input_numel_single_[i], batch_size,
          i);

      auto allocation =
          allocate_and_pin_buffer(sizes.total_bytes, want_pinned, slot_id, i);
      slot.base_ptrs[i] = allocation.ptr;
      buffer_infos[i] = allocation.info;

      slot.handles[i] = register_starpu_handle_or_throw(
          allocation.ptr, sizes, input_types_[i], i, slot, buffer_infos);
    }
  }
  catch (...) {
    cleanup_slot_allocations(slot, buffer_infos, n_in);
    throw;
  }
}

auto
InputSlotPool::host_buffer_infos(int slot_id) const
    -> const std::vector<HostBufferInfo>&
{
  return host_buffer_infos_.at(static_cast<size_t>(slot_id));
}

auto
InputSlotPool::product_dims(const std::vector<int64_t>& dims) -> size_t
{
  size_t prod = 1;
  const size_t start = dims.size() >= 2 ? 1 : 0;
  for (size_t i = start; i < dims.size(); ++i) {
    const auto dimension = dims[i];
    if (dimension <= 0) {
      throw std::invalid_argument("dims must be positive");
    }
    const auto dimension_size = static_cast<size_t>(dimension);
    if (prod > std::numeric_limits<size_t>::max() / dimension_size) {
      throw std::overflow_error("dimension product overflow");
    }
    prod *= dimension_size;
  }
  return prod;
}

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
namespace testing {
void
compute_input_sizes_for_tests(
    std::size_t per_sample_bytes, std::size_t per_sample_numel,
    std::size_t batch_size, std::size_t input_index)
{
  [[maybe_unused]] const auto sizes = compute_input_sizes(
      per_sample_bytes, per_sample_numel, batch_size, input_index);
}
}  // namespace testing
#endif  // SONAR_IGNORE_END

}  // namespace starpu_server
