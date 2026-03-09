#include "output_slot_pool.hpp"

#include <cuda_runtime_api.h>
#include <starpu.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdlib>
#include <format>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "slot_pool_buffer_utils.hpp"
#include "utils/datatype_utils.hpp"
#include "utils/exceptions.hpp"
#include "utils/logger.hpp"

namespace starpu_server {

auto
OutputSlotPool::alloc_host_buffer(
    size_t bytes, bool use_pinned, bool& cuda_pinned_out) -> HostBufferPtr
{
  return detail::allocate_host_buffer_with_optional_cuda_pinning(
      bytes, use_pinned, cuda_pinned_out,
      [use_pinned](size_t requested_bytes) -> std::byte* {
        void* raw_ptr = nullptr;
        const int err = output_cuda_host_alloc_hook()(
            &raw_ptr, requested_bytes, cudaHostAllocPortable);
        if (err != static_cast<int>(cudaSuccess) || raw_ptr == nullptr) {
          return nullptr;
        }

        bool keep_cuda_pinned = true;
        if (const auto& cuda_override = output_cuda_pinned_override_hook();
            cuda_override) {
          keep_cuda_pinned = cuda_override(requested_bytes, use_pinned, true);
        }
        if (!keep_cuda_pinned) {
          cudaFreeHost(raw_ptr);
          return nullptr;
        }

        return static_cast<std::byte*>(raw_ptr);
      },
      [](size_t requested_bytes) -> std::byte* {
        constexpr size_t kAlign = 64;
        void* raw_ptr = nullptr;
        if (int alloc_rc =
                output_host_allocator_hook()(&raw_ptr, kAlign, requested_bytes);
            alloc_rc != 0 || raw_ptr == nullptr) {
          throw std::bad_alloc();
        }
        return static_cast<std::byte*>(raw_ptr);
      });
}

auto
OutputSlotPool::checked_total_bytes(size_t per_sample_bytes, size_t batch_size)
    -> size_t
{
  return detail::checked_size_product(
      per_sample_bytes, batch_size,
      [](std::size_t checked_per_sample_bytes, std::size_t checked_batch_size) {
        return std::format(
            "OutputSlotPool: per-sample bytes ({}) times batch size ({}) "
            "exceeds size_t range",
            checked_per_sample_bytes, checked_batch_size);
      });
}

auto
OutputSlotPool::prepare_host_buffer(
    size_t per_sample_bytes, size_t batch_size, bool want_pinned, int slot_id,
    size_t output_index) -> PreparedHostBuffer
{
  PreparedHostBuffer prepared;
  prepared.info.bytes = checked_total_bytes(per_sample_bytes, batch_size);
  prepared.ptr = alloc_host_buffer(
      prepared.info.bytes, want_pinned, prepared.info.cuda_pinned);

  detail::try_pin_host_buffer_for_starpu(
      prepared.ptr, want_pinned, prepared.info, starpu_memory_pin_hook(),
      [slot_id, output_index](int pin_result) {
        log_warning(std::format(
            "starpu_memory_pin failed for output slot {}, index {}: rc={}",
            slot_id, output_index, pin_result));
      });

  return prepared;
}

void
OutputSlotPool::HostBufferDeleter::operator()(HostBufferPtr ptr) const
{
  if (ptr == nullptr) {
    return;
  }
  output_host_deallocator_hook()(static_cast<void*>(ptr));
}

auto
OutputSlotPool::starpu_vector_register_hook() -> OutputStarpuVectorRegisterHook&
{
  static OutputStarpuVectorRegisterHook hook = &starpu_vector_data_register;
  return hook;
}

auto
OutputSlotPool::starpu_register_failure_observer()
    -> OutputRegisterFailureObserverHook&
{
  static OutputRegisterFailureObserverHook observer = nullptr;
  return observer;
}

auto
OutputSlotPool::output_host_allocator_hook() -> OutputHostAllocatorHook&
{
  static OutputHostAllocatorHook allocator = &posix_memalign;
  return allocator;
}

auto
OutputSlotPool::output_cuda_host_alloc_hook() -> OutputCudaHostAllocHook&
{
  static OutputCudaHostAllocHook cuda_host_allocator =
      [](void** ptr, size_t size, unsigned int flags) -> int {
    return static_cast<int>(cudaHostAlloc(ptr, size, flags));
  };
  return cuda_host_allocator;
}

auto
OutputSlotPool::output_host_deallocator_hook() -> OutputHostDeallocatorHook&
{
  static OutputHostDeallocatorHook deallocator = [](void* ptr) noexcept {
    if (ptr == nullptr) {
      return;
    }
    using FreeDeleter = decltype(&std::free);
    std::unique_ptr<void, FreeDeleter> guard(ptr, &std::free);
  };
  return deallocator;
}

auto
OutputSlotPool::output_cuda_pinned_override_hook()
    -> OutputCudaPinnedOverrideHook&
{
  static OutputCudaPinnedOverrideHook override_hook;
  return override_hook;
}

auto
OutputSlotPool::starpu_memory_pin_hook() -> OutputStarpuMemoryPinHook&
{
  static OutputStarpuMemoryPinHook pin_hook = &starpu_memory_pin;
  return pin_hook;
}

void
OutputSlotPool::cleanup_slot_buffers(
    SlotInfo& slot, std::vector<HostBufferInfo>& buffer_infos, size_t count)
{
  detail::cleanup_slot_resources(
      slot, buffer_infos, count, free_host_buffer,
      detail::SlotCleanupOrder::FreeThenUnregister,
      /*reset_buffer_info=*/true);
}

auto
OutputSlotPool::checked_total_numel(size_t per_sample_numel, size_t batch_size)
    -> size_t
{
  return detail::checked_size_product(
      per_sample_numel, batch_size,
      [](std::size_t checked_per_sample_numel, std::size_t checked_batch_size) {
        return std::format(
            "OutputSlotPool: per-sample numel ({}) times batch size ({}) "
            "exceeds size_t range",
            checked_per_sample_numel, checked_batch_size);
      });
}

void
OutputSlotPool::free_host_buffer(
    std::byte* ptr, const HostBufferInfo& buffer_info)
{
  if (ptr == nullptr) {
    return;
  }
  HostBufferOwner managed_ptr(ptr);
  detail::try_unpin_host_buffer_for_starpu(
      managed_ptr.get(), buffer_info, [](int unpin_rc) {
        log_warning(std::format(
            "starpu_memory_unpin failed for output buffer: rc={}",
            std::to_string(unpin_rc)));
        (void)cudaGetLastError();
      });
  if (buffer_info.cuda_pinned) {
    HostBufferPtr cuda_ptr = managed_ptr.release();
    cudaError_t cuda_rc = cudaFreeHost(static_cast<void*>(cuda_ptr));
    if (cuda_rc != cudaSuccess) {
      log_warning(std::format(
          "cudaFreeHost failed for output buffer: rc={}",
          std::to_string(static_cast<int>(cuda_rc))));
    }
  }
}

OutputSlotPool::OutputSlotPool(const RuntimeConfig& opts, int slots)
{
  bmax_ = std::max(1, opts.batching.max_batch_size);

  if (!opts.model.has_value()) {
    throw std::invalid_argument("No model config provided for OutputSlotPool");
  }
  const auto& outputs = opts.model->outputs;
  output_types_.reserve(outputs.size());
  per_output_numel_single_.reserve(outputs.size());
  per_output_bytes_single_.reserve(outputs.size());

  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto& output_desc = outputs[i];
    output_types_.push_back(output_desc.type);
    if (output_desc.dims.size() >= 2) {
      const int64_t batch_dim = output_desc.dims[0];
      if (batch_dim <= 0) {
        throw std::invalid_argument("dims[0] (batch) must be positive");
      }
      if (batch_dim > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("dims[0] (batch) exceeds int max");
      }
      bmax_ = std::max(bmax_, static_cast<int>(batch_dim));
    }

    const size_t numel = product_dims(output_desc.dims);
    per_output_numel_single_.push_back(numel);
    const size_t elsize = element_size(output_desc.type);
    if (elsize != 0 && numel > std::numeric_limits<size_t>::max() / elsize) {
      throw std::overflow_error(std::format(
          "OutputSlotPool: per-sample bytes overflow for output {}", i));
    }
    per_output_bytes_single_.push_back(numel * elsize);
  }

  allocate_pool(opts, slots);
}

OutputSlotPool::~OutputSlotPool()
{
  detail::cleanup_pool_slots(slots(), host_buffer_infos_, free_host_buffer);
}

void
OutputSlotPool::allocate_pool(const RuntimeConfig& opts, int slots)
{
  detail::init_pool_slots(
      slots, this->slots(), host_buffer_infos_, this->free_ids(),
      [this, &opts](int slot_id) {
        allocate_slot_buffers_and_register(slot_id, opts);
      });
}

void
OutputSlotPool::allocate_slot_buffers_and_register(
    int slot_id, const RuntimeConfig& opts)
{
  const size_t num_outputs = per_output_numel_single_.size();
  auto& slot = slots().at(static_cast<size_t>(slot_id));
  slot.base_ptrs.assign(num_outputs, nullptr);
  slot.handles.assign(num_outputs, nullptr);
  auto& buffer_infos = host_buffer_infos_[static_cast<size_t>(slot_id)];
  buffer_infos.assign(num_outputs, HostBufferInfo{});

  const bool want_pinned = opts.devices.use_cuda;

  const auto batch_size = static_cast<size_t>(bmax_);

  try {
    for (size_t i = 0; i < num_outputs; ++i) {
      const auto prepared_buffer = prepare_host_buffer(
          per_output_bytes_single_[i], batch_size, want_pinned, slot_id, i);
      slot.base_ptrs[i] = prepared_buffer.ptr;
      buffer_infos[i] = prepared_buffer.info;

      const size_t total_numel =
          checked_total_numel(per_output_numel_single_[i], batch_size);
      starpu_data_handle_t handle = nullptr;
      starpu_vector_register_hook()(
          &handle, STARPU_MAIN_RAM,
          std::bit_cast<uintptr_t>(prepared_buffer.ptr), total_numel,
          element_size(output_types_[i]));
      if (handle == nullptr) {
        cleanup_slot_buffers(slot, buffer_infos, i + 1);
        if (starpu_register_failure_observer() != nullptr) {
          starpu_register_failure_observer()(slot, buffer_infos);
        }
        throw OutputSlotRegistrationError(
            "Failed to register StarPU vector handle for output");
      }
      slot.handles[i] = handle;
    }
  }
  catch (...) {
    cleanup_slot_buffers(slot, buffer_infos, num_outputs);
    throw;
  }
}

auto
OutputSlotPool::product_dims(const std::vector<int64_t>& dims) -> size_t
{
  size_t product = 1;
  const size_t start_dim_index = dims.size() >= 2 ? 1 : 0;
  for (size_t i = start_dim_index; i < dims.size(); ++i) {
    const auto dimension = dims[i];
    if (dimension <= 0) {
      throw std::invalid_argument("dimensions must be positive");
    }
    const auto dimension_unsigned = static_cast<size_t>(dimension);
    if (product > std::numeric_limits<size_t>::max() / dimension_unsigned) {
      throw std::overflow_error("dimension product overflow");
    }
    product *= dimension_unsigned;
  }
  return product;
}

}  // namespace starpu_server
