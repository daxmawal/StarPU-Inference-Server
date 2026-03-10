// Common host-buffer and lifecycle helpers for slot pools
#pragma once

#include <starpu.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>

namespace starpu_server::detail {

enum class SlotCleanupOrder : std::uint8_t {
  UnregisterThenFree,
  FreeThenUnregister,
};

template <typename MakeOverflowMessageFn>
auto
checked_size_product(
    std::size_t per_sample_value, std::size_t batch_size,
    MakeOverflowMessageFn&& make_overflow_message) -> std::size_t
{
  if (const std::size_t kMaxSizeT = std::numeric_limits<std::size_t>::max();
      per_sample_value != 0 && batch_size > kMaxSizeT / per_sample_value) {
    throw std::overflow_error(std::invoke(
        std::forward<MakeOverflowMessageFn>(make_overflow_message),
        per_sample_value, batch_size));
  }
  return per_sample_value * batch_size;
}

template <typename TryCudaPinnedAllocFn, typename FallbackAllocFn>
auto
allocate_host_buffer_with_optional_cuda_pinning(
    std::size_t bytes, bool use_pinned, bool& cuda_pinned_out,
    TryCudaPinnedAllocFn&& try_cuda_pinned_alloc,
    FallbackAllocFn&& fallback_alloc) -> std::byte*
{
  cuda_pinned_out = false;
  if (use_pinned) {
    if (std::byte* ptr = std::invoke(
            std::forward<TryCudaPinnedAllocFn>(try_cuda_pinned_alloc), bytes);
        ptr != nullptr) {
      cuda_pinned_out = true;
      return ptr;
    }
  }

  return std::invoke(std::forward<FallbackAllocFn>(fallback_alloc), bytes);
}

template <typename HostBufferInfo, typename StarpuMemoryPinFn, typename WarnFn>
void
try_pin_host_buffer_for_starpu(
    std::byte* ptr, bool want_pinned, HostBufferInfo& buffer_info,
    StarpuMemoryPinFn&& starpu_memory_pin_fn, WarnFn&& warn_on_pin_failure)
{
  if (!want_pinned || buffer_info.cuda_pinned || ptr == nullptr) {
    return;
  }

  buffer_info.starpu_pin_rc = std::invoke(
      std::forward<StarpuMemoryPinFn>(starpu_memory_pin_fn),
      static_cast<void*>(ptr), buffer_info.bytes);
  if (buffer_info.starpu_pin_rc == 0) {
    buffer_info.starpu_pinned = true;
    return;
  }

  std::invoke(
      std::forward<WarnFn>(warn_on_pin_failure), buffer_info.starpu_pin_rc);
}

template <typename HostBufferInfo, typename WarnFn>
void
try_unpin_host_buffer_for_starpu(
    std::byte* ptr, const HostBufferInfo& buffer_info,
    WarnFn&& warn_on_unpin_failure)
{
  if (ptr == nullptr || !buffer_info.starpu_pinned) {
    return;
  }

  const int unpin_rc =
      starpu_memory_unpin(static_cast<void*>(ptr), buffer_info.bytes);
  if (unpin_rc != 0) {
    std::invoke(std::forward<WarnFn>(warn_on_unpin_failure), unpin_rc);
  }
}

template <typename SlotInfo, typename BufferInfos, typename FreeBufferFn>
void
cleanup_slot_resources(
    SlotInfo& slot, BufferInfos& buffer_infos, std::size_t count,
    const FreeBufferFn& free_buffer_fn, SlotCleanupOrder cleanup_order,
    bool reset_buffer_info)
{
  const std::size_t safe_count = std::min(
      {count, slot.base_ptrs.size(), slot.handles.size(), buffer_infos.size()});

  auto unregister_handle = [&slot](std::size_t idx) {
    if (slot.handles[idx] != nullptr) {
      starpu_data_unregister(slot.handles[idx]);
      slot.handles[idx] = nullptr;
    }
  };

  auto free_buffer = [&](std::size_t idx) {
    if (slot.base_ptrs[idx] != nullptr) {
      std::invoke(free_buffer_fn, slot.base_ptrs[idx], buffer_infos[idx]);
      slot.base_ptrs[idx] = nullptr;
    }
    if (reset_buffer_info) {
      buffer_infos[idx] = {};
    }
  };

  for (std::size_t idx = 0; idx < safe_count; ++idx) {
    if (cleanup_order == SlotCleanupOrder::UnregisterThenFree) {
      unregister_handle(idx);
      free_buffer(idx);
    } else {
      free_buffer(idx);
      unregister_handle(idx);
    }
  }
}

template <
    typename SlotsStorage, typename BufferInfosStorage, typename FreeIdsStorage,
    typename AllocateSlotFn>
void
init_slots(
    int requested_slots, SlotsStorage& slots_storage,
    BufferInfosStorage& host_buffer_infos, FreeIdsStorage& free_ids_storage,
    AllocateSlotFn allocate_slot_fn)
{
  auto slot_count = requested_slots;
  if (slot_count <= 0) {
    const auto workers = static_cast<int>(starpu_worker_get_count());
    slot_count = std::max(2, workers);
  }

  const auto slot_count_size = static_cast<std::size_t>(slot_count);
  slots_storage.reserve(slot_count_size);
  host_buffer_infos.reserve(slot_count_size);
  free_ids_storage.reserve(slot_count_size);
  slots_storage.resize(slot_count_size);
  host_buffer_infos.resize(slot_count_size);

  for (int i = 0; i < slot_count; ++i) {
    const auto slot_index = static_cast<std::size_t>(i);
    slots_storage[slot_index].id = i;
    std::invoke(allocate_slot_fn, i);
    free_ids_storage.push_back(i);
  }
}

template <
    typename SlotsStorage, typename BufferInfosStorage, typename FreeBufferFn>
void
cleanup_slots(
    SlotsStorage& slots_storage, BufferInfosStorage& host_buffer_infos,
    FreeBufferFn free_buffer_fn,
    SlotCleanupOrder cleanup_order = SlotCleanupOrder::UnregisterThenFree,
    bool reset_buffer_info = false)
{
  const std::size_t slot_count =
      std::min(slots_storage.size(), host_buffer_infos.size());
  for (std::size_t slot_index = 0; slot_index < slot_count; ++slot_index) {
    auto& slot = slots_storage[slot_index];
    auto& buffer_infos = host_buffer_infos[slot_index];
    cleanup_slot_resources(
        slot, buffer_infos, slot.base_ptrs.size(), free_buffer_fn,
        cleanup_order, reset_buffer_info);
  }
}

template <
    typename SlotsStorage, typename BufferInfosStorage, typename FreeIdsStorage,
    typename AllocateSlotFn>
void
init_pool_slots(
    int requested_slots, SlotsStorage& slots_storage,
    BufferInfosStorage& host_buffer_infos, FreeIdsStorage& free_ids_storage,
    AllocateSlotFn&& allocate_slot_fn)
{
  init_slots(
      requested_slots, slots_storage, host_buffer_infos, free_ids_storage,
      std::forward<AllocateSlotFn>(allocate_slot_fn));
}

template <
    typename SlotsStorage, typename BufferInfosStorage, typename FreeBufferFn>
void
cleanup_pool_slots(
    SlotsStorage& slots_storage, BufferInfosStorage& host_buffer_infos,
    FreeBufferFn&& free_buffer_fn)
{
  cleanup_slots(
      slots_storage, host_buffer_infos,
      std::forward<FreeBufferFn>(free_buffer_fn),
      SlotCleanupOrder::UnregisterThenFree,
      /*reset_buffer_info=*/false);
}

}  // namespace starpu_server::detail
