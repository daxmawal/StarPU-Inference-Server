#pragma once

#include "core/output_slot_pool.hpp"

namespace starpu_server {

struct OutputSlotPoolTestHook {
  static void cleanup_slot_buffers(
      OutputSlotPool::SlotInfo& slot,
      std::vector<OutputSlotPool::HostBufferInfo>& buffer_infos, size_t count);
  static size_t checked_total_numel(size_t per_sample_numel, size_t batch_size);
  static auto host_buffer_infos(const OutputSlotPool& pool, int slot_id)
      -> const std::vector<OutputSlotPool::HostBufferInfo>&;
  static void free_host_buffer_for_tests(
      void* ptr, const OutputSlotPool::HostBufferInfo& buffer_info);
};

namespace testing {

using OutputStarpuVectorRegisterFn = decltype(&starpu_vector_data_register);
using OutputRegisterFailureObserverFn = void (*)(
    const OutputSlotPool::SlotInfo& slot,
    const std::vector<OutputSlotPool::HostBufferInfo>& buffer_infos);
using OutputHostAllocatorFn = int (*)(void**, size_t alignment, size_t size);
using OutputCudaPinnedOverrideFn =
    bool (*)(size_t bytes, bool use_pinned, bool default_cuda_pinned);
using OutputStarpuMemoryPinFn = int (*)(void* ptr, size_t size);

auto set_output_starpu_vector_register_hook_for_tests(
    OutputStarpuVectorRegisterFn fn) -> OutputStarpuVectorRegisterFn;

auto set_output_register_failure_observer_for_tests(
    OutputRegisterFailureObserverFn observer)
    -> OutputRegisterFailureObserverFn;

auto set_output_host_allocator_for_tests(OutputHostAllocatorFn allocator)
    -> OutputHostAllocatorFn;

auto set_output_cuda_pinned_override_for_tests(OutputCudaPinnedOverrideFn fn)
    -> OutputCudaPinnedOverrideFn;

auto set_output_starpu_memory_pin_hook_for_tests(OutputStarpuMemoryPinFn fn)
    -> OutputStarpuMemoryPinFn;

}  // namespace testing

}  // namespace starpu_server
