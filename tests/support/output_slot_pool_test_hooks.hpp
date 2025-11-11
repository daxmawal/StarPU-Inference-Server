#pragma once

#include <functional>

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
      std::byte* ptr, const OutputSlotPool::HostBufferInfo& buffer_info);
  static void invoke_host_buffer_deleter(std::byte* ptr);
  static auto starpu_vector_register_hook_ref()
      -> decltype(OutputSlotPool::starpu_vector_register_hook());
  static auto register_failure_observer_ref()
      -> decltype(OutputSlotPool::starpu_register_failure_observer());
  static auto host_allocator_hook_ref()
      -> decltype(OutputSlotPool::output_host_allocator_hook());
  static auto host_deallocator_hook_ref()
      -> decltype(OutputSlotPool::output_host_deallocator_hook());
  static auto cuda_pinned_override_hook_ref()
      -> decltype(OutputSlotPool::output_cuda_pinned_override_hook());
  static auto starpu_memory_pin_hook_ref()
      -> decltype(OutputSlotPool::starpu_memory_pin_hook());
};

namespace testing {

using OutputStarpuVectorRegisterFn = decltype(&starpu_vector_data_register);
using OutputRegisterFailureObserverFn = void (*)(
    const OutputSlotPool::SlotInfo& slot,
    const std::vector<OutputSlotPool::HostBufferInfo>& buffer_infos);
using OutputHostAllocatorFn = std::function<int(void**, size_t, size_t)>;
using OutputHostDeallocatorFn = std::function<void(void*)>;
using OutputCudaPinnedOverrideFn = std::function<bool(size_t, bool, bool)>;
using OutputStarpuMemoryPinFn = std::function<int(void*, size_t)>;

auto set_output_starpu_vector_register_hook_for_tests(
    OutputStarpuVectorRegisterFn vector_register_hook)
    -> OutputStarpuVectorRegisterFn;

auto set_output_register_failure_observer_for_tests(
    OutputRegisterFailureObserverFn observer)
    -> OutputRegisterFailureObserverFn;

auto set_output_host_allocator_for_tests(OutputHostAllocatorFn allocator)
    -> OutputHostAllocatorFn;

auto set_output_host_deallocator_for_tests(OutputHostDeallocatorFn deallocator)
    -> OutputHostDeallocatorFn;

auto set_output_cuda_pinned_override_for_tests(
    OutputCudaPinnedOverrideFn cuda_pinned_override_hook)
    -> OutputCudaPinnedOverrideFn;

auto set_output_starpu_memory_pin_hook_for_tests(
    OutputStarpuMemoryPinFn memory_pin_hook) -> OutputStarpuMemoryPinFn;

}  // namespace testing

}  // namespace starpu_server
