#include "support/output_slot_pool_test_hooks.hpp"

#include <cstdlib>
#include <utility>

namespace starpu_server {

void
OutputSlotPoolTestHook::cleanup_slot_buffers(
    OutputSlotPool::SlotInfo& slot,
    std::vector<OutputSlotPool::HostBufferInfo>& buffer_infos, size_t count)
{
  OutputSlotPool::cleanup_slot_buffers(slot, buffer_infos, count);
}

auto
OutputSlotPoolTestHook::checked_total_numel(
    size_t per_sample_numel, size_t batch_size) -> size_t
{
  return OutputSlotPool::checked_total_numel(per_sample_numel, batch_size);
}

auto
OutputSlotPoolTestHook::host_buffer_infos(
    const OutputSlotPool& pool,
    int slot_id) -> const std::vector<OutputSlotPool::HostBufferInfo>&
{
  return pool.host_buffer_infos_.at(static_cast<size_t>(slot_id));
}

void
OutputSlotPoolTestHook::free_host_buffer_for_tests(
    std::byte* ptr, const OutputSlotPool::HostBufferInfo& buffer_info)
{
  OutputSlotPool::free_host_buffer(ptr, buffer_info);
}

auto
OutputSlotPoolTestHook::starpu_vector_register_hook_ref()
    -> decltype(OutputSlotPool::starpu_vector_register_hook())
{
  return OutputSlotPool::starpu_vector_register_hook();
}

auto
OutputSlotPoolTestHook::register_failure_observer_ref()
    -> decltype(OutputSlotPool::starpu_register_failure_observer())
{
  return OutputSlotPool::starpu_register_failure_observer();
}

auto
OutputSlotPoolTestHook::host_allocator_hook_ref()
    -> decltype(OutputSlotPool::output_host_allocator_hook())
{
  return OutputSlotPool::output_host_allocator_hook();
}

auto
OutputSlotPoolTestHook::host_deallocator_hook_ref()
    -> decltype(OutputSlotPool::output_host_deallocator_hook())
{
  return OutputSlotPool::output_host_deallocator_hook();
}

auto
OutputSlotPoolTestHook::cuda_pinned_override_hook_ref()
    -> decltype(OutputSlotPool::output_cuda_pinned_override_hook())
{
  return OutputSlotPool::output_cuda_pinned_override_hook();
}

auto
OutputSlotPoolTestHook::starpu_memory_pin_hook_ref()
    -> decltype(OutputSlotPool::starpu_memory_pin_hook())
{
  return OutputSlotPool::starpu_memory_pin_hook();
}

namespace testing {

auto
set_output_starpu_vector_register_hook_for_tests(
    OutputStarpuVectorRegisterFn vector_register_hook)
    -> OutputStarpuVectorRegisterFn
{
  auto& hook = OutputSlotPoolTestHook::starpu_vector_register_hook_ref();
  const auto previous = hook;
  hook = vector_register_hook != nullptr ? vector_register_hook
                                         : &starpu_vector_data_register;
  return previous;
}

auto
set_output_register_failure_observer_for_tests(
    OutputRegisterFailureObserverFn observer) -> OutputRegisterFailureObserverFn
{
  auto& observer_hook = OutputSlotPoolTestHook::register_failure_observer_ref();
  const auto previous = observer_hook;
  observer_hook = observer;
  return previous;
}

auto
set_output_host_allocator_for_tests(OutputHostAllocatorFn allocator)
    -> OutputHostAllocatorFn
{
  auto& allocator_hook = OutputSlotPoolTestHook::host_allocator_hook_ref();
  const auto previous = allocator_hook;
  allocator_hook =
      allocator ? std::move(allocator) : OutputHostAllocatorFn{&posix_memalign};
  return previous;
}

auto
set_output_host_deallocator_for_tests(OutputHostDeallocatorFn deallocator)
    -> OutputHostDeallocatorFn
{
  auto& deallocator_hook = OutputSlotPoolTestHook::host_deallocator_hook_ref();
  const auto previous = deallocator_hook;
  deallocator_hook =
      deallocator ? std::move(deallocator)
                  : OutputHostDeallocatorFn{[](void* ptr) noexcept {
                      std::free(ptr);  // NOLINT(cppcoreguidelines-no-malloc)
                    }};
  return previous;
}

auto
set_output_cuda_pinned_override_for_tests(
    OutputCudaPinnedOverrideFn cuda_pinned_override_hook)
    -> OutputCudaPinnedOverrideFn
{
  auto& override_hook = OutputSlotPoolTestHook::cuda_pinned_override_hook_ref();
  const auto previous = override_hook;
  override_hook = std::move(cuda_pinned_override_hook);
  return previous;
}

auto
set_output_starpu_memory_pin_hook_for_tests(
    OutputStarpuMemoryPinFn memory_pin_hook) -> OutputStarpuMemoryPinFn
{
  auto& hook = OutputSlotPoolTestHook::starpu_memory_pin_hook_ref();
  const auto previous = hook;
  hook = memory_pin_hook ? std::move(memory_pin_hook)
                         : OutputStarpuMemoryPinFn{&starpu_memory_pin};
  return previous;
}

}  // namespace testing

}  // namespace starpu_server
