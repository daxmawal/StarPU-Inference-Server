#include "support/output_slot_pool_test_hooks.hpp"

#include <cstdlib>
#include <utility>

namespace starpu_server::detail {

void cleanup_slot_buffers_for_tests(
    OutputSlotPool::SlotInfo& slot,
    std::vector<OutputSlotPool::HostBufferInfo>& buffer_infos, size_t count);

auto checked_total_numel_for_tests(size_t per_sample_numel, size_t batch_size)
    -> size_t;

void free_host_buffer_for_tests(
    std::byte* ptr, const OutputSlotPool::HostBufferInfo& buffer_info);

auto starpu_vector_register_hook_ref()
    -> decltype(&starpu_vector_data_register)&;

auto register_failure_observer_ref()
    -> void (*&)(
        const OutputSlotPool::SlotInfo&,
        const std::vector<OutputSlotPool::HostBufferInfo>&);

auto host_allocator_hook_ref() -> std::function<int(void**, size_t, size_t)>&;

auto cuda_pinned_override_hook_ref()
    -> std::function<bool(size_t, bool, bool)>&;

auto starpu_memory_pin_hook_ref() -> std::function<int(void*, size_t)>&;

}  // namespace starpu_server::detail

namespace starpu_server {

void
OutputSlotPoolTestHook::cleanup_slot_buffers(
    OutputSlotPool::SlotInfo& slot,
    std::vector<OutputSlotPool::HostBufferInfo>& buffer_infos, size_t count)
{
  detail::cleanup_slot_buffers_for_tests(slot, buffer_infos, count);
}

auto
OutputSlotPoolTestHook::checked_total_numel(
    size_t per_sample_numel, size_t batch_size) -> size_t
{
  return detail::checked_total_numel_for_tests(per_sample_numel, batch_size);
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
  detail::free_host_buffer_for_tests(ptr, buffer_info);
}

namespace testing {

auto
set_output_starpu_vector_register_hook_for_tests(
    OutputStarpuVectorRegisterFn vector_register_hook)
    -> OutputStarpuVectorRegisterFn
{
  auto& hook = detail::starpu_vector_register_hook_ref();
  const auto previous = hook;
  hook = vector_register_hook != nullptr ? vector_register_hook
                                         : &starpu_vector_data_register;
  return previous;
}

auto
set_output_register_failure_observer_for_tests(
    OutputRegisterFailureObserverFn observer) -> OutputRegisterFailureObserverFn
{
  auto& observer_hook = detail::register_failure_observer_ref();
  const auto previous = observer_hook;
  observer_hook = observer;
  return previous;
}

auto
set_output_host_allocator_for_tests(OutputHostAllocatorFn allocator)
    -> OutputHostAllocatorFn
{
  auto& allocator_hook = detail::host_allocator_hook_ref();
  const auto previous = allocator_hook;
  allocator_hook =
      allocator ? std::move(allocator) : OutputHostAllocatorFn{&posix_memalign};
  return previous;
}

auto
set_output_cuda_pinned_override_for_tests(
    OutputCudaPinnedOverrideFn cuda_pinned_override_hook)
    -> OutputCudaPinnedOverrideFn
{
  auto& override_hook = detail::cuda_pinned_override_hook_ref();
  const auto previous = override_hook;
  override_hook = std::move(cuda_pinned_override_hook);
  return previous;
}

auto
set_output_starpu_memory_pin_hook_for_tests(
    OutputStarpuMemoryPinFn memory_pin_hook) -> OutputStarpuMemoryPinFn
{
  auto& hook = detail::starpu_memory_pin_hook_ref();
  const auto previous = hook;
  hook = memory_pin_hook ? std::move(memory_pin_hook)
                         : OutputStarpuMemoryPinFn{&starpu_memory_pin};
  return previous;
}

}  // namespace testing

}  // namespace starpu_server
