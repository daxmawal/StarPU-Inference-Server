#include "output_slot_pool.hpp"

#ifdef UNIT_TEST
#include "../../tests/support/output_slot_pool_test_hooks.hpp"
#endif

#include <cuda_runtime_api.h>
#include <starpu.h>

#include <algorithm>
#include <bit>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "utils/datatype_utils.hpp"
#include "utils/logger.hpp"

namespace starpu_server {

namespace {
constexpr size_t kMaxSizeT = std::numeric_limits<size_t>::max();

using OutputStarpuVectorRegisterHook = decltype(&starpu_vector_data_register);
using OutputRegisterFailureObserverHook = void (*)(
    const OutputSlotPool::SlotInfo&,
    const std::vector<OutputSlotPool::HostBufferInfo>&);
using OutputHostAllocatorHook = int (*)(void**, size_t alignment, size_t size);
using OutputCudaPinnedOverrideHook =
    bool (*)(size_t bytes, bool use_pinned, bool default_cuda_pinned);
using OutputStarpuMemoryPinHook = int (*)(void* ptr, size_t size);

auto
starpu_vector_register_hook() -> OutputStarpuVectorRegisterHook&
{
  static OutputStarpuVectorRegisterHook hook = &starpu_vector_data_register;
  return hook;
}

auto
starpu_register_failure_observer() -> OutputRegisterFailureObserverHook&
{
  static OutputRegisterFailureObserverHook observer = nullptr;
  return observer;
}

auto
output_host_allocator_hook() -> OutputHostAllocatorHook&
{
  static OutputHostAllocatorHook allocator = &posix_memalign;
  return allocator;
}

auto
output_cuda_pinned_override_hook() -> OutputCudaPinnedOverrideHook&
{
  static OutputCudaPinnedOverrideHook override_hook = nullptr;
  return override_hook;
}

auto
starpu_memory_pin_hook() -> OutputStarpuMemoryPinHook&
{
  static OutputStarpuMemoryPinHook hook = &starpu_memory_pin;
  return hook;
}

auto
alloc_host_buffer(size_t bytes, bool use_pinned, bool& cuda_pinned_out) -> void*
{
  void* ptr = nullptr;
  cuda_pinned_out = false;
  if (use_pinned) {
    void* cuda_ptr = nullptr;
    const auto err = cudaHostAlloc(&cuda_ptr, bytes, cudaHostAllocPortable);
    if (err == cudaSuccess && cuda_ptr != nullptr) {
      bool keep_cuda_pinned = true;
      if (output_cuda_pinned_override_hook() != nullptr) {
        keep_cuda_pinned =
            output_cuda_pinned_override_hook()(bytes, use_pinned, true);
      }
      if (keep_cuda_pinned) {
        cuda_pinned_out = true;
        return cuda_ptr;
      }
      cudaFreeHost(cuda_ptr);
    }
  }
  constexpr size_t kAlign = 64;
  int alloc_rc = output_host_allocator_hook()(&ptr, kAlign, bytes);
  if (alloc_rc != 0 || ptr == nullptr) {
    throw std::bad_alloc();
  }
  return ptr;
}

void
free_host_buffer(void* ptr, const OutputSlotPool::HostBufferInfo& buffer_info)
{
  if (ptr == nullptr) {
    return;
  }
  std::unique_ptr<void, decltype(&std::free)> managed_ptr(ptr, &std::free);
  if (buffer_info.starpu_pinned) {
    const int unpin_rc = starpu_memory_unpin(ptr, buffer_info.bytes);
    if (unpin_rc != 0) {
      log_warning(
          "starpu_memory_unpin failed for output buffer: rc=" +
          std::to_string(unpin_rc));
      (void)cudaGetLastError();
    }
  }
  if (buffer_info.cuda_pinned) {
    void* raw_ptr = managed_ptr.release();
    cudaError_t cuda_rc = cudaFreeHost(raw_ptr);
    if (cuda_rc != cudaSuccess) {
      log_warning(
          "cudaFreeHost failed for output buffer: rc=" +
          std::to_string(static_cast<int>(cuda_rc)));
    }
  }
}

auto
checked_total_bytes(size_t per_sample_bytes, size_t batch_size) -> size_t
{
  if (per_sample_bytes != 0 && batch_size > kMaxSizeT / per_sample_bytes) {
    throw std::overflow_error(
        "OutputSlotPool: per-sample bytes (" +
        std::to_string(per_sample_bytes) + ") times batch size (" +
        std::to_string(batch_size) + ") exceeds size_t range");
  }
  return per_sample_bytes * batch_size;
}

auto
checked_total_numel(size_t per_sample_numel, size_t batch_size) -> size_t
{
  if (per_sample_numel != 0 && batch_size > kMaxSizeT / per_sample_numel) {
    throw std::overflow_error(
        "OutputSlotPool: per-sample numel (" +
        std::to_string(per_sample_numel) + ") times batch size (" +
        std::to_string(batch_size) + ") exceeds size_t range");
  }
  return per_sample_numel * batch_size;
}

struct PreparedHostBuffer {
  void* ptr = nullptr;
  OutputSlotPool::HostBufferInfo info;
};

auto
prepare_host_buffer(
    size_t per_sample_bytes, size_t batch_size, bool want_pinned, int slot_id,
    size_t output_index) -> PreparedHostBuffer
{
  PreparedHostBuffer prepared;
  prepared.info.bytes = checked_total_bytes(per_sample_bytes, batch_size);
  prepared.ptr = alloc_host_buffer(
      prepared.info.bytes, want_pinned, prepared.info.cuda_pinned);

  if (want_pinned && !prepared.info.cuda_pinned) {
    prepared.info.starpu_pin_rc =
        starpu_memory_pin_hook()(prepared.ptr, prepared.info.bytes);
    if (prepared.info.starpu_pin_rc == 0) {
      prepared.info.starpu_pinned = true;
    } else {
      log_warning(
          "starpu_memory_pin failed for output slot " +
          std::to_string(slot_id) + ", index " + std::to_string(output_index) +
          ": rc=" + std::to_string(prepared.info.starpu_pin_rc));
    }
  }

  return prepared;
}

void
cleanup_slot_buffers_impl(
    OutputSlotPool::SlotInfo& slot,
    std::vector<OutputSlotPool::HostBufferInfo>& buffer_infos, size_t count)
{
  for (size_t j = 0; j < count; ++j) {
    if (slot.base_ptrs[j] != nullptr) {
      free_host_buffer(slot.base_ptrs[j], buffer_infos[j]);
      slot.base_ptrs[j] = nullptr;
      buffer_infos[j] = {};
    }
    if (slot.handles[j] != nullptr) {
      starpu_data_unregister(slot.handles[j]);
      slot.handles[j] = nullptr;
    }
  }
}

#ifdef UNIT_TEST
auto
call_checked_total_numel(size_t per_sample_numel, size_t batch_size) -> size_t
{
  return checked_total_numel(per_sample_numel, batch_size);
}
#endif
}  // namespace

#ifdef UNIT_TEST

void
OutputSlotPoolTestHook::cleanup_slot_buffers(
    OutputSlotPool::SlotInfo& slot,
    std::vector<OutputSlotPool::HostBufferInfo>& buffer_infos, size_t count)
{
  cleanup_slot_buffers_impl(slot, buffer_infos, count);
}

auto
OutputSlotPoolTestHook::checked_total_numel(
    size_t per_sample_numel, size_t batch_size) -> size_t
{
  return call_checked_total_numel(per_sample_numel, batch_size);
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
    void* ptr, const OutputSlotPool::HostBufferInfo& buffer_info)
{
  free_host_buffer(ptr, buffer_info);
}

namespace testing {

auto
set_output_starpu_vector_register_hook_for_tests(
    OutputStarpuVectorRegisterFn vector_register_hook)
    -> OutputStarpuVectorRegisterFn
{
  auto& hook = starpu_vector_register_hook();
  const auto previous = hook;
  hook = vector_register_hook != nullptr ? vector_register_hook
                                         : &starpu_vector_data_register;
  return previous;
}

auto
set_output_register_failure_observer_for_tests(
    OutputRegisterFailureObserverFn observer) -> OutputRegisterFailureObserverFn
{
  auto& observer_hook = starpu_register_failure_observer();
  const auto previous = observer_hook;
  observer_hook = observer;
  return previous;
}

auto
set_output_host_allocator_for_tests(OutputHostAllocatorFn allocator)
    -> OutputHostAllocatorFn
{
  auto& allocator_hook = output_host_allocator_hook();
  const auto previous = allocator_hook;
  allocator_hook = allocator != nullptr ? allocator : &posix_memalign;
  return previous;
}

auto
set_output_cuda_pinned_override_for_tests(
    OutputCudaPinnedOverrideFn cuda_pinned_override_hook)
    -> OutputCudaPinnedOverrideFn
{
  auto& override_hook = output_cuda_pinned_override_hook();
  const auto previous = override_hook;
  override_hook = cuda_pinned_override_hook;
  return previous;
}

auto
set_output_starpu_memory_pin_hook_for_tests(
    OutputStarpuMemoryPinFn memory_pin_hook) -> OutputStarpuMemoryPinFn
{
  auto& hook = starpu_memory_pin_hook();
  const auto previous = hook;
  hook = memory_pin_hook != nullptr ? memory_pin_hook : &starpu_memory_pin;
  return previous;
}

}  // namespace testing

#endif  // UNIT_TEST

OutputSlotPool::OutputSlotPool(const RuntimeConfig& opts, int slots)
{
  bmax_ = std::max(1, opts.max_batch_size);

  if (opts.models.empty()) {
    throw std::invalid_argument("No model config provided for OutputSlotPool");
  }
  const auto& outputs = opts.models[0].outputs;
  output_types_.reserve(outputs.size());
  per_output_numel_single_.reserve(outputs.size());
  per_output_bytes_single_.reserve(outputs.size());

  for (const auto& output_desc : outputs) {
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
    per_output_bytes_single_.push_back(numel * elsize);
  }

  allocate_pool(opts, slots);
}

OutputSlotPool::~OutputSlotPool()
{
  for (size_t slot_index = 0; slot_index < slots_.size(); ++slot_index) {
    auto& slot = slots_[slot_index];
    for (auto& handle : slot.handles) {
      if (handle != nullptr) {
        starpu_data_unregister(handle);
        handle = nullptr;
      }
    }
    for (size_t i = 0; i < slot.base_ptrs.size(); ++i) {
      free_host_buffer(slot.base_ptrs[i], host_buffer_infos_[slot_index][i]);
      slot.base_ptrs[i] = nullptr;
    }
  }
}

void
OutputSlotPool::allocate_pool(const RuntimeConfig& opts, int slots)
{
  int requested_slots = slots;
  if (requested_slots <= 0) {
    int workers = static_cast<int>(starpu_worker_get_count());
    requested_slots = std::max(2, workers);
  }
  slots_.reserve(static_cast<size_t>(requested_slots));
  host_buffer_infos_.reserve(static_cast<size_t>(requested_slots));
  free_ids_.reserve(static_cast<size_t>(requested_slots));
  slots_.resize(static_cast<size_t>(requested_slots));
  host_buffer_infos_.resize(static_cast<size_t>(requested_slots));
  for (int i = 0; i < requested_slots; ++i) {
    slots_[static_cast<size_t>(i)].id = i;
    allocate_slot_buffers_and_register(i, opts);
    free_ids_.push_back(i);
  }
}

void
OutputSlotPool::allocate_slot_buffers_and_register(
    int slot_id, const RuntimeConfig& opts)
{
  const size_t num_outputs = per_output_numel_single_.size();
  auto& slot = slots_.at(static_cast<size_t>(slot_id));
  slot.base_ptrs.assign(num_outputs, nullptr);
  slot.handles.assign(num_outputs, nullptr);
  auto& buffer_infos = host_buffer_infos_[static_cast<size_t>(slot_id)];
  buffer_infos.assign(num_outputs, HostBufferInfo{});

  const bool want_pinned = opts.use_cuda;

  const auto batch_size = static_cast<size_t>(bmax_);

  for (size_t i = 0; i < num_outputs; ++i) {
    const auto prepared_buffer = prepare_host_buffer(
        per_output_bytes_single_[i], batch_size, want_pinned, slot_id, i);
    slot.base_ptrs[i] = prepared_buffer.ptr;
    buffer_infos[i] = prepared_buffer.info;

    const size_t total_numel =
        checked_total_numel(per_output_numel_single_[i], batch_size);
    starpu_data_handle_t handle = nullptr;
    starpu_vector_register_hook()(
        &handle, STARPU_MAIN_RAM, std::bit_cast<uintptr_t>(prepared_buffer.ptr),
        total_numel, element_size(output_types_[i]));
    if (handle == nullptr) {
      cleanup_slot_buffers_impl(slot, buffer_infos, i + 1);
      if (starpu_register_failure_observer() != nullptr) {
        starpu_register_failure_observer()(slot, buffer_infos);
      }
      throw std::runtime_error(
          "Failed to register StarPU vector handle for output");
    }
    slot.handles[i] = handle;
  }
}

auto
OutputSlotPool::acquire() -> int
{
  std::unique_lock<std::mutex> pool_lock(mtx_);
  cv_.wait(pool_lock, [&] { return !free_ids_.empty(); });
  const int slot_id_value = free_ids_.back();
  free_ids_.pop_back();
  return slot_id_value;
}

auto
OutputSlotPool::try_acquire() -> std::optional<int>
{
  std::scoped_lock<std::mutex> pool_lock(mtx_);
  if (free_ids_.empty()) {
    return std::nullopt;
  }
  const int slot_id_value = free_ids_.back();
  free_ids_.pop_back();
  return slot_id_value;
}

void
OutputSlotPool::release(int slot_id)
{
  {
    const std::scoped_lock<std::mutex> pool_lock(mtx_);
    free_ids_.push_back(slot_id);
  }
  cv_.notify_one();
}

auto
OutputSlotPool::slot_info(int slot_id) const -> const SlotInfo&
{
  return slots_.at(static_cast<size_t>(slot_id));
}

auto
OutputSlotPool::handles(int slot_id) const
    -> const std::vector<starpu_data_handle_t>&
{
  return slots_.at(static_cast<size_t>(slot_id)).handles;
}

auto
OutputSlotPool::base_ptrs(int slot_id) const -> const std::vector<void*>&
{
  return slots_.at(static_cast<size_t>(slot_id)).base_ptrs;
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
