#include "output_slot_pool.hpp"

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
#include <stdexcept>
#include <string>
#include <utility>

#include "utils/datatype_utils.hpp"
#include "utils/exceptions.hpp"
#include "utils/logger.hpp"

namespace starpu_server {

namespace {
constexpr size_t kMaxSizeT = std::numeric_limits<size_t>::max();
}  // namespace

auto
OutputSlotPool::alloc_host_buffer(
    size_t bytes, bool use_pinned, bool& cuda_pinned_out) -> HostBufferPtr
{
  HostBufferPtr ptr = nullptr;
  cuda_pinned_out = false;
  if (use_pinned) {
    HostBufferPtr cuda_ptr = nullptr;
    const auto err = cudaHostAlloc(
        reinterpret_cast<void**>(&cuda_ptr), bytes, cudaHostAllocPortable);
    if (err == cudaSuccess && cuda_ptr != nullptr) {
      bool keep_cuda_pinned = true;
      auto& cuda_override = output_cuda_pinned_override_hook();
      if (cuda_override) {
        keep_cuda_pinned = cuda_override(bytes, use_pinned, true);
      }
      if (keep_cuda_pinned) {
        cuda_pinned_out = true;
        return cuda_ptr;
      }
      cudaFreeHost(static_cast<void*>(cuda_ptr));
    }
  }
  constexpr size_t kAlign = 64;

  int alloc_rc = output_host_allocator_hook()(
      reinterpret_cast<void**>(&ptr), kAlign, bytes);
  if (alloc_rc != 0 || ptr == nullptr) {
    throw std::bad_alloc();
  }

  return ptr;
}

auto
OutputSlotPool::checked_total_bytes(size_t per_sample_bytes, size_t batch_size)
    -> size_t
{
  if (per_sample_bytes != 0 && batch_size > kMaxSizeT / per_sample_bytes) {
    throw std::overflow_error(std::format(
        "OutputSlotPool: per-sample bytes ({}) times batch size ({}) exceeds "
        "size_t range",
        per_sample_bytes, batch_size));
  }
  return per_sample_bytes * batch_size;
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

  if (want_pinned && !prepared.info.cuda_pinned) {
    prepared.info.starpu_pin_rc =
        starpu_memory_pin_hook()(prepared.ptr, prepared.info.bytes);
    if (prepared.info.starpu_pin_rc == 0) {
      prepared.info.starpu_pinned = true;
    } else {
      log_warning(std::format(
          "starpu_memory_pin failed for output slot {}, index {}: rc={}",
          slot_id, output_index, prepared.info.starpu_pin_rc));
    }
  }

  return prepared;
}

void
OutputSlotPool::HostBufferDeleter::operator()(HostBufferPtr ptr) const
{
  std::free(static_cast<void*>(ptr));
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

auto
OutputSlotPool::checked_total_numel(size_t per_sample_numel, size_t batch_size)
    -> size_t
{
  if (per_sample_numel != 0 && batch_size > kMaxSizeT / per_sample_numel) {
    throw std::overflow_error(std::format(
        "OutputSlotPool: per-sample numel ({}) times batch size ({}) exceeds "
        "size_t range",
        per_sample_numel, batch_size));
  }
  return per_sample_numel * batch_size;
}

void
OutputSlotPool::free_host_buffer(
    std::byte* ptr, const HostBufferInfo& buffer_info)
{
  if (ptr == nullptr) {
    return;
  }
  HostBufferOwner managed_ptr(ptr);
  if (buffer_info.starpu_pinned) {
    const int unpin_rc = starpu_memory_unpin(
        static_cast<void*>(managed_ptr.get()), buffer_info.bytes);
    if (unpin_rc != 0) {
      log_warning(std::format(
          "starpu_memory_unpin failed for output buffer: rc={}",
          std::to_string(unpin_rc)));
      (void)cudaGetLastError();
    }
  }
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
  auto requested_slots = slots;
  if (requested_slots <= 0) {
    auto workers = static_cast<int>(starpu_worker_get_count());
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

  const bool want_pinned = opts.devices.use_cuda;

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

auto
OutputSlotPool::acquire() -> int
{
  std::unique_lock pool_lock(mtx_);
  cv_.wait(pool_lock, [this] { return !free_ids_.empty(); });
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
OutputSlotPool::base_ptrs(int slot_id) const -> const std::vector<std::byte*>&
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
