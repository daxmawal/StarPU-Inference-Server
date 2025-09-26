#include "input_slot_pool.hpp"

#include <cuda_runtime_api.h>
#include <starpu.h>

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#include "utils/logger.hpp"

namespace starpu_server {

namespace {

struct ComputedInputSizes {
  size_t total_bytes = 0;
  size_t total_numel = 0;
};

struct AllocatedHostBuffer {
  void* ptr = nullptr;
  InputSlotPool::HostBufferInfo info;
};

auto
alloc_host_buffer(size_t bytes, bool use_pinned, bool& cuda_pinned_out) -> void*
{
  void* ptr = nullptr;
  cuda_pinned_out = false;
  if (use_pinned) {
    const auto err = cudaHostAlloc(&ptr, bytes, cudaHostAllocPortable);
    if (err == cudaSuccess && ptr != nullptr) {
      cuda_pinned_out = true;
      return ptr;
    }
  }
  constexpr size_t kAlign = 64;
  int result_code = posix_memalign(&ptr, kAlign, bytes);
  if (result_code != 0 || ptr == nullptr) {
    throw std::bad_alloc();
  }
  return ptr;
}

void
free_host_buffer(void* ptr, const InputSlotPool::HostBufferInfo& buffer_info)
{
  if (ptr == nullptr) {
    return;
  }
  if (buffer_info.starpu_pinned) {
    const int result_code = starpu_memory_unpin(ptr, buffer_info.bytes);
    if (result_code != 0) {
      log_warning(
          "starpu_memory_unpin failed for input buffer: rc=" +
          std::to_string(result_code));
    }
  }
  if (buffer_info.cuda_pinned) {
    cudaFreeHost(ptr);
  } else {
    free(ptr);
  }
}

auto
compute_input_sizes(
    size_t per_sample_bytes, size_t per_sample_numel, size_t batch_size,
    size_t input_index) -> ComputedInputSizes
{
  constexpr size_t kMaxSizeT = std::numeric_limits<size_t>::max();
  if (per_sample_bytes != 0 && batch_size > kMaxSizeT / per_sample_bytes) {
    throw std::overflow_error(
        "InputSlotPool: per-sample bytes (" + std::to_string(per_sample_bytes) +
        ") times batch size (" + std::to_string(batch_size) +
        ") exceeds size_t range for input " + std::to_string(input_index));
  }
  if (per_sample_numel != 0 && batch_size > kMaxSizeT / per_sample_numel) {
    throw std::overflow_error(
        "InputSlotPool: per-sample numel (" + std::to_string(per_sample_numel) +
        ") times batch size (" + std::to_string(batch_size) +
        ") exceeds size_t range for input " + std::to_string(input_index));
  }
  return {
      .total_bytes = per_sample_bytes * batch_size,
      .total_numel = per_sample_numel * batch_size};
}

auto
allocate_and_pin_buffer(
    size_t bytes, bool want_pinned, int slot_id,
    size_t input_index) -> AllocatedHostBuffer
{
  bool cuda_pinned = false;
  void* ptr = alloc_host_buffer(bytes, want_pinned, cuda_pinned);

  AllocatedHostBuffer allocation;
  allocation.ptr = ptr;
  allocation.info.cuda_pinned = cuda_pinned;
  allocation.info.bytes = bytes;

  const bool should_starpu_pin = want_pinned && !cuda_pinned;
  if (should_starpu_pin) {
    const int pin_result = starpu_memory_pin(ptr, bytes);
    allocation.info.starpu_pin_rc = pin_result;
    if (pin_result == 0) {
      allocation.info.starpu_pinned = true;
    } else {
      log_warning(
          "starpu_memory_pin failed for input slot " + std::to_string(slot_id) +
          ", index " + std::to_string(input_index) +
          ": rc=" + std::to_string(pin_result));
    }
  }

  return allocation;
}

void
cleanup_slot_allocations(
    InputSlotPool::SlotInfo& slot,
    std::vector<InputSlotPool::HostBufferInfo>& buffer_infos, size_t count)
{
  for (size_t idx = 0; idx < count; ++idx) {
    if (slot.handles[idx] != nullptr) {
      starpu_data_unregister(slot.handles[idx]);
      slot.handles[idx] = nullptr;
    }
    if (slot.base_ptrs[idx] != nullptr) {
      free_host_buffer(slot.base_ptrs[idx], buffer_infos[idx]);
      slot.base_ptrs[idx] = nullptr;
    }
  }
}

auto
register_starpu_handle_or_throw(
    void* ptr, const ComputedInputSizes& sizes, at::ScalarType dtype,
    size_t input_index, InputSlotPool::SlotInfo& slot,
    std::vector<InputSlotPool::HostBufferInfo>& buffer_infos)
    -> starpu_data_handle_t
{
  starpu_data_handle_t starpu_handle = nullptr;
  starpu_vector_data_register(
      &starpu_handle, STARPU_MAIN_RAM, std::bit_cast<uintptr_t>(ptr),
      sizes.total_numel, element_size(dtype));
  if (starpu_handle == nullptr) {
    cleanup_slot_allocations(slot, buffer_infos, input_index + 1);
    throw std::runtime_error("Failed to register StarPU vector handle");
  }
  return starpu_handle;
}

}  // namespace

InputSlotPool::InputSlotPool(const RuntimeConfig& opts, int slots)
{
  bmax_ = std::max(1, opts.max_batch_size);

  if (opts.models.empty()) {
    throw std::invalid_argument("No model config provided for InputSlotPool");
  }
  const auto& inputs = opts.models[0].inputs;
  input_types_.reserve(inputs.size());
  per_input_numel_single_.reserve(inputs.size());
  per_input_bytes_single_.reserve(inputs.size());

  for (const auto& input_spec : inputs) {
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
    per_input_bytes_single_.push_back(numel * elsize);
  }

  allocate_pool(opts, slots);
}

InputSlotPool::~InputSlotPool()
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
InputSlotPool::allocate_pool(const RuntimeConfig& opts, int slots)
{
  int slot_count = slots;
  if (slot_count <= 0) {
    int workers = static_cast<int>(starpu_worker_get_count());
    slot_count = std::max(2, workers);
  }
  slots_.reserve(static_cast<size_t>(slot_count));
  host_buffer_infos_.reserve(static_cast<size_t>(slot_count));
  free_ids_.reserve(static_cast<size_t>(slot_count));
  slots_.resize(static_cast<size_t>(slot_count));
  host_buffer_infos_.resize(static_cast<size_t>(slot_count));
  for (int i = 0; i < slot_count; ++i) {
    slots_[static_cast<size_t>(i)].id = i;
    allocate_slot_buffers_and_register(i, opts);
    free_ids_.push_back(i);
  }
}

void
InputSlotPool::allocate_slot_buffers_and_register(
    int slot_id, const RuntimeConfig& opts)
{
  const size_t n_in = per_input_numel_single_.size();
  auto& slot = slots_.at(static_cast<size_t>(slot_id));
  slot.base_ptrs.resize(n_in);
  slot.handles.resize(n_in);
  auto& buffer_infos = host_buffer_infos_[static_cast<size_t>(slot_id)];
  buffer_infos.resize(n_in);

  const bool want_pinned = opts.use_cuda;
  const auto batch_size = static_cast<size_t>(bmax_);

  for (size_t i = 0; i < n_in; ++i) {
    const auto sizes = compute_input_sizes(
        per_input_bytes_single_[i], per_input_numel_single_[i], batch_size, i);

    auto allocation =
        allocate_and_pin_buffer(sizes.total_bytes, want_pinned, slot_id, i);
    slot.base_ptrs[i] = allocation.ptr;
    buffer_infos[i] = allocation.info;

    slot.handles[i] = register_starpu_handle_or_throw(
        allocation.ptr, sizes, input_types_[i], i, slot, buffer_infos);
  }
}

auto
InputSlotPool::acquire() -> int
{
  std::unique_lock lock(mtx_);
  cv_.wait(lock, [&] { return !free_ids_.empty(); });
  const int slot_id = free_ids_.back();
  free_ids_.pop_back();
  return slot_id;
}

auto
InputSlotPool::try_acquire() -> std::optional<int>
{
  std::scoped_lock lock(mtx_);
  if (free_ids_.empty()) {
    return std::nullopt;
  }
  const int slot_id = free_ids_.back();
  free_ids_.pop_back();
  return slot_id;
}

void
InputSlotPool::release(int slot_id)
{
  {
    const std::scoped_lock lock(mtx_);
    free_ids_.push_back(slot_id);
  }
  cv_.notify_one();
}

auto
InputSlotPool::slot_info(int slot_id) const -> const SlotInfo&
{
  return slots_.at(static_cast<size_t>(slot_id));
}

auto
InputSlotPool::handles(int slot_id) const
    -> const std::vector<starpu_data_handle_t>&
{
  return slots_.at(static_cast<size_t>(slot_id)).handles;
}

auto
InputSlotPool::base_ptrs(int slot_id) const -> const std::vector<void*>&
{
  return slots_.at(static_cast<size_t>(slot_id)).base_ptrs;
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

}  // namespace starpu_server
