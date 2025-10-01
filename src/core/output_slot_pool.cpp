#include "output_slot_pool.hpp"

#include <cuda_runtime_api.h>
#include <starpu.h>

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "utils/logger.hpp"

namespace starpu_server {

namespace {
constexpr size_t kMaxSizeT = std::numeric_limits<size_t>::max();

testing::OutputStarpuVectorRegisterFn g_starpu_vector_register_hook =
    &starpu_vector_data_register;
testing::OutputRegisterFailureObserverFn g_starpu_register_failure_observer =
    nullptr;

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
  int alloc_rc = posix_memalign(&ptr, kAlign, bytes);
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
        starpu_memory_pin(prepared.ptr, prepared.info.bytes);
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
    if (slot.handles[j] != nullptr) {
      starpu_data_unregister(slot.handles[j]);
      slot.handles[j] = nullptr;
    }
    if (slot.base_ptrs[j] != nullptr) {
      free_host_buffer(slot.base_ptrs[j], buffer_infos[j]);
      slot.base_ptrs[j] = nullptr;
      buffer_infos[j] = {};
    }
  }
}
}  // namespace

void
OutputSlotPoolTestHook::cleanup_slot_buffers(
    OutputSlotPool::SlotInfo& slot,
    std::vector<OutputSlotPool::HostBufferInfo>& buffer_infos, size_t count)
{
  cleanup_slot_buffers_impl(slot, buffer_infos, count);
}

namespace testing {

auto
set_output_starpu_vector_register_hook_for_tests(
    OutputStarpuVectorRegisterFn fn) -> OutputStarpuVectorRegisterFn
{
  const auto previous = g_starpu_vector_register_hook;
  g_starpu_vector_register_hook =
      fn != nullptr ? fn : &starpu_vector_data_register;
  return previous;
}

auto
set_output_register_failure_observer_for_tests(
    OutputRegisterFailureObserverFn observer) -> OutputRegisterFailureObserverFn
{
  const auto previous = g_starpu_register_failure_observer;
  g_starpu_register_failure_observer = observer;
  return previous;
}

}  // namespace testing

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
    g_starpu_vector_register_hook(
        &handle, STARPU_MAIN_RAM, std::bit_cast<uintptr_t>(prepared_buffer.ptr),
        total_numel, element_size(output_types_[i]));
    if (handle == nullptr) {
      cleanup_slot_buffers_impl(slot, buffer_infos, i + 1);
      if (g_starpu_register_failure_observer != nullptr) {
        g_starpu_register_failure_observer(slot, buffer_infos);
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
