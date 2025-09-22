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

static auto
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
  int rc = posix_memalign(&ptr, kAlign, bytes);
  if (rc != 0 || ptr == nullptr) {
    throw std::bad_alloc();
  }
  return ptr;
}

static void
free_host_buffer(void* ptr, const InputSlotPool::HostBufferInfo& buffer_info)
{
  if (!ptr)
    return;
  if (buffer_info.starpu_pinned) {
    const int rc = starpu_memory_unpin(ptr, buffer_info.bytes);
    if (rc != 0) {
      log_warning(
          "starpu_memory_unpin failed for input buffer: rc=" +
          std::to_string(rc));
    }
  }
  if (buffer_info.cuda_pinned) {
    cudaFreeHost(ptr);
  } else {
    free(ptr);
  }
}

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

  for (const auto& t : inputs) {
    input_types_.push_back(t.type);
    if (t.dims.size() >= 2) {
      const int64_t batch_dim = t.dims[0];
      if (batch_dim <= 0) {
        throw std::invalid_argument("dims[0] (batch) must be positive");
      }
      if (batch_dim > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("dims[0] (batch) exceeds int max");
      }
      bmax_ = std::max(bmax_, static_cast<int>(batch_dim));
    }

    const size_t numel = product_dims(t.dims);
    per_input_numel_single_.push_back(numel);
    const size_t elsize = element_size(t.type);
    per_input_bytes_single_.push_back(numel * elsize);
  }

  allocate_pool(opts, slots);
}

InputSlotPool::~InputSlotPool()
{
  for (size_t s = 0; s < slots_.size(); ++s) {
    auto& slot = slots_[s];
    for (auto& h : slot.handles) {
      if (h) {
        starpu_data_unregister(h);
        h = nullptr;
      }
    }
    for (size_t i = 0; i < slot.base_ptrs.size(); ++i) {
      free_host_buffer(slot.base_ptrs[i], host_buffer_infos_[s][i]);
      slot.base_ptrs[i] = nullptr;
    }
  }
}

void
InputSlotPool::allocate_pool(const RuntimeConfig& opts, int slots)
{
  int k = slots;
  if (k <= 0) {
    int workers = static_cast<int>(starpu_worker_get_count());
    k = std::max(2, workers);
  }
  slots_.reserve(static_cast<size_t>(k));
  host_buffer_infos_.reserve(static_cast<size_t>(k));
  free_ids_.reserve(static_cast<size_t>(k));
  slots_.resize(static_cast<size_t>(k));
  host_buffer_infos_.resize(static_cast<size_t>(k));
  for (int i = 0; i < k; ++i) {
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

  for (size_t i = 0; i < n_in; ++i) {
    const size_t bytes =
        per_input_bytes_single_[i] * static_cast<size_t>(bmax_);
    bool cuda_pinned = false;
    void* ptr = alloc_host_buffer(bytes, want_pinned, cuda_pinned);
    slot.base_ptrs[i] = ptr;
    auto& info = buffer_infos[i];
    info.cuda_pinned = cuda_pinned;
    info.bytes = bytes;
    info.starpu_pinned = false;
    info.starpu_pin_rc = 0;
    const bool should_starpu_pin = want_pinned && !cuda_pinned;
    if (should_starpu_pin) {
      const int rc = starpu_memory_pin(ptr, bytes);
      info.starpu_pin_rc = rc;
      if (rc == 0) {
        info.starpu_pinned = true;
      } else {
        log_warning(
            "starpu_memory_pin failed for input slot " +
            std::to_string(slot_id) + ", index " + std::to_string(i) +
            ": rc=" + std::to_string(rc));
      }
    }

    const size_t total_numel =
        per_input_numel_single_[i] * static_cast<size_t>(bmax_);
    starpu_data_handle_t h = nullptr;
    starpu_vector_data_register(
        &h, STARPU_MAIN_RAM, std::bit_cast<uintptr_t>(ptr), total_numel,
        element_size(input_types_[i]));
    if (!h) {
      for (size_t j = 0; j <= i; ++j) {
        if (slot.handles[j]) {
          starpu_data_unregister(slot.handles[j]);
          slot.handles[j] = nullptr;
        }
        if (slot.base_ptrs[j]) {
          free_host_buffer(slot.base_ptrs[j], buffer_infos[j]);
          slot.base_ptrs[j] = nullptr;
        }
      }
      throw std::runtime_error("Failed to register StarPU vector handle");
    }
    slot.handles[i] = h;
  }
}

auto
InputSlotPool::acquire() -> int
{
  std::unique_lock lk(mtx_);
  cv_.wait(lk, [&] { return !free_ids_.empty(); });
  const int id = free_ids_.back();
  free_ids_.pop_back();
  return id;
}

auto
InputSlotPool::try_acquire() -> std::optional<int>
{
  std::scoped_lock lk(mtx_);
  if (free_ids_.empty()) {
    return std::nullopt;
  }
  const int id = free_ids_.back();
  free_ids_.pop_back();
  return id;
}

void
InputSlotPool::release(int slot_id)
{
  {
    const std::scoped_lock lk(mtx_);
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
    const auto d = dims[i];
    if (d <= 0) {
      throw std::invalid_argument("dims must be positive");
    }
    const auto du = static_cast<size_t>(d);
    if (prod > std::numeric_limits<size_t>::max() / du) {
      throw std::overflow_error("dimension product overflow");
    }
    prod *= du;
  }
  return prod;
}

}  // namespace starpu_server
