#include "output_slot_pool.hpp"

#include <cuda_runtime_api.h>
#include <starpu.h>

#include <algorithm>
#include <bit>
#include <cassert>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "utils/logger.hpp"

namespace starpu_server {

namespace {
static auto
alloc_host_buffer(size_t bytes, bool use_pinned, bool& pinned_out) -> void*
{
  void* ptr = nullptr;
  pinned_out = false;
  if (use_pinned) {
    const auto err = cudaHostAlloc(&ptr, bytes, cudaHostAllocPortable);
    if (err == cudaSuccess && ptr != nullptr) {
      pinned_out = true;
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
free_host_buffer(void* ptr, bool pinned)
{
  if (!ptr)
    return;
  if (pinned) {
    cudaFreeHost(ptr);
  } else {
    free(ptr);
  }
}
}  // namespace

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

  for (const auto& t : outputs) {
    output_types_.push_back(t.type);
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
    per_output_numel_single_.push_back(numel);
    const size_t elsize = element_size(t.type);
    per_output_bytes_single_.push_back(numel * elsize);
  }

  allocate_pool(opts, slots);
}

OutputSlotPool::~OutputSlotPool()
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
      free_host_buffer(slot.base_ptrs[i], pinned_flags_[s][i]);
      slot.base_ptrs[i] = nullptr;
    }
  }
}

void
OutputSlotPool::allocate_pool(const RuntimeConfig& opts, int slots)
{
  int k = slots;
  if (k <= 0) {
    int workers = static_cast<int>(starpu_worker_get_count());
    k = std::max(2, workers);
  }
  slots_.reserve(static_cast<size_t>(k));
  pinned_flags_.reserve(static_cast<size_t>(k));
  free_ids_.reserve(static_cast<size_t>(k));
  slots_.resize(static_cast<size_t>(k));
  pinned_flags_.resize(static_cast<size_t>(k));
  for (int i = 0; i < k; ++i) {
    slots_[static_cast<size_t>(i)].id = i;
    allocate_slot_buffers_and_register(i, opts);
    free_ids_.push_back(i);
  }
}

void
OutputSlotPool::allocate_slot_buffers_and_register(
    int slot_id, const RuntimeConfig& opts)
{
  const size_t n_out = per_output_numel_single_.size();
  auto& slot = slots_.at(static_cast<size_t>(slot_id));
  slot.base_ptrs.resize(n_out);
  slot.handles.resize(n_out);
  pinned_flags_[static_cast<size_t>(slot_id)].resize(n_out);

  const bool want_pinned = opts.use_cuda;

  for (size_t i = 0; i < n_out; ++i) {
    const size_t bytes =
        per_output_bytes_single_[i] * static_cast<size_t>(bmax_);
    bool pinned = false;
    void* ptr = alloc_host_buffer(bytes, want_pinned, pinned);
    slot.base_ptrs[i] = ptr;
    pinned_flags_[static_cast<size_t>(slot_id)][i] = pinned;

    const size_t total_numel =
        per_output_numel_single_[i] * static_cast<size_t>(bmax_);
    starpu_data_handle_t h = nullptr;
    starpu_vector_data_register(
        &h, STARPU_MAIN_RAM, std::bit_cast<uintptr_t>(ptr), total_numel,
        element_size(output_types_[i]));
    if (!h) {
      for (size_t j = 0; j <= i; ++j) {
        if (slot.handles[j]) {
          starpu_data_unregister(slot.handles[j]);
          slot.handles[j] = nullptr;
        }
        if (slot.base_ptrs[j]) {
          free_host_buffer(
              slot.base_ptrs[j],
              pinned_flags_[static_cast<size_t>(slot_id)][j]);
          slot.base_ptrs[j] = nullptr;
        }
      }
      throw std::runtime_error(
          "Failed to register StarPU vector handle for output");
    }
    slot.handles[i] = h;
  }
}

auto
OutputSlotPool::acquire() -> int
{
  std::unique_lock lk(mtx_);
  cv_.wait(lk, [&] { return !free_ids_.empty(); });
  const int id = free_ids_.back();
  free_ids_.pop_back();
  return id;
}

auto
OutputSlotPool::try_acquire() -> std::optional<int>
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
OutputSlotPool::release(int slot_id)
{
  {
    const std::scoped_lock lk(mtx_);
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

size_t
OutputSlotPool::product_dims(const std::vector<int64_t>& dims)
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
