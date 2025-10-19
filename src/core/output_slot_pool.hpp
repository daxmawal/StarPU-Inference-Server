// Reusable StarPU output slot pool
#pragma once

#include <starpu.h>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "utils/runtime_config.hpp"

namespace starpu_server {

class OutputSlotPool {
 public:
  struct SlotInfo {
    int id = -1;
    std::vector<std::byte*> base_ptrs;
    std::vector<size_t> per_output_numel_single;
    std::vector<size_t> per_output_bytes_single;
    std::vector<starpu_data_handle_t> handles;
  };

  struct HostBufferInfo {
    bool cuda_pinned = false;
    bool starpu_pinned = false;
    int starpu_pin_rc = 0;
    size_t bytes = 0;
  };

  OutputSlotPool(const RuntimeConfig& opts, int slots);
  ~OutputSlotPool();

  OutputSlotPool(const OutputSlotPool&) = delete;
  auto operator=(const OutputSlotPool&) -> OutputSlotPool& = delete;
  OutputSlotPool(OutputSlotPool&&) = delete;
  auto operator=(OutputSlotPool&&) -> OutputSlotPool& = delete;

  auto acquire() -> int;
  [[nodiscard]] auto try_acquire() -> std::optional<int>;
  void release(int slot_id);

  // Accessors
  [[nodiscard]] auto slot_info(int slot_id) const -> const SlotInfo&;
  [[nodiscard]] auto handles(int slot_id) const
      -> const std::vector<starpu_data_handle_t>&;
  [[nodiscard]] auto base_ptrs(int slot_id) const
      -> const std::vector<std::byte*>&;
  [[nodiscard]] auto max_batch_size() const -> int { return bmax_; }
  [[nodiscard]] auto num_outputs() const -> size_t
  {
    return per_output_numel_single_.size();
  }

 private:
  friend struct OutputSlotPoolTestHook;

  using OutputStarpuVectorRegisterHook = decltype(&starpu_vector_data_register);
  using OutputRegisterFailureObserverHook =
      void (*)(const SlotInfo&, const std::vector<HostBufferInfo>&);
  using OutputHostAllocatorHook = std::function<int(void**, size_t, size_t)>;
  using OutputCudaPinnedOverrideHook = std::function<bool(size_t, bool, bool)>;
  using OutputHostDeallocatorHook = std::function<void(void*)>;
  using OutputStarpuMemoryPinHook = std::function<int(void*, size_t)>;

  void allocate_pool(const RuntimeConfig& opts, int slots);
  void allocate_slot_buffers_and_register(
      int slot_id, const RuntimeConfig& opts);
  static auto product_dims(const std::vector<int64_t>& dims) -> size_t;
  using HostBufferPtr = std::byte*;
  struct HostBufferDeleter {
    void operator()(HostBufferPtr ptr) const;
  };
  using HostBufferOwner = std::unique_ptr<std::byte, HostBufferDeleter>;
  struct PreparedHostBuffer {
    HostBufferPtr ptr = nullptr;
    HostBufferInfo info;
  };

  static auto alloc_host_buffer(
      size_t bytes, bool use_pinned, bool& cuda_pinned) -> HostBufferPtr;
  static auto prepare_host_buffer(
      size_t per_sample_bytes, size_t batch_size, bool want_pinned, int slot_id,
      size_t output_index) -> PreparedHostBuffer;
  static auto checked_total_bytes(size_t per_sample_bytes, size_t batch_size)
      -> size_t;
  static void cleanup_slot_buffers(
      SlotInfo& slot, std::vector<HostBufferInfo>& buffer_infos, size_t count);
  static auto checked_total_numel(size_t per_sample_numel, size_t batch_size)
      -> size_t;
  static void free_host_buffer(
      std::byte* ptr, const HostBufferInfo& buffer_info);
  static auto starpu_vector_register_hook() -> OutputStarpuVectorRegisterHook&;
  static auto starpu_register_failure_observer()
      -> OutputRegisterFailureObserverHook&;
  static auto output_host_allocator_hook() -> OutputHostAllocatorHook&;
  static auto output_cuda_pinned_override_hook()
      -> OutputCudaPinnedOverrideHook&;
  static auto output_host_deallocator_hook() -> OutputHostDeallocatorHook&;
  static auto starpu_memory_pin_hook() -> OutputStarpuMemoryPinHook&;

  std::vector<size_t> per_output_numel_single_;
  std::vector<size_t> per_output_bytes_single_;
  std::vector<at::ScalarType> output_types_;
  std::vector<SlotInfo> slots_;
  int bmax_ = 1;
  std::vector<std::vector<HostBufferInfo>> host_buffer_infos_;
  std::vector<int> free_ids_;
  std::mutex mtx_;
  std::condition_variable cv_;
};

}  // namespace starpu_server
