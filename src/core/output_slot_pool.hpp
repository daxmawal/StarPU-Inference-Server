// Reusable StarPU output slot pool
#pragma once

#include <starpu.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "core/slot_pool_base.hpp"
#include "utils/runtime_config.hpp"

namespace starpu_server {

class OutputSlotPool : public SlotPoolBase<SlotPoolSlot> {
 public:
  using SlotInfo = SlotPoolSlot;

  struct HostBufferInfo {
    bool cuda_pinned = false;
    bool starpu_pinned = false;
    int starpu_pin_rc = 0;
    size_t bytes = 0;
  };

  using HostBufferPtr = std::byte*;
  using StarpuVectorRegisterFn = decltype(&starpu_vector_data_register);
  using RegisterFailureObserverFn =
      void (*)(const SlotInfo&, const std::vector<HostBufferInfo>&);
  using HostAllocatorFn = std::function<int(HostBufferPtr*, size_t, size_t)>;
  using CudaHostAllocFn = std::function<int(void**, size_t, unsigned int)>;
  using CudaPinnedOverrideFn = std::function<bool(size_t, bool, bool)>;
  using HostDeallocatorFn = std::function<void(void*)>;
  using StarpuMemoryPinFn = std::function<int(void*, size_t)>;

  struct Dependencies {
    StarpuVectorRegisterFn starpu_vector_register = nullptr;
    RegisterFailureObserverFn register_failure_observer = nullptr;
    HostAllocatorFn host_allocator;
    CudaHostAllocFn cuda_host_alloc;
    CudaPinnedOverrideFn cuda_pinned_override;
    HostDeallocatorFn host_deallocator;
    StarpuMemoryPinFn starpu_memory_pin;
  };

  OutputSlotPool(const RuntimeConfig& opts, int slots);
  OutputSlotPool(
      const RuntimeConfig& opts, int slots, Dependencies dependencies);
  ~OutputSlotPool();

  OutputSlotPool(const OutputSlotPool&) = delete;
  auto operator=(const OutputSlotPool&) -> OutputSlotPool& = delete;
  OutputSlotPool(OutputSlotPool&&) = delete;
  auto operator=(OutputSlotPool&&) -> OutputSlotPool& = delete;

  using SlotPoolBase<SlotPoolSlot>::acquire;
  using SlotPoolBase<SlotPoolSlot>::try_acquire;
  using SlotPoolBase<SlotPoolSlot>::release;
  using SlotPoolBase<SlotPoolSlot>::slot_info;
  using SlotPoolBase<SlotPoolSlot>::handles;
  using SlotPoolBase<SlotPoolSlot>::base_ptrs;

  [[nodiscard]] auto max_batch_size() const -> int { return bmax_; }
  [[nodiscard]] auto num_outputs() const -> size_t
  {
    return per_output_numel_single_.size();
  }

 private:
  friend struct OutputSlotPoolTestHook;

  void allocate_pool(const RuntimeConfig& opts, int slots);
  void allocate_slot_buffers_and_register(
      int slot_id, const RuntimeConfig& opts);
  static auto product_dims(const std::vector<int64_t>& dims) -> size_t;
  struct PreparedHostBuffer {
    HostBufferPtr ptr = nullptr;
    HostBufferInfo info;
  };

  static auto alloc_host_buffer(
      size_t bytes, bool use_pinned, bool& cuda_pinned,
      const Dependencies& dependencies) -> HostBufferPtr;
  static auto prepare_host_buffer(
      size_t per_sample_bytes, size_t batch_size, bool want_pinned, int slot_id,
      size_t output_index,
      const Dependencies& dependencies) -> PreparedHostBuffer;
  static auto checked_total_bytes(size_t per_sample_bytes, size_t batch_size)
      -> size_t;
  static void cleanup_slot_buffers(
      SlotInfo& slot, std::vector<HostBufferInfo>& buffer_infos, size_t count,
      const Dependencies& dependencies);
  static auto checked_total_numel(size_t per_sample_numel, size_t batch_size)
      -> size_t;
  static void free_host_buffer(
      std::byte* ptr, const HostBufferInfo& buffer_info,
      const Dependencies& dependencies);

  std::vector<size_t> per_output_numel_single_;
  std::vector<size_t> per_output_bytes_single_;
  std::vector<at::ScalarType> output_types_;
  int bmax_ = 1;
  std::vector<std::vector<HostBufferInfo>> host_buffer_infos_;
  Dependencies dependencies_;
};

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
namespace testing {
auto output_slot_pool_default_dependencies_for_tests()
    -> OutputSlotPool::Dependencies&;
}  // namespace testing
#endif  // SONAR_IGNORE_END

}  // namespace starpu_server
