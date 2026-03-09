// Reusable StarPU input slot pool
#pragma once

#include <starpu.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "core/slot_pool_base.hpp"
#include "utils/exceptions.hpp"
#include "utils/runtime_config.hpp"

namespace starpu_server {

class InputSlotPool : public SlotPoolBase<SlotPoolSlot> {
 public:
  using SlotInfo = SlotPoolSlot;

  struct HostBufferInfo {
    bool cuda_pinned = false;
    bool starpu_pinned = false;
    int starpu_pin_rc = 0;
    size_t bytes = 0;
  };

  using StarpuVectorRegisterFn = decltype(&starpu_vector_data_register);
  using RegisterFailureObserverFn = std::function<void(const SlotInfo& slot)>;

  struct Dependencies {
    StarpuVectorRegisterFn starpu_vector_register = nullptr;
    RegisterFailureObserverFn register_failure_observer;
  };

  InputSlotPool(const RuntimeConfig& opts, int slots);
  InputSlotPool(
      const RuntimeConfig& opts, int slots, Dependencies dependencies);
  ~InputSlotPool();

  InputSlotPool(const InputSlotPool&) = delete;
  auto operator=(const InputSlotPool&) -> InputSlotPool& = delete;
  InputSlotPool(InputSlotPool&&) = delete;
  auto operator=(InputSlotPool&&) -> InputSlotPool& = delete;

  using SlotPoolBase<SlotPoolSlot>::acquire;
  using SlotPoolBase<SlotPoolSlot>::try_acquire;
  using SlotPoolBase<SlotPoolSlot>::release;

  using SlotPoolBase<SlotPoolSlot>::slot_info;
  using SlotPoolBase<SlotPoolSlot>::handles;
  using SlotPoolBase<SlotPoolSlot>::base_ptrs;

  [[nodiscard]] auto host_buffer_infos(int slot_id) const
      -> const std::vector<HostBufferInfo>&;
  [[nodiscard]] auto max_batch_size() const -> int { return bmax_; }
  [[nodiscard]] auto num_inputs() const -> size_t
  {
    return per_input_numel_single_.size();
  }

 private:
  void allocate_pool(const RuntimeConfig& opts, int slots);
  void allocate_slot_buffers_and_register(
      int slot_id, const RuntimeConfig& opts);
  static auto product_dims(const std::vector<int64_t>& dims) -> size_t;

  std::vector<size_t> per_input_numel_single_;
  std::vector<size_t> per_input_bytes_single_;
  std::vector<at::ScalarType> input_types_;

  int bmax_ = 1;

  std::vector<std::vector<HostBufferInfo>> host_buffer_infos_;
  Dependencies dependencies_;
};

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
namespace testing {
auto input_slot_pool_default_dependencies_for_tests()
    -> InputSlotPool::Dependencies&;
}  // namespace testing
#endif  // SONAR_IGNORE_END

}  // namespace starpu_server
