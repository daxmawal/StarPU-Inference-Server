// Reusable StarPU input slot pool
#pragma once

#include <starpu.h>

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <vector>

#include "utils/exceptions.hpp"
#include "utils/runtime_config.hpp"

namespace starpu_server {

class InputSlotPool {
 public:
  struct SlotInfo {
    int id = -1;
    std::vector<std::byte*> base_ptrs;
    std::vector<size_t> per_input_numel_single;
    std::vector<size_t> per_input_bytes_single;
    std::vector<starpu_data_handle_t> handles;
  };

  struct HostBufferInfo {
    bool cuda_pinned = false;
    bool starpu_pinned = false;
    int starpu_pin_rc = 0;
    size_t bytes = 0;
  };

  InputSlotPool(const RuntimeConfig& opts, int slots);
  ~InputSlotPool();

  InputSlotPool(const InputSlotPool&) = delete;
  auto operator=(const InputSlotPool&) -> InputSlotPool& = delete;
  InputSlotPool(InputSlotPool&&) = delete;
  auto operator=(InputSlotPool&&) -> InputSlotPool& = delete;

  auto acquire() -> int;
  [[nodiscard]] auto try_acquire() -> std::optional<int>;
  void release(int slot_id);

  [[nodiscard]] auto slot_info(int slot_id) const -> const SlotInfo&;
  [[nodiscard]] auto handles(int slot_id) const
      -> const std::vector<starpu_data_handle_t>&;
  [[nodiscard]] auto base_ptrs(int slot_id) const
      -> const std::vector<std::byte*>&;
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

  std::vector<SlotInfo> slots_;
  int bmax_ = 1;

  std::vector<std::vector<HostBufferInfo>> host_buffer_infos_;

  std::vector<int> free_ids_;
  std::mutex mtx_;
  std::condition_variable cv_;
};

namespace detail {

using StarpuVectorRegisterFn = decltype(&starpu_vector_data_register);
using RegisterFailureObserverFn =
    std::function<void(const InputSlotPool::SlotInfo& slot)>;

auto starpu_vector_register_hook() -> StarpuVectorRegisterFn&;
auto starpu_register_failure_observer() -> RegisterFailureObserverFn&;

}  // namespace detail

}  // namespace starpu_server
