// Common helper for StarPU slot pools
#pragma once

#include <starpu.h>

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <optional>
#include <vector>

namespace starpu_server {

struct SlotPoolSlot {
  int id = -1;
  std::vector<std::byte*> base_ptrs;
  std::vector<starpu_data_handle_t> handles;
};

template <typename SlotInfoType = SlotPoolSlot>
class SlotPoolBase {
 public:
  using SlotInfo = SlotInfoType;

  auto acquire() -> int
  {
    std::unique_lock lock(mtx_);
    cv_.wait(lock, [this] { return !free_ids_.empty(); });
    const int slot_id = free_ids_.back();
    free_ids_.pop_back();
    return slot_id;
  }

  [[nodiscard]] auto try_acquire() -> std::optional<int>
  {
    std::scoped_lock lock(mtx_);
    if (free_ids_.empty()) {
      return std::nullopt;
    }
    const int slot_id = free_ids_.back();
    free_ids_.pop_back();
    return slot_id;
  }

  void release(int slot_id)
  {
    {
      const std::scoped_lock lock(mtx_);
      free_ids_.push_back(slot_id);
    }
    cv_.notify_one();
  }

  [[nodiscard]] auto slot_info(int slot_id) const -> const SlotInfo&
  {
    return slots_.at(static_cast<size_t>(slot_id));
  }

  [[nodiscard]] auto handles(int slot_id) const
      -> const std::vector<starpu_data_handle_t>&
  {
    return slots_.at(static_cast<size_t>(slot_id)).handles;
  }

  [[nodiscard]] auto base_ptrs(int slot_id) const
      -> const std::vector<std::byte*>&
  {
    return slots_.at(static_cast<size_t>(slot_id)).base_ptrs;
  }

 protected:
  SlotPoolBase() = default;
  ~SlotPoolBase() = default;

  SlotPoolBase(const SlotPoolBase&) = delete;
  auto operator=(const SlotPoolBase&) -> SlotPoolBase& = delete;
  SlotPoolBase(SlotPoolBase&&) = delete;
  auto operator=(SlotPoolBase&&) -> SlotPoolBase& = delete;

  std::vector<SlotInfo> slots_;
  std::vector<int> free_ids_;
  mutable std::mutex mtx_;
  std::condition_variable cv_;
};

}  // namespace starpu_server
