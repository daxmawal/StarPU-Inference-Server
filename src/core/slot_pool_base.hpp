// Common helper for StarPU slot pools
#pragma once

#include <starpu.h>

#include <cassert>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
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

  SlotPoolBase(const SlotPoolBase&) = delete;
  auto operator=(const SlotPoolBase&) -> SlotPoolBase& = delete;
  SlotPoolBase(SlotPoolBase&&) = delete;
  auto operator=(SlotPoolBase&&) -> SlotPoolBase& = delete;

  auto acquire() -> int
  {
    std::unique_lock lock(mtx_);
    cv_.wait(lock, [this] { return !free_ids_.empty(); });
#if !defined(NDEBUG)
    initialize_debug_slot_tracking_locked();
#endif
    const int slot_id = free_ids_.back();
    free_ids_.pop_back();
#if !defined(NDEBUG)
    mark_slot_acquired_locked(slot_id);
#endif
    return slot_id;
  }

  [[nodiscard]] auto try_acquire() -> std::optional<int>
  {
    std::scoped_lock lock(mtx_);
    if (free_ids_.empty()) {
      return std::nullopt;
    }
#if !defined(NDEBUG)
    initialize_debug_slot_tracking_locked();
#endif
    const int slot_id = free_ids_.back();
    free_ids_.pop_back();
#if !defined(NDEBUG)
    mark_slot_acquired_locked(slot_id);
#endif
    return slot_id;
  }

  void release(int slot_id)
  {
    {
      const std::scoped_lock lock(mtx_);
#if !defined(NDEBUG)
      initialize_debug_slot_tracking_locked();
      mark_slot_released_locked(slot_id);
#endif
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

  auto slots() -> std::vector<SlotInfo>& { return slots_; }
  auto slots() const -> const std::vector<SlotInfo>& { return slots_; }
  auto free_ids() -> std::vector<int>& { return free_ids_; }
  auto free_ids() const -> const std::vector<int>& { return free_ids_; }

 private:
#if !defined(NDEBUG)
  void initialize_debug_slot_tracking_locked()
  {
    if (debug_slot_tracking_initialized_) {
      return;
    }
    debug_slot_in_use_.assign(slots_.size(), std::uint8_t{1});
    for (const int free_id : free_ids_) {
      assert(
          free_id >= 0 &&
          "SlotPoolBase invariant violation: free slot id must be "
          "non-negative");
      const auto index = static_cast<std::size_t>(free_id);
      assert(
          index < slots_.size() &&
          "SlotPoolBase invariant violation: free slot id is out of bounds");
      assert(
          debug_slot_in_use_[index] != 0 &&
          "SlotPoolBase invariant violation: duplicate free slot id");
      debug_slot_in_use_[index] = std::uint8_t{0};
    }
    debug_slot_tracking_initialized_ = true;
  }

  auto debug_slot_index_or_abort_locked(int slot_id) const -> std::size_t
  {
    assert(slot_id >= 0 && "SlotPoolBase detected invalid slot id (negative)");
    const auto index = static_cast<std::size_t>(slot_id);
    assert(
        index < slots_.size() &&
        "SlotPoolBase detected invalid slot id (out of bounds)");
    return index;
  }

  void mark_slot_acquired_locked(int slot_id)
  {
    const auto index = debug_slot_index_or_abort_locked(slot_id);
    assert(
        debug_slot_in_use_[index] == 0 &&
        "SlotPoolBase invariant violation: acquired slot already marked "
        "in-use");
    debug_slot_in_use_[index] = std::uint8_t{1};
  }

  void mark_slot_released_locked(int slot_id)
  {
    const auto index = debug_slot_index_or_abort_locked(slot_id);
    assert(
        debug_slot_in_use_[index] != 0 &&
        "SlotPoolBase::release detected double release or never-acquired slot");
    debug_slot_in_use_[index] = std::uint8_t{0};
  }
#endif

  std::vector<SlotInfo> slots_;
  std::vector<int> free_ids_;
  mutable std::mutex mtx_;
  std::condition_variable cv_;
#if !defined(NDEBUG)
  std::vector<std::uint8_t> debug_slot_in_use_;
  bool debug_slot_tracking_initialized_ = false;
#endif
};

}  // namespace starpu_server
