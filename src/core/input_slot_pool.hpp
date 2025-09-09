// Reusable StarPU input slot pool
#pragma once

#include <starpu.h>
#include <torch/script.h>

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "utils/datatype_utils.hpp"
#include "utils/runtime_config.hpp"

namespace starpu_server {

// InputSlotPool manages K reusable input slots.
// Each slot holds, for every model input, a contiguous host buffer capable of
// storing up to Bmax samples (max batch size) and a persistent
// starpu_data_handle_t registered once for the lifetime of the server.
class InputSlotPool {
 public:
  struct SlotInfo {
    int id = -1;
    std::vector<void*> base_ptrs;                 // per-input base pointers
    std::vector<size_t> per_input_numel_single;   // per-sample numel per input
    std::vector<size_t> per_input_bytes_single;   // per-sample bytes per input
    std::vector<starpu_data_handle_t> handles;    // per-input StarPU handles
  };

  // Construct the pool. If slots<=0, an auto default is used.
  InputSlotPool(const RuntimeConfig& opts, int slots);
  ~InputSlotPool();

  InputSlotPool(const InputSlotPool&) = delete;
  auto operator=(const InputSlotPool&) -> InputSlotPool& = delete;
  InputSlotPool(InputSlotPool&&) = delete;
  auto operator=(InputSlotPool&&) -> InputSlotPool& = delete;

  // Acquire a free slot (blocking). Returns the slot id.
  auto acquire() -> int;
  void release(int slot_id);

  // Accessors
  [[nodiscard]] auto slot_info(int slot_id) const -> const SlotInfo&;
  [[nodiscard]] auto handles(int slot_id) const -> const std::vector<starpu_data_handle_t>&;
  [[nodiscard]] auto base_ptrs(int slot_id) const -> const std::vector<void*>&;
  [[nodiscard]] int max_batch_size() const { return bmax_; }
  [[nodiscard]] size_t num_inputs() const { return per_input_numel_single_.size(); }

 private:
  void allocate_pool(const RuntimeConfig& opts, int slots);
  void allocate_slot_buffers_and_register(int slot_id, const RuntimeConfig& opts);
  static size_t product_dims(const std::vector<int64_t>& dims);

  // Per-input metadata
  std::vector<size_t> per_input_numel_single_;
  std::vector<size_t> per_input_bytes_single_;
  std::vector<at::ScalarType> input_types_;

  // Slots
  std::vector<SlotInfo> slots_;
  int bmax_ = 1;

  // Memory management flags for each slot/input (whether pinned was used)
  std::vector<std::vector<bool>> pinned_flags_;

  // Free-list management
  std::vector<int> free_ids_;
  std::mutex mtx_;
  std::condition_variable cv_;
};

}  // namespace starpu_server

