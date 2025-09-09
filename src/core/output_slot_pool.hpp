// Reusable StarPU output slot pool
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

// OutputSlotPool manages K reusable output slots.
// Each slot holds, for every model output, a contiguous host buffer capable of
// storing up to Bmax samples (max batch size) and a persistent
// starpu_data_handle_t registered once for the lifetime of the server.
class OutputSlotPool {
 public:
  struct SlotInfo {
    int id = -1;
    std::vector<void*> base_ptrs;                 // per-output base pointers
    std::vector<size_t> per_output_numel_single;  // per-sample numel per output
    std::vector<size_t> per_output_bytes_single;  // per-sample bytes per output
    std::vector<starpu_data_handle_t> handles;    // per-output StarPU handles
  };

  // Construct the pool. If slots<=0, an auto default is used.
  OutputSlotPool(const RuntimeConfig& opts, int slots);
  ~OutputSlotPool();

  OutputSlotPool(const OutputSlotPool&) = delete;
  auto operator=(const OutputSlotPool&) -> OutputSlotPool& = delete;
  OutputSlotPool(OutputSlotPool&&) = delete;
  auto operator=(OutputSlotPool&&) -> OutputSlotPool& = delete;

  // Acquire a free slot (blocking). Returns the slot id.
  auto acquire() -> int;
  void release(int slot_id);

  // Accessors
  [[nodiscard]] auto slot_info(int slot_id) const -> const SlotInfo&;
  [[nodiscard]] auto handles(int slot_id) const -> const std::vector<starpu_data_handle_t>&;
  [[nodiscard]] auto base_ptrs(int slot_id) const -> const std::vector<void*>&;
  [[nodiscard]] int max_batch_size() const { return bmax_; }
  [[nodiscard]] size_t num_outputs() const { return per_output_numel_single_.size(); }

 private:
  void allocate_pool(const RuntimeConfig& opts, int slots);
  void allocate_slot_buffers_and_register(int slot_id, const RuntimeConfig& opts);
  static size_t product_dims(const std::vector<int64_t>& dims);

  // Per-output metadata
  std::vector<size_t> per_output_numel_single_;
  std::vector<size_t> per_output_bytes_single_;
  std::vector<at::ScalarType> output_types_;

  // Slots
  std::vector<SlotInfo> slots_;
  int bmax_ = 1;

  // Memory management flags for each slot/output (whether pinned was used)
  std::vector<std::vector<bool>> pinned_flags_;

  // Free-list management
  std::vector<int> free_ids_;
  std::mutex mtx_;
  std::condition_variable cv_;
};

}  // namespace starpu_server

