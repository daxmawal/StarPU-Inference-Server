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

class OutputSlotPool {
 public:
  struct SlotInfo {
    int id = -1;
    std::vector<void*> base_ptrs;
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
  [[nodiscard]] auto base_ptrs(int slot_id) const -> const std::vector<void*>&;
  [[nodiscard]] int max_batch_size() const { return bmax_; }
  [[nodiscard]] size_t num_outputs() const
  {
    return per_output_numel_single_.size();
  }

 private:
  void allocate_pool(const RuntimeConfig& opts, int slots);
  void allocate_slot_buffers_and_register(
      int slot_id, const RuntimeConfig& opts);
  static size_t product_dims(const std::vector<int64_t>& dims);

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

struct OutputSlotPoolTestHook {
  static void cleanup_slot_buffers(
      OutputSlotPool::SlotInfo& slot,
      std::vector<OutputSlotPool::HostBufferInfo>& buffer_infos, size_t count);
};

}  // namespace starpu_server
