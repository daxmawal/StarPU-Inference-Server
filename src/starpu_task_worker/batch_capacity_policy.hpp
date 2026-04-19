#pragma once

#include <cstdint>
#include <memory>

#include "utils/runtime_config.hpp"

namespace starpu_server {

class InferenceJob;
class StarPUSetup;

class BatchCapacityPolicy {
 public:
  BatchCapacityPolicy() = default;
  virtual ~BatchCapacityPolicy() = default;
  BatchCapacityPolicy(const BatchCapacityPolicy&) = default;
  auto operator=(const BatchCapacityPolicy&) -> BatchCapacityPolicy& = default;
  BatchCapacityPolicy(BatchCapacityPolicy&&) = default;
  auto operator=(BatchCapacityPolicy&&) -> BatchCapacityPolicy& = default;

  [[nodiscard]] virtual auto job_sample_size(
      const RuntimeConfig* opts,
      const std::shared_ptr<InferenceJob>& job) const -> int64_t = 0;

  [[nodiscard]] virtual auto sample_limit_per_batch(
      const RuntimeConfig* opts, StarPUSetup* starpu) const -> int64_t = 0;

  [[nodiscard]] virtual auto exceeds_sample_limit(
      const RuntimeConfig* opts, int64_t accumulated_samples,
      const std::shared_ptr<InferenceJob>& job,
      int64_t max_samples_cap) const -> bool = 0;
};

class RuntimeBatchCapacityPolicy final : public BatchCapacityPolicy {
 public:
  [[nodiscard]] auto job_sample_size(
      const RuntimeConfig* opts,
      const std::shared_ptr<InferenceJob>& job) const -> int64_t override;

  [[nodiscard]] auto sample_limit_per_batch(
      const RuntimeConfig* opts, StarPUSetup* starpu) const -> int64_t override;

  [[nodiscard]] auto exceeds_sample_limit(
      const RuntimeConfig* opts, int64_t accumulated_samples,
      const std::shared_ptr<InferenceJob>& job,
      int64_t max_samples_cap) const -> bool override;
};

}  // namespace starpu_server
