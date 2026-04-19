#pragma once

#include <torch/torch.h>

#include <memory>
#include <optional>
#include <vector>

namespace starpu_server {

class InferenceJob;

class BatchCompositionPolicy {
 public:
  BatchCompositionPolicy() = default;
  virtual ~BatchCompositionPolicy() = default;
  BatchCompositionPolicy(const BatchCompositionPolicy&) = default;
  auto operator=(const BatchCompositionPolicy&) -> BatchCompositionPolicy& =
                                                       default;
  BatchCompositionPolicy(BatchCompositionPolicy&&) = default;
  auto operator=(BatchCompositionPolicy&&) -> BatchCompositionPolicy& = default;

  [[nodiscard]] virtual auto should_hold_job(
      const std::shared_ptr<InferenceJob>& candidate,
      const std::shared_ptr<InferenceJob>& reference,
      const std::optional<int>& target_worker) const -> bool = 0;

  [[nodiscard]] virtual auto can_merge_jobs(
      const std::shared_ptr<InferenceJob>& lhs,
      const std::shared_ptr<InferenceJob>& rhs) const -> bool = 0;

  [[nodiscard]] virtual auto merge_input_tensors(
      const std::vector<std::shared_ptr<InferenceJob>>& jobs,
      int64_t total_samples) const -> std::vector<torch::Tensor> = 0;

  [[nodiscard]] virtual auto merge_input_memory_holders(
      const std::vector<std::shared_ptr<InferenceJob>>& jobs) const
      -> std::vector<std::shared_ptr<const void>> = 0;
};

class TensorBatchCompositionPolicy final : public BatchCompositionPolicy {
 public:
  [[nodiscard]] auto should_hold_job(
      const std::shared_ptr<InferenceJob>& candidate,
      const std::shared_ptr<InferenceJob>& reference,
      const std::optional<int>& target_worker) const -> bool override;

  [[nodiscard]] auto can_merge_jobs(
      const std::shared_ptr<InferenceJob>& lhs,
      const std::shared_ptr<InferenceJob>& rhs) const -> bool override;

  [[nodiscard]] auto merge_input_tensors(
      const std::vector<std::shared_ptr<InferenceJob>>& jobs,
      int64_t total_samples) const -> std::vector<torch::Tensor> override;

  [[nodiscard]] auto merge_input_memory_holders(
      const std::vector<std::shared_ptr<InferenceJob>>& jobs) const
      -> std::vector<std::shared_ptr<const void>> override;
};

void validate_batch_prototype_tensor(const torch::Tensor& tensor);
void validate_tensor_against_batch_prototype(
    const torch::Tensor& tensor, const torch::Tensor& prototype);

}  // namespace starpu_server
