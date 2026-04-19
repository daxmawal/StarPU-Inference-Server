#include "batch_composition_policy.hpp"

#include <cstddef>

#include "exceptions.hpp"
#include "inference_task.hpp"

namespace starpu_server {
namespace {

void
validate_prototype_tensor_impl(const torch::Tensor& tensor)
{
  if (!tensor.defined()) {
    throw InvalidInputTensorException(
        "Input tensor must be defined before batching");
  }
  if (tensor.dim() <= 0) {
    throw InvalidInputTensorException(
        "Input tensor must have at least one dimension");
  }
}

void
validate_tensor_against_prototype_impl(
    const torch::Tensor& tensor, const torch::Tensor& prototype)
{
  if (!tensor.defined()) {
    throw InvalidInputTensorException(
        "Input tensor must be defined before batching");
  }
  if (tensor.dim() != prototype.dim()) {
    throw InvalidInputTensorException(
        "Input tensor rank mismatch during batching");
  }
  if (tensor.dim() <= 0) {
    throw InvalidInputTensorException(
        "Input tensor must have at least one dimension");
  }
  for (int64_t dim = 1; dim < tensor.dim(); ++dim) {
    if (tensor.size(dim) != prototype.size(dim)) {
      throw InvalidInputTensorException(
          "Input tensor shape mismatch during batching");
    }
  }
}

auto
accumulate_samples_for_tensor(
    const std::vector<std::shared_ptr<InferenceJob>>& jobs, size_t tensor_idx,
    const torch::Tensor& prototype) -> int64_t
{
  int64_t accumulated_samples = 0;
  for (const auto& job : jobs) {
    const auto& tensors = job->get_input_tensors();
    if (tensor_idx >= tensors.size()) {
      throw InconsistentInputTensorCountException(
          "Inconsistent input tensor count");
    }
    const auto& tensor = tensors[tensor_idx];
    validate_tensor_against_prototype_impl(tensor, prototype);
    accumulated_samples += tensor.size(0);
  }
  return accumulated_samples;
}

void
copy_tensor_slices_to_merged(
    const std::vector<std::shared_ptr<InferenceJob>>& jobs, size_t tensor_idx,
    const torch::Tensor& merged_tensor)
{
  int64_t offset = 0;
  for (const auto& job : jobs) {
    const auto& tensor = job->get_input_tensors()[tensor_idx];
    const int64_t slice = tensor.size(0);
    merged_tensor.narrow(0, offset, slice).copy_(tensor);
    offset += slice;
  }
}

}  // namespace

auto
TensorBatchCompositionPolicy::should_hold_job(
    const std::shared_ptr<InferenceJob>& candidate,
    const std::shared_ptr<InferenceJob>& reference,
    const std::optional<int>& target_worker) const -> bool
{
  if (!candidate) {
    return false;
  }
  if (candidate->batch().has_aggregated_sub_jobs() ||
      candidate->batch().logical_job_count() > 1) {
    return true;
  }
  if (target_worker != candidate->get_fixed_worker_id()) {
    return true;
  }
  return !can_merge_jobs(reference, candidate);
}

auto
TensorBatchCompositionPolicy::can_merge_jobs(
    const std::shared_ptr<InferenceJob>& lhs,
    const std::shared_ptr<InferenceJob>& rhs) const -> bool
{
  if (!lhs || !rhs) {
    return false;
  }

  const auto& lhs_inputs = lhs->get_input_tensors();
  const auto& rhs_inputs = rhs->get_input_tensors();
  if (lhs_inputs.size() != rhs_inputs.size()) {
    return false;
  }

  const auto& lhs_types = lhs->get_input_types();
  const auto& rhs_types = rhs->get_input_types();
  if (lhs_types.size() != rhs_types.size()) {
    return false;
  }
  for (size_t idx = 0; idx < lhs_types.size(); ++idx) {
    if (lhs_types[idx] != rhs_types[idx]) {
      return false;
    }
  }

  for (size_t idx = 0; idx < lhs_inputs.size(); ++idx) {
    const auto& lhs_tensor = lhs_inputs[idx];
    const auto& rhs_tensor = rhs_inputs[idx];
    if (!lhs_tensor.defined() || !rhs_tensor.defined()) {
      return false;
    }
    if (lhs_tensor.dim() != rhs_tensor.dim()) {
      return false;
    }
    if (lhs_tensor.dim() <= 0) {
      return false;
    }
    if (lhs_tensor.dim() <= 1) {
      continue;
    }
    for (int64_t dim = 1; dim < lhs_tensor.dim(); ++dim) {
      if (lhs_tensor.size(dim) != rhs_tensor.size(dim)) {
        return false;
      }
    }
  }

  return true;
}

auto
TensorBatchCompositionPolicy::merge_input_tensors(
    const std::vector<std::shared_ptr<InferenceJob>>& jobs,
    int64_t total_samples) const -> std::vector<torch::Tensor>
{
  std::vector<torch::Tensor> merged;
  if (jobs.empty()) {
    return merged;
  }
  const auto& first_inputs = jobs.front()->get_input_tensors();
  merged.reserve(first_inputs.size());
  const bool single_job_batch = jobs.size() == 1;

  for (size_t tensor_idx = 0; tensor_idx < first_inputs.size(); ++tensor_idx) {
    if (single_job_batch) {
      merged.push_back(first_inputs[tensor_idx]);
      continue;
    }

    const auto& prototype = first_inputs[tensor_idx];
    validate_prototype_tensor_impl(prototype);

    const int64_t accumulated_samples =
        accumulate_samples_for_tensor(jobs, tensor_idx, prototype);
    const int64_t target_samples =
        total_samples > 0 ? total_samples : accumulated_samples;
    if (accumulated_samples != target_samples) {
      throw InvalidInputTensorException(
          "Total samples mismatch while batching inputs");
    }

    auto shape = prototype.sizes().vec();
    shape.front() = target_samples;
    auto merged_tensor = torch::empty(shape, prototype.options());
    copy_tensor_slices_to_merged(jobs, tensor_idx, merged_tensor);

    merged.emplace_back(std::move(merged_tensor));
  }

  return merged;
}

auto
TensorBatchCompositionPolicy::merge_input_memory_holders(
    const std::vector<std::shared_ptr<InferenceJob>>& jobs) const
    -> std::vector<std::shared_ptr<const void>>
{
  std::vector<std::shared_ptr<const void>> holders;
  std::size_t total_holders = 0;
  for (const auto& job : jobs) {
    total_holders += job->get_input_memory_holders().size();
  }
  holders.reserve(total_holders);
  for (const auto& job : jobs) {
    const auto& job_holders = job->get_input_memory_holders();
    holders.insert(holders.end(), job_holders.begin(), job_holders.end());
  }
  return holders;
}

void
validate_batch_prototype_tensor(const torch::Tensor& tensor)
{
  validate_prototype_tensor_impl(tensor);
}

void
validate_tensor_against_batch_prototype(
    const torch::Tensor& tensor, const torch::Tensor& prototype)
{
  validate_tensor_against_prototype_impl(tensor, prototype);
}

}  // namespace starpu_server
