#include "batch_capacity_policy.hpp"

#include <algorithm>

#include "starpu_setup.hpp"
#include "task_runner_internal.hpp"

namespace starpu_server {

auto
RuntimeBatchCapacityPolicy::job_sample_size(
    const RuntimeConfig* opts, const std::shared_ptr<InferenceJob>& job) const
    -> int64_t
{
  if (!job) {
    return 0;
  }
  return task_runner_internal::resolve_batch_size_for_job(opts, job);
}

auto
RuntimeBatchCapacityPolicy::sample_limit_per_batch(
    const RuntimeConfig* opts, StarPUSetup* starpu) const -> int64_t
{
  const int configured_limit =
      opts != nullptr ? resolved_batch_capacity(opts->batching) : 1;

  if (starpu != nullptr && starpu->has_input_pool()) {
    const int pool_limit = std::max(1, starpu->input_pool().max_batch_size());
    return std::min(configured_limit, pool_limit);
  }

  return configured_limit;
}

auto
RuntimeBatchCapacityPolicy::exceeds_sample_limit(
    const RuntimeConfig* opts, int64_t accumulated_samples,
    const std::shared_ptr<InferenceJob>& job, int64_t max_samples_cap) const
    -> bool
{
  if (max_samples_cap <= 0) {
    return false;
  }
  const int64_t next_samples = job_sample_size(opts, job);
  return accumulated_samples + next_samples > max_samples_cap;
}

}  // namespace starpu_server
