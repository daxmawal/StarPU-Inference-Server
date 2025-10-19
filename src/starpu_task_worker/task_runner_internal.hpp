#pragma once

#include <torch/torch.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "core/inference_runner.hpp"

namespace starpu_server::task_runner_internal {

using Clock = std::chrono::high_resolution_clock;

struct BatchAggregationInfo {
  std::vector<InferenceJob::AggregatedSubJob> sub_jobs;
  int logical_jobs{0};
  int64_t total_samples{0};
  Clock::time_point earliest_start;
  Clock::time_point earliest_enqueued;
  Clock::time_point earliest_batch_collect_start;
};

struct SubJobSliceResult {
  std::vector<torch::Tensor> outputs;
  int64_t processed_length{1};
};

struct SubJobSliceOptions {
  std::size_t offset;
  int64_t batch_size;
};

auto slice_outputs_for_sub_job(
    const std::vector<torch::Tensor>& aggregated_outputs,
    SubJobSliceOptions options) -> SubJobSliceResult;

auto aggregate_batch_metadata(const std::vector<std::shared_ptr<InferenceJob>>&
                                  jobs) -> BatchAggregationInfo;

auto resize_outputs_for_batch(
    const std::vector<torch::Tensor>& prototype_outputs,
    int64_t batch_size) -> std::vector<torch::Tensor>;

void release_inputs_from_additional_jobs(
    std::vector<std::shared_ptr<InferenceJob>>& jobs);

void set_submit_inference_task_hook(std::function<void()> hook);
void reset_submit_inference_task_hook();

}  // namespace starpu_server::task_runner_internal
