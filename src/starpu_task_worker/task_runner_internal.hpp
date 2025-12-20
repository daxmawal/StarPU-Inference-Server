#pragma once

#include <torch/torch.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#if defined(STARPU_TESTING)
#include <cuda_runtime_api.h>

#include <functional>
#include <span>

#include "core/input_slot_pool.hpp"
#include "core/output_slot_pool.hpp"
#endif
#include <memory>
#include <vector>

#include "core/inference_runner.hpp"
#include "utils/runtime_config.hpp"
#if defined(STARPU_TESTING)
struct starpu_vector_interface;

namespace starpu_server {
class BatchCollector;
class InferenceQueue;
class SlotManager;
}  // namespace starpu_server
#endif

namespace starpu_server::task_runner_internal {

using Clock = std::chrono::high_resolution_clock;

struct BatchAggregationInfo {
  std::vector<InferenceJob::AggregatedSubJob> sub_jobs;
  int logical_jobs{0};
  int64_t total_samples{0};
  Clock::time_point earliest_start;
  Clock::time_point earliest_enqueued;
  Clock::time_point earliest_batch_collect_start;
  Clock::time_point latest_enqueued;
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

auto aggregate_batch_metadata(
    const std::vector<std::shared_ptr<InferenceJob>>& jobs,
    const RuntimeConfig* opts = nullptr) -> BatchAggregationInfo;

auto resize_outputs_for_batch(
    const std::vector<torch::Tensor>& prototype_outputs,
    int64_t batch_size) -> std::vector<torch::Tensor>;

void release_inputs_from_additional_jobs(
    std::vector<std::shared_ptr<InferenceJob>>& jobs);

auto build_request_ids_for_trace(const std::shared_ptr<InferenceJob>& job)
    -> std::vector<int>;

auto build_request_arrival_us_for_trace(
    const std::shared_ptr<InferenceJob>& job) -> std::vector<int64_t>;

#if defined(STARPU_TESTING)
void set_submit_inference_task_hook(std::function<void()> hook);
void reset_submit_inference_task_hook();

namespace testing {
struct VectorResizeSpecShim {
  std::size_t element_count;
  std::size_t byte_count;
};

void validate_tensor_against_prototype(
    const torch::Tensor& tensor, const torch::Tensor& prototype);
void validate_prototype_tensor(const torch::Tensor& tensor);
void resize_starpu_vector_interface(
    starpu_vector_interface* vector_interface, VectorResizeSpecShim spec,
    bool is_input_handle);
auto batch_size_from_inputs(const std::vector<torch::Tensor>& inputs)
    -> std::size_t;

auto cuda_copy_batch_create(bool enable) -> void*;
void cuda_copy_batch_destroy(void* batch);
auto cuda_copy_batch_enqueue(
    void* batch, std::byte* dst, const std::byte* src, std::size_t bytes,
    bool allow_async) -> bool;
void cuda_copy_batch_finalize(void* batch);
auto cuda_copy_batch_enabled(const void* batch) -> bool;
auto cuda_copy_batch_pending(const void* batch) -> bool;
auto cuda_copy_batch_stream(const void* batch) -> cudaStream_t;

void slot_handle_lease_construct(
    void* storage, std::span<const starpu_data_handle_t> handles,
    starpu_data_access_mode mode);
void slot_handle_lease_destroy(void* storage);

auto slot_manager_copy_job_inputs_to_slot(
    const std::shared_ptr<InferenceJob>& job,
    std::span<const std::shared_ptr<InferenceJob>> pending_jobs,
    std::span<const starpu_data_handle_t> handles,
    std::span<std::byte* const> base_ptrs,
    std::span<const InputSlotPool::HostBufferInfo> buffer_infos,
    void* copy_batch) -> std::size_t;
auto slot_manager_validate_batch_and_copy_inputs(
    SlotManager* slot_manager, const std::shared_ptr<InferenceJob>& job,
    int64_t batch, InputSlotPool* input_pool, int input_slot,
    OutputSlotPool* output_pool, int output_slot) -> int64_t;
auto batch_collector_job_sample_size(
    const BatchCollector* collector,
    const std::shared_ptr<InferenceJob>& job) -> int64_t;
auto batch_collector_exceeds_sample_limit(
    const BatchCollector* collector, int64_t accumulated_samples,
    const std::shared_ptr<InferenceJob>& job, int64_t max_samples_cap) -> bool;
auto batch_collector_should_hold_job(
    const std::shared_ptr<InferenceJob>& candidate,
    const std::shared_ptr<InferenceJob>& reference,
    const std::optional<int>& target_worker) -> bool;
void batch_collector_disable_prepared_job_sync(BatchCollector* collector);
void batch_collector_set_queue(
    BatchCollector* collector, InferenceQueue* queue);
auto batch_collector_get_queue(const BatchCollector* collector)
    -> InferenceQueue*;
void batch_collector_set_pending_job(
    BatchCollector* collector, const std::shared_ptr<InferenceJob>& job);
}  // namespace testing
#endif

}  // namespace starpu_server::task_runner_internal
