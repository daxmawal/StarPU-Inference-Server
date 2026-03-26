#pragma once

#include <starpu.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "starpu_task_worker.hpp"

namespace starpu_server {

class CudaCopyBatch;

class SlotManager {
 public:
  SlotManager(
      StarPUSetup* starpu, const RuntimeConfig* opts,
      torch::jit::script::Module* model_cpu,
      std::vector<torch::jit::script::Module>* models_gpu,
      const std::vector<detail::GpuReplicaAssignment>* gpu_replica_assignments,
      const InferenceTaskDependencies& dependencies,
      std::shared_ptr<RuntimeObservability> observability);

  auto acquire_pools() const -> StarPUTaskRunner::PoolResources;

  [[nodiscard]] auto validate_batch_and_copy_inputs(
      const std::shared_ptr<InferenceJob>& job, int64_t batch,
      const StarPUTaskRunner::PoolResources& pools) const -> int64_t;

  auto submit_inference_task(const std::shared_ptr<InferenceJob>& job) const
      -> void;

  static auto configure_task_context(
      InferenceTask& task, const StarPUTaskRunner::PoolResources& pools,
      std::vector<starpu_data_handle_t> input_handles,
      std::vector<starpu_data_handle_t> output_handles,
      int64_t batch_size) -> std::shared_ptr<InferenceCallbackContext>;

  [[noreturn]] static void handle_submission_failure(
      const StarPUTaskRunner::PoolResources& pools,
      const std::shared_ptr<InferenceCallbackContext>& ctx, int submit_code);

  static auto copy_job_inputs_to_slot(
      const std::shared_ptr<InferenceJob>& job,
      std::span<const std::shared_ptr<InferenceJob>> pending_jobs,
      std::span<const starpu_data_handle_t> handles,
      std::span<std::byte* const> base_ptrs,
      std::span<const InputSlotPool::HostBufferInfo> buffer_infos,
      CudaCopyBatch& copy_batch) -> std::size_t;

  static void release_pending_jobs(
      const std::shared_ptr<InferenceJob>& job,
      std::vector<std::shared_ptr<InferenceJob>>& pending_jobs);

// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  static auto validate_batch_and_copy_inputs_for_test(
      SlotManager* manager, const std::shared_ptr<InferenceJob>& job,
      int64_t batch, InputSlotPool* input_pool, int input_slot,
      OutputSlotPool* output_pool, int output_slot) -> int64_t;
#endif  // SONAR_IGNORE_END
        // GCOVR_EXCL_STOP

 private:
  StarPUSetup* starpu_;
  const RuntimeConfig* opts_;
  torch::jit::script::Module* model_cpu_;
  std::vector<torch::jit::script::Module>* models_gpu_;
  const std::vector<detail::GpuReplicaAssignment>* gpu_replica_assignments_;
  InferenceTaskDependencies dependencies_;
  std::shared_ptr<RuntimeObservability> observability_;

  [[nodiscard]] auto resolve_batch_size(
      const std::shared_ptr<InferenceJob>& job) const -> int64_t;
};

}  // namespace starpu_server
