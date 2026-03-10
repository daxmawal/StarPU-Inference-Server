#include "utils/nvtx.hpp"

namespace starpu_server::inline starpu_task_worker_detail {
void resize_output_handles_for_job(
    const std::vector<torch::Tensor>& outputs,
    const std::vector<starpu_data_handle_t>& handles);
void log_batch_submitted_if_enabled(
    const std::shared_ptr<InferenceJob>& job, bool warmup_job);
}  // namespace starpu_server::inline starpu_task_worker_detail

namespace starpu_server {

struct StarPUTaskRunner::SubmitPipelineContext {
  std::shared_ptr<InferenceJob> job;
  bool warmup_job = false;
  PoolResources pools{};
  int64_t batch_size = 1;
  struct Handles {
    std::vector<starpu_data_handle_t> input_storage;
    std::vector<starpu_data_handle_t> output_storage;
    std::vector<starpu_data_handle_t> input_for_context;
    std::vector<starpu_data_handle_t> output_for_context;
  };
  Handles handles{};
  std::shared_ptr<InferenceCallbackContext> callback_context;
  starpu_task* task_ptr = nullptr;
};

auto
StarPUTaskRunner::acquire_pools() -> PoolResources
{
  return slot_manager_->acquire_pools();
}

auto
StarPUTaskRunner::validate_batch_and_copy_inputs(
    const std::shared_ptr<InferenceJob>& job,
    const PoolResources& pools) -> int64_t
{
  const int64_t batch = resolve_batch_size(job);
  if (!pools.has_input()) {
    return batch;
  }
  return slot_manager_->validate_batch_and_copy_inputs(job, batch, pools);
}

[[nodiscard]] auto
StarPUTaskRunner::resolve_batch_size(
    const std::shared_ptr<InferenceJob>& job) const -> int64_t
{
  if (!job) {
    return 1;
  }
  return task_runner_internal::resolve_batch_size_for_job(opts_, job);
}

auto
StarPUTaskRunner::configure_task_context(
    InferenceTask& task, const PoolResources& pools,
    std::vector<starpu_data_handle_t> input_handles,
    std::vector<starpu_data_handle_t> output_handles,
    int64_t batch_size) -> std::shared_ptr<InferenceCallbackContext>
{
  auto ctx =
      task.create_context(std::move(input_handles), std::move(output_handles));
  ctx->keep_input_handles = pools.has_input();
  ctx->keep_output_handles = pools.has_output();
  if (pools.has_output()) {
    ctx->output_pool = pools.output_pool;
    ctx->output_slot_id = pools.output_slot;
  }
  ctx->on_finished =
      [input_pool = pools.input_pool, input_slot = pools.input_slot,
       output_pool = pools.output_pool, output_slot = pools.output_slot]() {
        if (input_pool != nullptr && input_slot >= 0) {
          input_pool->release(input_slot);
        }
        if (output_pool != nullptr && output_slot >= 0) {
          output_pool->release(output_slot);
        }
      };
  if (ctx->job) {
    resize_output_handles_for_job(
        ctx->job->get_output_tensors(), ctx->outputs_handles);
  }
  if (ctx->inference_params) {
    ctx->inference_params->batch_size = batch_size;
  }
  return ctx;
}

void
StarPUTaskRunner::handle_submission_failure(
    const PoolResources& pools,
    const std::shared_ptr<InferenceCallbackContext>& ctx, int submit_code)
{
  InferenceTask::cleanup(ctx);
  if (pools.has_input() && pools.input_slot >= 0) {
    pools.input_pool->release(pools.input_slot);
  }
  if (pools.has_output() && pools.output_slot >= 0) {
    pools.output_pool->release(pools.output_slot);
  }
  throw StarPUTaskSubmissionException(std::format(
      "[ERROR] StarPU task submission failed (code {})", submit_code));
}

auto
StarPUTaskRunner::make_submit_pipeline_context(
    const std::shared_ptr<InferenceJob>& job) -> SubmitPipelineContext
{
  SubmitPipelineContext context{};
  context.job = job;
  context.warmup_job = is_warmup_job(job);
  return context;
}

void
StarPUTaskRunner::submit_pipeline_acquire_pools(SubmitPipelineContext& context)
{
  context.pools = acquire_pools();
}

void
StarPUTaskRunner::submit_pipeline_prepare_batch(SubmitPipelineContext& context)
{
  context.batch_size =
      validate_batch_and_copy_inputs(context.job, context.pools);
}

void
StarPUTaskRunner::submit_pipeline_prepare_handles(
    SubmitPipelineContext& context, InferenceTask& task)
{
  if (context.pools.has_input()) {
    context.handles.input_for_context =
        context.pools.input_pool->handles(context.pools.input_slot);
  } else {
    context.handles.input_storage = task.prepare_input_handles();
    context.handles.input_for_context = context.handles.input_storage;
  }

  if (context.pools.has_output()) {
    context.handles.output_for_context =
        context.pools.output_pool->handles(context.pools.output_slot);
  } else {
    context.handles.output_storage = task.prepare_output_handles();
    context.handles.output_for_context = context.handles.output_storage;
  }
}

void
StarPUTaskRunner::submit_pipeline_build_task(
    SubmitPipelineContext& context, InferenceTask& task)
{
  context.callback_context = configure_task_context(
      task, context.pools, std::move(context.handles.input_for_context),
      std::move(context.handles.output_for_context), context.batch_size);

  context.task_ptr = task.create_task(
      context.callback_context->inputs_handles,
      context.callback_context->outputs_handles, context.callback_context);
}

auto
StarPUTaskRunner::submit_pipeline_submit(SubmitPipelineContext& context) -> int
{
  const auto submitted_at = MonotonicClock::now();
  context.job->update_timing_info([submitted_at](detail::TimingInfo& timing) {
    timing.before_starpu_submitted_time = submitted_at;
  });
  return starpu_task_submit(context.task_ptr);
}

void
StarPUTaskRunner::submit_pipeline_cleanup_on_failure(
    const SubmitPipelineContext& context, int submit_code)
{
  handle_submission_failure(
      context.pools, context.callback_context, submit_code);
}

void
StarPUTaskRunner::submit_inference_task(
    const std::shared_ptr<InferenceJob>& job)
{
  task_runner_internal::invoke_submit_inference_task_hook();

  auto label =
      std::format("submit job {}", task_runner_internal::job_identifier(*job));
  NvtxRange nvtx_job_scope(label);
  const bool warmup_job = is_warmup_job(job);
  if (!(starpu_->has_input_pool() || starpu_->has_output_pool())) {
    InferenceTask task(
        starpu_, job, model_cpu_, models_gpu_, opts_, dependencies_);
    task.submit();
    log_batch_submitted_if_enabled(job, warmup_job);
    return;
  }

  struct SlotPoolReleaseGuard {
    explicit SlotPoolReleaseGuard(const PoolResources& pool_resources) noexcept
        : pools(pool_resources)
    {
    }
    SlotPoolReleaseGuard(const SlotPoolReleaseGuard&) = delete;
    SlotPoolReleaseGuard(SlotPoolReleaseGuard&&) = delete;
    auto operator=(const SlotPoolReleaseGuard&) -> SlotPoolReleaseGuard& =
                                                       delete;
    auto operator=(SlotPoolReleaseGuard&&) -> SlotPoolReleaseGuard& = delete;
    ~SlotPoolReleaseGuard() noexcept
    {
      if (!active) {
        return;
      }
      release();
    }

    void dismiss() noexcept { active = false; }

   private:
    void release() noexcept
    {
      if (pools.has_input() && pools.input_slot >= 0) {
        pools.input_pool->release(pools.input_slot);
      }
      if (pools.has_output() && pools.output_slot >= 0) {
        pools.output_pool->release(pools.output_slot);
      }
    }

    const PoolResources& pools;
    bool active{true};
  };

  auto context = make_submit_pipeline_context(job);
  submit_pipeline_acquire_pools(context);
  SlotPoolReleaseGuard pool_guard(context.pools);
  submit_pipeline_prepare_batch(context);

  InferenceTask task(
      starpu_, job, model_cpu_, models_gpu_, opts_, dependencies_);
  submit_pipeline_prepare_handles(context, task);
  submit_pipeline_build_task(context, task);

  const int submit_code = submit_pipeline_submit(context);
  if (submit_code != 0) {
    pool_guard.dismiss();
    submit_pipeline_cleanup_on_failure(context, submit_code);
  }

  pool_guard.dismiss();
  log_batch_submitted_if_enabled(context.job, context.warmup_job);
}

}  // namespace starpu_server
