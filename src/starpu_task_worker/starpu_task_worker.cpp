#include "starpu_task_worker.hpp"

#include <chrono>
#include <cstring>
#include <exception>
#include <format>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "exceptions.hpp"
#include "inference_task.hpp"
#include "logger.hpp"
#include "utils/nvtx.hpp"
#include "utils/perf_observer.hpp"

namespace starpu_server {
namespace {

inline void
validate_not_null(const void* ptr, std::string_view field_name)
{
  if (ptr != nullptr) {
    return;
  }
  throw std::invalid_argument(std::format(
      "[ERROR] StarPUTaskRunnerConfig::{} must not be null", field_name));
}

inline auto
batch_size_from_inputs(const std::vector<torch::Tensor>& inputs) -> std::size_t
{
  if (inputs.empty()) {
    return 1;
  }

  const auto& first = inputs.front();
  if (first.dim() <= 0) {
    return 1;
  }

  const auto dim0 = first.size(0);
  return dim0 > 0 ? static_cast<std::size_t>(dim0) : std::size_t{1};
}
}  // namespace
// =============================================================================
// Constructor
// =============================================================================

StarPUTaskRunner::StarPUTaskRunner(const StarPUTaskRunnerConfig& config)
    : queue_(config.queue), model_cpu_(config.model_cpu),
      models_gpu_(config.models_gpu), starpu_(config.starpu),
      opts_(config.opts), results_(config.results),
      results_mutex_(config.results_mutex),
      completed_jobs_(config.completed_jobs), all_done_cv_(config.all_done_cv),
      dependencies_(
          config.dependencies != nullptr ? config.dependencies
                                         : &kDefaultInferenceTaskDependencies)
{
  for (const auto& [ptr, name] :
       std::initializer_list<std::pair<const void*, std::string_view>>{
           {queue_, "queue"},
           {model_cpu_, "model_cpu"},
           {models_gpu_, "models_gpu"},
           {starpu_, "starpu"},
           {opts_, "opts"},
           {results_, "results"},
           {results_mutex_, "results_mutex"},
           {completed_jobs_, "completed_jobs"},
           {all_done_cv_, "all_done_cv"},
       }) {
    validate_not_null(ptr, name);
  }
}


// =============================================================================
// Job Queue Management
// =============================================================================

auto
StarPUTaskRunner::wait_for_next_job() -> std::shared_ptr<InferenceJob>
{
  std::shared_ptr<InferenceJob> job;
  if (!queue_->wait_and_pop(job)) {
    return nullptr;
  }
  return job;
}

auto
StarPUTaskRunner::should_shutdown(
    const std::shared_ptr<InferenceJob>& job) const -> bool
{
  if (job->is_shutdown()) {
    log_info(
        opts_->verbosity,
        "Received shutdown signal. Exiting StarPUTaskRunner loop.");
    return true;
  }
  return false;
}

// =============================================================================
// Completion Callback Handling
// =============================================================================

void
StarPUTaskRunner::log_job_timings(
    int job_id, double latency_ms, const detail::TimingInfo& timing_info) const
{
  using duration_f = std::chrono::duration<double, std::milli>;
  const auto queue_ms =
      duration_f(timing_info.dequeued_time - timing_info.enqueued_time).count();
  const auto submit_ms =
      duration_f(
          timing_info.before_starpu_submitted_time - timing_info.dequeued_time)
          .count();
  const auto scheduling_ms = duration_f(
                                 timing_info.codelet_start_time -
                                 timing_info.before_starpu_submitted_time)
                                 .count();
  const auto codelet_ms =
      duration_f(timing_info.codelet_end_time - timing_info.codelet_start_time)
          .count();
  const auto inference_ms =
      duration_f(
          timing_info.callback_start_time - timing_info.inference_start_time)
          .count();
  const auto callback_ms =
      duration_f(
          timing_info.callback_end_time - timing_info.callback_start_time)
          .count();

  log_stats(
      opts_->verbosity,
      std::format(
          "Job {} done. Latency = {:.3f} ms | Queue = {:.3f} ms, Submit = "
          "{:.3f} ms, Scheduling = {:.3f} ms, Codelet = {:.3f} ms, Inference = "
          "{:.3f} ms, Callback = {:.3f} ms",
          job_id, latency_ms, queue_ms, submit_ms, scheduling_ms, codelet_ms,
          inference_ms, callback_ms));
}

void
StarPUTaskRunner::prepare_job_completion_callback(
    const std::shared_ptr<InferenceJob>& job)
{
  auto prev_callback = job->get_on_complete();
  job->set_on_complete(
      [this, job_sptr = job, prev_callback](
          const std::vector<torch::Tensor>& results, double latency_ms) {
        const auto batch_size =
            batch_size_from_inputs(job_sptr->get_input_tensors());
        {
          auto input_tensors = job_sptr->release_input_tensors();

          const std::scoped_lock lock(*results_mutex_);
          auto& stored_result = results_->emplace_back();
          if (opts_->validate_results) {
            stored_result.inputs = std::move(input_tensors);
            stored_result.results = results;
          }
          stored_result.latency_ms = latency_ms;
          stored_result.timing_info = job_sptr->timing_info();
          stored_result.job_id = job_sptr->get_job_id();
          stored_result.device_id = job_sptr->get_device_id();
          stored_result.worker_id = job_sptr->get_worker_id();
          stored_result.executed_on = job_sptr->get_executed_on();
        }

        auto& timing = job_sptr->timing_info();
        using clock = std::chrono::high_resolution_clock;
        const auto zero_tp = clock::time_point{};
        const auto now = clock::now();

        if (timing.callback_start_time == zero_tp) {
          timing.callback_start_time = now;
        }
        if (timing.callback_end_time == zero_tp) {
          timing.callback_end_time = now;
        }
        if (timing.callback_end_time <= timing.callback_start_time) {
          timing.callback_end_time =
              timing.callback_start_time + clock::duration{1};
        }
        if (timing.enqueued_time == zero_tp ||
            timing.enqueued_time >= timing.callback_end_time) {
          timing.enqueued_time = timing.callback_start_time;
        }

        perf_observer::record_job(
            job_sptr->timing_info().enqueued_time,
            job_sptr->timing_info().callback_end_time, batch_size,
            job_sptr->get_fixed_worker_id().has_value());

        log_job_timings(
            job_sptr->get_job_id(), latency_ms, job_sptr->timing_info());

        if (prev_callback) {
          prev_callback(results, latency_ms);
        }

        completed_jobs_->fetch_add(1, std::memory_order_release);
        all_done_cv_->notify_one();
      });
}

// =============================================================================
// Error Handling for Failed Jobs
// =============================================================================

void
StarPUTaskRunner::handle_job_exception(
    const std::shared_ptr<InferenceJob>& job, const std::exception& exception)
{
  const auto job_id = job->get_job_id();
  log_error(std::format("[Exception] Job {}: {}", job_id, exception.what()));

  if (job->has_on_complete()) {
    try {
      job->get_on_complete()({}, -1);
    }
    catch (const std::exception& e) {
      log_error("Exception in completion callback: " + std::string(e.what()));
    }
    catch (...) {
      log_error("Unknown exception in completion callback");
    }
  }
}

// =============================================================================
// StarPU Task Submission
// =============================================================================

auto
StarPUTaskRunner::acquire_pools() -> PoolResources
{
  PoolResources pools{};
  if (starpu_->has_input_pool()) {
    pools.input_pool = &starpu_->input_pool();
    pools.input_slot = pools.input_pool->acquire();
  }
  if (starpu_->has_output_pool()) {
    pools.output_pool = &starpu_->output_pool();
    pools.output_slot = pools.output_pool->acquire();
  }
  return pools;
}

auto
StarPUTaskRunner::validate_batch_and_copy_inputs(
    const std::shared_ptr<InferenceJob>& job,
    const PoolResources& pools) -> int64_t
{
  int64_t batch = 1;
  const auto& inputs = job->get_input_tensors();

  if (!opts_->models.empty() && !opts_->models[0].inputs.empty() &&
      !inputs.empty()) {
    const auto per_sample_rank =
        static_cast<int64_t>(opts_->models[0].inputs[0].dims.size());
    const auto rank0 = inputs[0].dim();
    batch = (rank0 == per_sample_rank + 1) ? inputs[0].size(0) : 1;
  }

  if (!pools.has_input()) {
    return batch;
  }

  const auto& base_ptrs = pools.input_pool->base_ptrs(pools.input_slot);
  if (inputs.size() != base_ptrs.size()) {
    throw std::runtime_error("Input count mismatch between job and slot");
  }

  if (batch < 1 || batch > pools.input_pool->max_batch_size()) {
    throw std::runtime_error("Batch size exceeds input pool capacity");
  }

  NvtxRange nvtx_copy_scope("HtoD-staged host copy (pooled inputs)");
  const auto& handles = pools.input_pool->handles(pools.input_slot);
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& tin = inputs[i];
    if (!tin.defined() || !tin.is_cpu() || !tin.is_contiguous()) {
      throw std::runtime_error(
          "Input tensor must be defined, CPU and contiguous");
    }
    const int status = starpu_data_acquire(handles[i], STARPU_W);
    if (status != 0) {
      throw std::runtime_error("starpu_data_acquire(W) failed");
    }
    const auto nbytes = static_cast<size_t>(tin.nbytes());
    std::memcpy(base_ptrs[i], tin.data_ptr(), nbytes);
    starpu_data_release(handles[i]);
  }

  return batch;
}

auto
StarPUTaskRunner::configure_task_context(
    InferenceTask& task, const PoolResources& pools,
    const std::vector<starpu_data_handle_t>& input_handles,
    const std::vector<starpu_data_handle_t>& output_handles,
    int64_t batch_size) -> std::shared_ptr<InferenceCallbackContext>
{
  auto ctx = task.create_context(input_handles, output_handles);
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

void
StarPUTaskRunner::submit_inference_task(
    const std::shared_ptr<InferenceJob>& job)
{
  NvtxRange nvtx_job_scope(
      std::string("submit job ") + std::to_string(job->get_job_id()));
  if (!(starpu_->has_input_pool() || starpu_->has_output_pool())) {
    InferenceTask task(
        starpu_, job, model_cpu_, models_gpu_, opts_, *dependencies_);
    task.submit();
    return;
  }

  auto pools = acquire_pools();
  bool copied_ok = !pools.has_input();
  const bool should_release_output_slot =
      pools.has_output() && pools.output_slot >= 0;
  bool release_output_slot_on_exception = false;

  try {
    const auto batch = validate_batch_and_copy_inputs(job, pools);
    copied_ok = true;
    release_output_slot_on_exception = should_release_output_slot;

    InferenceTask task(
        starpu_, job, model_cpu_, models_gpu_, opts_, *dependencies_);

    std::vector<starpu_data_handle_t> input_handles_storage;
    const std::vector<starpu_data_handle_t>* input_handles = nullptr;
    if (pools.has_input()) {
      input_handles = &pools.input_pool->handles(pools.input_slot);
    } else {
      input_handles_storage = task.prepare_input_handles();
      input_handles = &input_handles_storage;
    }

    std::vector<starpu_data_handle_t> output_handles_storage;
    const std::vector<starpu_data_handle_t>* output_handles = nullptr;
    if (pools.has_output()) {
      output_handles = &pools.output_pool->handles(pools.output_slot);
    } else {
      output_handles_storage = task.prepare_output_handles();
      output_handles = &output_handles_storage;
    }

    auto ctx = configure_task_context(
        task, pools, *input_handles, *output_handles, batch);

    starpu_task* task_ptr =
        task.create_task(*input_handles, *output_handles, ctx);

    job->timing_info().before_starpu_submitted_time =
        std::chrono::high_resolution_clock::now();

    const int ret = starpu_task_submit(task_ptr);
    if (ret != 0) {
      release_output_slot_on_exception = false;
      handle_submission_failure(pools, ctx, ret);
    }
    release_output_slot_on_exception = false;
  }
  catch (...) {
    if (!copied_ok) {
      if (pools.has_input() && pools.input_slot >= 0) {
        pools.input_pool->release(pools.input_slot);
      }
      if (pools.has_output() && pools.output_slot >= 0) {
        pools.output_pool->release(pools.output_slot);
      }
    } else if (release_output_slot_on_exception) {
      pools.output_pool->release(pools.output_slot);
    }
    throw;
  }
}

// =============================================================================
// Main run loop: pull jobs, submit them, handle shutdown and errors
// =============================================================================

void
StarPUTaskRunner::run()
{
  log_info(opts_->verbosity, "StarPUTaskRunner started.");

  while (true) {
    auto job = wait_for_next_job();
    if (!job || should_shutdown(job)) {
      break;
    }

    const auto job_id = job->get_job_id();
    log_trace(
        opts_->verbosity,
        std::format(
            "Dequeued job ID: {}, queue size : {}", job_id, queue_->size()));

    prepare_job_completion_callback(job);

    try {
      job->timing_info().dequeued_time =
          std::chrono::high_resolution_clock::now();

      log_debug(opts_->verbosity, std::format("Submitting job ID: {}", job_id));

      submit_inference_task(job);
    }
    catch (const InferenceEngineException& exception) {
      StarPUTaskRunner::handle_job_exception(job, exception);
    }
    catch (const std::exception& e) {
      log_error(
          std::format("Unexpected exception for job {}: {}", job_id, e.what()));
      StarPUTaskRunner::handle_job_exception(job, e);
    }
  }

  log_info(opts_->verbosity, "StarPUTaskRunner stopped.");
}
}  // namespace starpu_server
