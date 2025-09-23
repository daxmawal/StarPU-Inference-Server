#include "starpu_task_worker.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <format>
#include <iomanip>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/input_slot_pool.hpp"
#include "exceptions.hpp"
#include "inference_queue.hpp"
#include "inference_runner.hpp"
#include "inference_task.hpp"
#include "logger.hpp"
#include "runtime_config.hpp"
#include "starpu_setup.hpp"
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
      completed_jobs_(config.completed_jobs), all_done_cv_(config.all_done_cv)
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
        {
          const std::scoped_lock lock(*results_mutex_);
          results_->emplace_back(
              job_sptr->get_input_tensors(), results, latency_ms,
              job_sptr->timing_info(), job_sptr->get_job_id(),
              job_sptr->get_device_id(), job_sptr->get_worker_id(),
              job_sptr->get_executed_on());
        }

        perf_observer::record_job(
            job_sptr->timing_info().enqueued_time,
            job_sptr->timing_info().callback_end_time,
            batch_size_from_inputs(job_sptr->get_input_tensors()),
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

void
StarPUTaskRunner::submit_inference_task(
    const std::shared_ptr<InferenceJob>& job)
{
  NvtxRange nvtx_job_scope(
      std::string("submit job ") + std::to_string(job->get_job_id()));
  // If pools are available, wait for a free slot (blocking) instead of
  // opportunistic try-acquire. This ensures we reuse pooled handles and only
  // copy inputs once a slot is actually free.
  if (starpu_->has_input_pool() || starpu_->has_output_pool()) {
    auto* in_pool_ptr =
        starpu_->has_input_pool() ? &starpu_->input_pool() : nullptr;
    auto* out_pool_ptr =
        starpu_->has_output_pool() ? &starpu_->output_pool() : nullptr;

    const bool use_in_pool = (in_pool_ptr != nullptr);
    const bool use_out_pool = (out_pool_ptr != nullptr);

    int in_slot_id = -1;
    int out_slot_id = -1;

    // Block until a slot is available for each enabled pool
    if (use_in_pool) {
      in_slot_id = in_pool_ptr->acquire();
    }
    if (use_out_pool) {
      out_slot_id = out_pool_ptr->acquire();
    }

    bool copied_ok = false;
    try {
      const auto& inputs = job->get_input_tensors();
      if (use_in_pool) {
        const auto& base_ptrs = in_pool_ptr->base_ptrs(in_slot_id);
        if (inputs.size() != base_ptrs.size()) {
          throw std::runtime_error("Input count mismatch between job and slot");
        }
      }

      // Determine batch size b from first input vs configured per-sample rank
      int64_t b = 1;
      if (!opts_->models.empty() && !opts_->models[0].inputs.empty()) {
        const auto per_sample_rank =
            static_cast<int64_t>(opts_->models[0].inputs[0].dims.size());
        const auto rank0 = inputs[0].dim();
        if (rank0 == per_sample_rank + 1) {
          b = inputs[0].size(0);
        } else {
          b = 1;
        }
      }
      if (use_in_pool) {
        if (b < 1 || b > in_pool_ptr->max_batch_size()) {
          throw std::runtime_error("Batch size exceeds input pool capacity");
        }
      }

      // Mark handles as being written on host, copy data, then release to bump
      // coherence/version so StarPU transfers fresh data to device workers.
      if (use_in_pool) {
        NvtxRange nvtx_copy_scope("HtoD-staged host copy (pooled inputs)");
        const auto& base_ptrs = in_pool_ptr->base_ptrs(in_slot_id);
        const auto& h_in = in_pool_ptr->handles(in_slot_id);
        for (size_t i = 0; i < inputs.size(); ++i) {
          const auto& tin = inputs[i];
          if (!tin.defined() || !tin.is_cpu() || !tin.is_contiguous()) {
            throw std::runtime_error(
                "Input tensor must be defined, CPU and contiguous");
          }
          const int rc = starpu_data_acquire(h_in[i], STARPU_W);
          if (rc != 0) {
            throw std::runtime_error("starpu_data_acquire(W) failed");
          }
          const size_t nbytes = static_cast<size_t>(tin.nbytes());
          std::memcpy(base_ptrs[i], tin.data_ptr(), nbytes);
          starpu_data_release(h_in[i]);
        }
      }
      copied_ok = true;

      // Build and submit the task using pooled handles when available
      InferenceTask task(starpu_, job, model_cpu_, models_gpu_, opts_);
      const auto input_handles = use_in_pool ? in_pool_ptr->handles(in_slot_id)
                                             : task.prepare_input_handles();
      const auto output_handles = use_out_pool
                                      ? out_pool_ptr->handles(out_slot_id)
                                      : task.prepare_output_handles();
      auto ctx = task.create_context(input_handles, output_handles);
      ctx->keep_input_handles = use_in_pool;
      ctx->keep_output_handles = use_out_pool;
      if (use_out_pool) {
        ctx->output_pool = out_pool_ptr;
        ctx->output_slot_id = out_slot_id;
      }
      ctx->on_finished = [&, in_pool_ptr, in_slot_id, use_in_pool, out_pool_ptr,
                          out_slot_id, use_out_pool]() {
        if (use_in_pool)
          in_pool_ptr->release(in_slot_id);
        if (use_out_pool)
          out_pool_ptr->release(out_slot_id);
      };
      if (ctx->inference_params) {
        ctx->inference_params->batch_size = b;
      }
      starpu_task* task_ptr =
          task.create_task(input_handles, output_handles, ctx);

      job->timing_info().before_starpu_submitted_time =
          std::chrono::high_resolution_clock::now();

      const int ret = starpu_task_submit(task_ptr);
      if (ret != 0) {
        // on error, cleanup and release slots immediately
        InferenceTask::cleanup(ctx);
        if (use_in_pool)
          in_pool_ptr->release(in_slot_id);
        if (use_out_pool)
          out_pool_ptr->release(out_slot_id);
        throw StarPUTaskSubmissionException(std::format(
            "[ERROR] StarPU task submission failed (code {})", ret));
      }
    }
    catch (...) {
      // If copy failed, release acquired slots before rethrowing
      if (!copied_ok) {
        if (use_in_pool && in_slot_id >= 0)
          in_pool_ptr->release(in_slot_id);
        if (use_out_pool && out_slot_id >= 0)
          out_pool_ptr->release(out_slot_id);
      }
      throw;
    }
  } else {
    // No pools configured; fallback to per-job registration
    InferenceTask task(starpu_, job, model_cpu_, models_gpu_, opts_);
    task.submit();
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
