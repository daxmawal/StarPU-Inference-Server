#include "starpu_task_worker.hpp"

#include <torch/torch.h>

#include <algorithm>
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
  if (pending_job_) {
    return std::exchange(pending_job_, nullptr);
  }
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
    int request_id, double latency_ms,
    const detail::TimingInfo& timing_info) const
{
  if (!should_log(VerbosityLevel::Stats, opts_->verbosity)) {
    return;
  }

  const auto submission_id = timing_info.submission_id;
  using duration_f = std::chrono::duration<double, std::milli>;
  const auto queue_ms =
      duration_f(timing_info.dequeued_time - timing_info.enqueued_time).count();
  const auto batch_ms = std::max(
      0.0, duration_f(
               timing_info.batch_collect_end_time -
               timing_info.batch_collect_start_time)
               .count());
  const auto submit_ms = std::max(
      0.0, duration_f(
               timing_info.before_starpu_submitted_time -
               timing_info.batch_collect_end_time)
               .count());
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

  const auto header =
      submission_id >= 0
          ? std::format(
                "Job {} done. Latency = {:.3f} ms | Queue = ", submission_id,
                latency_ms)
          : std::format(
                "Job {} done. Latency = {:.3f} ms | Queue = ", request_id,
                latency_ms);

  log_stats(
      opts_->verbosity,
      std::format(
          "{}{:.3f} ms, Batching = {:.3f} ms, Submit = {:.3f} ms, Scheduling = "
          "{:.3f} ms, Codelet = {:.3f} ms, Inference = {:.3f} ms, Callback = "
          "{:.3f} ms",
          header, queue_ms, batch_ms, submit_ms, scheduling_ms, codelet_ms,
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
          stored_result.request_id = job_sptr->get_request_id();
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

        job_sptr->timing_info().submission_id = job_sptr->submission_id();
        log_job_timings(
            job_sptr->get_request_id(), latency_ms, job_sptr->timing_info());

        if (prev_callback) {
          prev_callback(results, latency_ms);
        }

        const int logical_jobs = std::max(1, job_sptr->logical_job_count());
        completed_jobs_->fetch_add(logical_jobs, std::memory_order_release);
        all_done_cv_->notify_all();
      });
}

// =============================================================================
// Error Handling for Failed Jobs
// =============================================================================

void
StarPUTaskRunner::handle_job_exception(
    const std::shared_ptr<InferenceJob>& job, const std::exception& exception)
{
  const auto request_id = job->get_request_id();
  log_error(
      std::format("[Exception] Job {}: {}", request_id, exception.what()));

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
StarPUTaskRunner::collect_batch(const std::shared_ptr<InferenceJob>& first_job)
    -> std::vector<std::shared_ptr<InferenceJob>>
{
  std::vector<std::shared_ptr<InferenceJob>> jobs;
  if (first_job == nullptr) {
    return jobs;
  }

  jobs.push_back(first_job);
  if (first_job->has_aggregated_sub_jobs() ||
      first_job->logical_job_count() > 1) {
    return jobs;
  }

  const int max_batch_size = std::max(1, opts_->max_batch_size);
  if (max_batch_size <= 1) {
    return jobs;
  }

  const bool enable_wait = opts_->delay_ms > 0;
  auto wait_duration = std::chrono::milliseconds(1);
  if (enable_wait) {
    wait_duration = std::chrono::milliseconds(std::min(opts_->delay_ms, 1000));
  }

  const auto& target_worker = first_job->get_fixed_worker_id();
  while (jobs.size() < static_cast<size_t>(max_batch_size)) {
    std::shared_ptr<InferenceJob> next;
    bool got_job = queue_->try_pop(next);
    if (!got_job) {
      if (!enable_wait) {
        break;
      }
      got_job = queue_->wait_for_and_pop(next, wait_duration);
      if (!got_job) {
        break;
      }
    }
    if (!next) {
      continue;
    }
    if (next->is_shutdown() || next->has_aggregated_sub_jobs() ||
        next->logical_job_count() > 1) {
      pending_job_ = next;
      break;
    }
    if (target_worker != next->get_fixed_worker_id()) {
      pending_job_ = next;
      break;
    }
    if (!can_merge_jobs(jobs.front(), next)) {
      pending_job_ = next;
      break;
    }
    jobs.push_back(next);
  }

  return jobs;
}

auto
StarPUTaskRunner::can_merge_jobs(
    const std::shared_ptr<InferenceJob>& lhs,
    const std::shared_ptr<InferenceJob>& rhs) -> bool
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
    const auto& a = lhs_inputs[idx];
    const auto& b = rhs_inputs[idx];
    if (!a.defined() || !b.defined()) {
      return false;
    }
    if (a.dim() != b.dim()) {
      return false;
    }
    if (a.dim() <= 0) {
      return false;
    }
    if (a.dim() <= 1) {
      continue;
    }
    for (int64_t dim = 1; dim < a.dim(); ++dim) {
      if (a.size(dim) != b.size(dim)) {
        return false;
      }
    }
  }

  return true;
}

auto
StarPUTaskRunner::merge_input_tensors(
    const std::vector<std::shared_ptr<InferenceJob>>& jobs)
    -> std::vector<torch::Tensor>
{
  std::vector<torch::Tensor> merged;
  if (jobs.empty()) {
    return merged;
  }
  const auto& first_inputs = jobs.front()->get_input_tensors();
  merged.reserve(first_inputs.size());

  for (size_t tensor_idx = 0; tensor_idx < first_inputs.size(); ++tensor_idx) {
    std::vector<torch::Tensor> to_concat;
    to_concat.reserve(jobs.size());
    for (const auto& job : jobs) {
      const auto& tensors = job->get_input_tensors();
      if (tensor_idx >= tensors.size()) {
        throw std::runtime_error("Inconsistent input tensor count");
      }
      to_concat.push_back(tensors[tensor_idx]);
    }
    if (to_concat.size() == 1) {
      merged.push_back(to_concat.front());
    } else {
      merged.push_back(torch::cat(to_concat, 0));
    }
  }

  return merged;
}

auto
StarPUTaskRunner::merge_input_memory_holders(
    const std::vector<std::shared_ptr<InferenceJob>>& jobs)
    -> std::vector<std::shared_ptr<const void>>
{
  std::vector<std::shared_ptr<const void>> holders;
  for (const auto& job : jobs) {
    const auto& job_holders = job->get_input_memory_holders();
    holders.insert(holders.end(), job_holders.begin(), job_holders.end());
  }
  return holders;
}

auto
StarPUTaskRunner::maybe_build_batched_job(
    std::vector<std::shared_ptr<InferenceJob>>& jobs)
    -> std::shared_ptr<InferenceJob>
{
  if (jobs.empty()) {
    return nullptr;
  }

  auto master = jobs.front();
  if (jobs.size() == 1) {
    master->set_logical_job_count(1);
    master->set_aggregated_sub_jobs({});
    return master;
  }

  std::vector<InferenceJob::AggregatedSubJob> sub_jobs;
  sub_jobs.reserve(jobs.size());
  int logical_jobs = 0;
  int64_t total_samples = 0;

  using clock = std::chrono::high_resolution_clock;
  auto earliest_start = master->get_start_time();
  auto earliest_enqueued = master->timing_info().enqueued_time;
  auto earliest_batch_collect_start =
      master->timing_info().batch_collect_start_time;

  for (const auto& job : jobs) {
    const auto job_batch =
        static_cast<int64_t>(batch_size_from_inputs(job->get_input_tensors()));
    total_samples += job_batch > 0 ? job_batch : 1;
    logical_jobs += std::max(1, job->logical_job_count());

    if (job->get_start_time() < earliest_start) {
      earliest_start = job->get_start_time();
    }

    const auto job_enqueued = job->timing_info().enqueued_time;
    if (job_enqueued != clock::time_point{} &&
        (earliest_enqueued == clock::time_point{} ||
         job_enqueued < earliest_enqueued)) {
      earliest_enqueued = job_enqueued;
    }

    sub_jobs.push_back(
        {std::weak_ptr<InferenceJob>(job), job->get_on_complete(), job_batch});

    const auto job_batch_start = job->timing_info().batch_collect_start_time;
    if (job_batch_start != clock::time_point{} &&
        (earliest_batch_collect_start == clock::time_point{} ||
         job_batch_start < earliest_batch_collect_start)) {
      earliest_batch_collect_start = job_batch_start;
    }
  }

  if (earliest_start == clock::time_point{}) {
    earliest_start = earliest_enqueued != clock::time_point{}
                         ? earliest_enqueued
                         : clock::now();
  }
  if (earliest_batch_collect_start == clock::time_point{}) {
    earliest_batch_collect_start = master->timing_info().dequeued_time;
  }

  master->set_logical_job_count(logical_jobs);
  master->set_aggregated_sub_jobs(std::move(sub_jobs));

  auto lifetimes = merge_input_memory_holders(jobs);
  if (!lifetimes.empty()) {
    master->set_input_memory_holders(std::move(lifetimes));
  }

  auto merged_inputs = merge_input_tensors(jobs);
  if (!merged_inputs.empty()) {
    master->set_input_tensors(merged_inputs);
  }

  auto prototype_outputs = master->get_output_tensors();
  std::vector<torch::Tensor> resized_outputs;
  resized_outputs.reserve(prototype_outputs.size());
  const int64_t effective_batch =
      total_samples > 0 ? total_samples : static_cast<int64_t>(jobs.size());
  for (const auto& out : prototype_outputs) {
    if (!out.defined()) {
      resized_outputs.emplace_back(out);
      continue;
    }
    std::vector<int64_t> shape(out.sizes().begin(), out.sizes().end());
    if (!shape.empty()) {
      shape[0] = effective_batch;
    }
    resized_outputs.emplace_back(torch::empty(shape, out.options()));
  }
  master->set_output_tensors(resized_outputs);

  master->set_start_time(earliest_start);
  master->timing_info().enqueued_time = earliest_enqueued;
  master->timing_info().batch_collect_start_time = earliest_batch_collect_start;

  auto master_wp = std::weak_ptr<InferenceJob>(master);
  master->set_on_complete(
      [master_wp](
          const std::vector<torch::Tensor>& aggregated_outputs,
          double latency_ms) {
        if (auto master_sp = master_wp.lock()) {
          propagate_completion_to_sub_jobs(
              master_sp, aggregated_outputs, latency_ms);
        }
      });

  for (size_t idx = 1; idx < jobs.size(); ++idx) {
    if (jobs[idx]) {
      static_cast<void>(jobs[idx]->release_input_tensors());
      jobs[idx]->set_input_memory_holders(
          std::vector<std::shared_ptr<const void>>{});
    }
  }

  if (should_log(VerbosityLevel::Trace, opts_->verbosity)) {
    log_trace(
        opts_->verbosity,
        std::format(
            "Formed batch for job ID {} with {} requests ({} samples)",
            master->get_request_id(), jobs.size(), effective_batch));
  }

  return master;
}

void
StarPUTaskRunner::enqueue_prepared_job(const std::shared_ptr<InferenceJob>& job)
{
  {
    const std::scoped_lock lock(prepared_mutex_);
    prepared_jobs_.push_back(job);
  }
  prepared_cv_.notify_one();
}

auto
StarPUTaskRunner::wait_for_prepared_job() -> std::shared_ptr<InferenceJob>
{
  std::unique_lock lock(prepared_mutex_);
  prepared_cv_.wait(
      lock, [this] { return !prepared_jobs_.empty() || batching_done_; });
  if (prepared_jobs_.empty()) {
    return nullptr;
  }
  auto job = prepared_jobs_.front();
  prepared_jobs_.pop_front();
  return job;
}

void
StarPUTaskRunner::batching_loop()
{
  while (true) {
    auto job = wait_for_next_job();
    if (!job) {
      break;
    }

    if (job->is_shutdown()) {
      enqueue_prepared_job(job);
      break;
    }

    const auto dequeue_time = std::chrono::high_resolution_clock::now();
    job->timing_info().dequeued_time = dequeue_time;
    job->timing_info().batch_collect_start_time = dequeue_time;
    job->timing_info().batch_collect_end_time = dequeue_time;

    auto jobs = collect_batch(job);
    job = maybe_build_batched_job(jobs);
    if (!job) {
      continue;
    }

    job->timing_info().batch_collect_end_time =
        std::chrono::high_resolution_clock::now();

    enqueue_prepared_job(job);
  }

  {
    const std::scoped_lock lock(prepared_mutex_);
    batching_done_ = true;
  }
  prepared_cv_.notify_all();
}

void
StarPUTaskRunner::propagate_completion_to_sub_jobs(
    const std::shared_ptr<InferenceJob>& aggregated_job,
    const std::vector<torch::Tensor>& aggregated_outputs, double latency_ms)
{
  if (!aggregated_job) {
    return;
  }

  const auto& sub_jobs = aggregated_job->aggregated_sub_jobs();
  if (sub_jobs.empty()) {
    return;
  }

  size_t offset = 0;
  for (const auto& entry : sub_jobs) {
    auto job_sp = entry.job.lock();
    if (!job_sp) {
      offset += static_cast<size_t>(std::max<int64_t>(1, entry.batch_size));
      continue;
    }

    std::vector<torch::Tensor> outputs;
    outputs.reserve(aggregated_outputs.size());
    const auto slice_size = std::max<int64_t>(1, entry.batch_size);
    int64_t processed = slice_size;
    if (aggregated_outputs.empty()) {
      outputs = {};
    } else {
      bool determined_length = false;
      for (const auto& tensor : aggregated_outputs) {
        if (!tensor.defined() || tensor.dim() == 0) {
          outputs.push_back(tensor);
          continue;
        }
        const int64_t available = tensor.size(0);
        const int64_t slice_start = static_cast<int64_t>(offset);
        const int64_t slice_end = std::min<int64_t>(
            available, slice_start + std::max<int64_t>(1, slice_size));
        const int64_t length = std::max<int64_t>(0, slice_end - slice_start);
        if (length <= 0) {
          outputs.emplace_back();
          continue;
        }
        if (!determined_length) {
          processed = length;
          determined_length = true;
        }
        outputs.push_back(tensor.narrow(0, slice_start, length).contiguous());
      }
    }

    job_sp->timing_info() = aggregated_job->timing_info();
    job_sp->get_device_id() = aggregated_job->get_device_id();
    job_sp->get_worker_id() = aggregated_job->get_worker_id();
    job_sp->get_executed_on() = aggregated_job->get_executed_on();
    job_sp->set_submission_id(aggregated_job->submission_id());
    job_sp->timing_info().submission_id = aggregated_job->submission_id();

    if (entry.callback) {
      entry.callback(outputs, latency_ms);
    }

    offset += static_cast<size_t>(std::max<int64_t>(1, processed));
  }
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
      std::string("submit job ") + std::to_string(job->get_request_id()));
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

  {
    const std::scoped_lock lock(prepared_mutex_);
    prepared_jobs_.clear();
    batching_done_ = false;
  }

  batching_thread_ = std::thread(&StarPUTaskRunner::batching_loop, this);

  while (true) {
    auto job = wait_for_prepared_job();
    if (!job || should_shutdown(job)) {
      break;
    }

    const auto submission_id = next_submission_id_.fetch_add(1);
    job->set_submission_id(submission_id);
    job->timing_info().submission_id = submission_id;

    const auto logical_jobs = job ? job->logical_job_count() : 0;
    const auto request_id = job->get_request_id();
    if (should_log(VerbosityLevel::Trace, opts_->verbosity)) {
      log_trace(
          opts_->verbosity,
          std::format(
              "Dequeued job ID: {}, queue size : {}, aggregated requests: {}",
              request_id, queue_->size(), logical_jobs));
    }

    prepare_job_completion_callback(job);

    try {
      if (should_log(VerbosityLevel::Debug, opts_->verbosity)) {
        log_debug(
            opts_->verbosity,
            std::format("Submitting job ID: {}", submission_id));
      }

      submit_inference_task(job);
    }
    catch (const InferenceEngineException& exception) {
      StarPUTaskRunner::handle_job_exception(job, exception);
    }
    catch (const std::exception& e) {
      log_error(std::format(
          "Unexpected exception for job {}: {}", request_id, e.what()));
      StarPUTaskRunner::handle_job_exception(job, e);
    }
  }

  if (batching_thread_.joinable()) {
    batching_thread_.join();
  }

  log_info(opts_->verbosity, "StarPUTaskRunner stopped.");
}
}  // namespace starpu_server
