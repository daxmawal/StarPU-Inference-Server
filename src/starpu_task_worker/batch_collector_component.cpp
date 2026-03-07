#include "batch_collector_component.hpp"

#include <algorithm>
#include <chrono>
#include <format>
#include <optional>
#include <utility>

#include "exceptions.hpp"
#include "logger.hpp"
#include "monitoring/metrics.hpp"
#include "result_dispatcher_component.hpp"
#include "task_runner_internal.hpp"
#include "utils/monotonic_clock.hpp"

namespace starpu_server {

using clock = task_runner_internal::Clock;

BatchCollector::BatchCollector(
    InferenceQueue* queue, const RuntimeConfig* opts, StarPUSetup* starpu,
    std::shared_ptr<InferenceJob>* pending_job,
    const PreparedBatchingContext& prepared, const InflightContext& inflight)
    : queue_(queue), opts_(opts), starpu_(starpu), pending_job_(pending_job),
      inflight_tasks_(inflight.inflight_tasks),
      inflight_cv_(inflight.inflight_cv),
      inflight_mutex_(inflight.inflight_mutex),
      max_inflight_tasks_(inflight.max_inflight_tasks),
      prepared_mutex_(prepared.prepared_mutex),
      prepared_cv_(prepared.prepared_cv),
      prepared_jobs_(prepared.prepared_jobs),
      batching_done_(prepared.batching_done)
{
}

auto
BatchCollector::wait_for_next_job() -> std::shared_ptr<InferenceJob>
{
  if (max_inflight_tasks_ > 0 && inflight_tasks_ != nullptr &&
      inflight_mutex_ != nullptr && inflight_cv_ != nullptr) {
    std::unique_lock lock(*inflight_mutex_);
    while (inflight_tasks_->load(std::memory_order_acquire) >=
           max_inflight_tasks_) {
      if (should_abort_inflight_wait()) {
        return nullptr;
      }
      // Bounded waiting avoids indefinite blocking when shutdown happens on a
      // different condition variable.
      static_cast<void>(
          inflight_cv_->wait_for(lock, std::chrono::milliseconds(10)));
    }
  }

  if (pending_job_ != nullptr && *pending_job_) {
    return std::exchange(*pending_job_, nullptr);
  }
  std::shared_ptr<InferenceJob> job;
  if (queue_ == nullptr || !queue_->wait_and_pop(job)) {
    return nullptr;
  }
  return job;
}

auto
BatchCollector::collect_batch(const std::shared_ptr<InferenceJob>& first_job)
    -> std::vector<std::shared_ptr<InferenceJob>>
{
  std::vector<std::shared_ptr<InferenceJob>> jobs;
  if (first_job == nullptr) {
    set_batch_pending_jobs(0);
    return jobs;
  }

  jobs.push_back(first_job);
  if (!opts_->batching.dynamic_batching) {
    set_batch_pending_jobs(jobs.size());
    return jobs;
  }
  if (first_job->has_aggregated_sub_jobs() ||
      first_job->logical_job_count() > 1) {
    set_batch_pending_jobs(jobs.size());
    return jobs;
  }

  const int max_job_count = std::max(1, opts_->batching.max_batch_size);
  if (max_job_count <= 1) {
    set_batch_pending_jobs(jobs.size());
    return jobs;
  }

  const int64_t max_samples_cap = sample_limit_per_batch();
  int64_t accumulated_samples = job_sample_size(first_job);

  const bool enable_wait = opts_->batching.batch_coalesce_timeout_ms > 0;
  const auto batch_coalesce_timeout =
      std::chrono::milliseconds(opts_->batching.batch_coalesce_timeout_ms);
  const auto coalesce_deadline =
      enable_wait ? clock::now() + batch_coalesce_timeout : clock::time_point{};
  const auto& target_worker = first_job->get_fixed_worker_id();

  while (jobs.size() < static_cast<size_t>(max_job_count)) {
    auto next = try_acquire_next_job(enable_wait, coalesce_deadline);
    const bool should_break =
        next == nullptr || should_hold_job(next, jobs.front(), target_worker) ||
        exceeds_sample_limit(accumulated_samples, next, max_samples_cap);
    if (next && should_break) {
      store_pending_job(next);
    }
    if (should_break) {
      break;
    }
    accumulated_samples += job_sample_size(next);
    jobs.push_back(std::move(next));
  }

  set_batch_pending_jobs(jobs.size());
  return jobs;
}

auto
BatchCollector::try_acquire_next_job(
    bool enable_wait,
    clock::time_point coalesce_deadline) -> std::shared_ptr<InferenceJob>
{
  if (queue_ == nullptr) {
    return nullptr;
  }

  while (true) {
    std::shared_ptr<InferenceJob> next;
    bool got_job = queue_->try_pop(next);
    if (!got_job && enable_wait) {
      const auto now = clock::now();
      if (now >= coalesce_deadline) {
        return nullptr;
      }
      got_job = queue_->wait_for_and_pop(next, coalesce_deadline - now);
    }
    if (!got_job) {
      return nullptr;
    }
    if (next) {
      return next;
    }
  }
}

void
BatchCollector::store_pending_job(const std::shared_ptr<InferenceJob>& job)
{
  if (pending_job_ != nullptr) {
    *pending_job_ = job;
  }
}

[[nodiscard]] auto
BatchCollector::is_batching_done() const -> bool
{
  if (batching_done_ == nullptr) {
    return false;
  }
  if (prepared_mutex_ == nullptr) {
    return *batching_done_;
  }
  const std::scoped_lock lock(*prepared_mutex_);
  return *batching_done_;
}

[[nodiscard]] auto
BatchCollector::should_abort_inflight_wait() const -> bool
{
  if (is_batching_done()) {
    return true;
  }
  if (queue_ == nullptr) {
    return true;
  }
  if (pending_job_ != nullptr && *pending_job_ != nullptr) {
    return false;
  }
  return queue_->is_shutdown() && queue_->size() == 0;
}

[[nodiscard]] auto
BatchCollector::should_hold_job(
    const std::shared_ptr<InferenceJob>& candidate,
    const std::shared_ptr<InferenceJob>& reference,
    const std::optional<int>& target_worker) -> bool
{
  if (!candidate) {
    return false;
  }
  if (candidate->has_aggregated_sub_jobs() ||
      candidate->logical_job_count() > 1) {
    return true;
  }
  if (target_worker != candidate->get_fixed_worker_id()) {
    return true;
  }
  return !can_merge_jobs(reference, candidate);
}

[[nodiscard]] auto
BatchCollector::exceeds_sample_limit(
    int64_t accumulated_samples, const std::shared_ptr<InferenceJob>& job,
    int64_t max_samples_cap) const -> bool
{
  if (max_samples_cap <= 0) {
    return false;
  }
  const int64_t next_samples = job_sample_size(job);
  return accumulated_samples + next_samples > max_samples_cap;
}

auto
BatchCollector::can_merge_jobs(
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
BatchCollector::merge_input_tensors(
    const std::vector<std::shared_ptr<InferenceJob>>& jobs,
    int64_t total_samples) -> std::vector<torch::Tensor>
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
BatchCollector::merge_input_memory_holders(
    const std::vector<std::shared_ptr<InferenceJob>>& jobs)
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

auto
BatchCollector::job_sample_size(const std::shared_ptr<InferenceJob>& job) const
    -> int64_t
{
  if (!job) {
    return 0;
  }
  return task_runner_internal::resolve_batch_size_for_job(opts_, job);
}

auto
BatchCollector::sample_limit_per_batch() const -> int
{
  const int configured_limit =
      opts_ != nullptr ? std::max(1, opts_->batching.max_batch_size) : 1;

  if (starpu_ != nullptr && starpu_->has_input_pool()) {
    const int pool_limit = std::max(1, starpu_->input_pool().max_batch_size());
    return std::min(configured_limit, pool_limit);
  }

  return configured_limit;
}

auto
BatchCollector::maybe_build_batched_job(
    std::vector<std::shared_ptr<InferenceJob>>& jobs)
    -> std::shared_ptr<InferenceJob>
{
  if (jobs.empty()) {
    return nullptr;
  }

  auto master = jobs.front();
  if (jobs.size() == 1) {
    const auto samples = std::max<int64_t>(1, job_sample_size(master));
    observe_batch_efficiency(static_cast<double>(samples));
    master->set_logical_job_count(1);
    master->set_aggregated_sub_jobs({});
    return master;
  }

  auto batch_info = task_runner_internal::aggregate_batch_metadata(jobs, opts_);
  auto earliest_start = batch_info.earliest_start;
  auto earliest_enqueued = batch_info.earliest_enqueued;
  auto earliest_batch_collect_start = batch_info.earliest_batch_collect_start;

  if (earliest_start == clock::time_point{}) {
    earliest_start = earliest_enqueued != clock::time_point{}
                         ? earliest_enqueued
                         : clock::now();
  }
  if (earliest_batch_collect_start == clock::time_point{}) {
    earliest_batch_collect_start = master->timing_info_snapshot().dequeued_time;
  }

  master->set_logical_job_count(batch_info.logical_jobs);
  master->set_aggregated_sub_jobs(std::move(batch_info.sub_jobs));

  if (auto lifetimes = merge_input_memory_holders(jobs); !lifetimes.empty()) {
    master->set_input_memory_holders(std::move(lifetimes));
  }

  const auto prototype_outputs = master->get_output_tensors();
  const int64_t effective_batch = batch_info.total_samples > 0
                                      ? batch_info.total_samples
                                      : static_cast<int64_t>(jobs.size());
  master->set_output_tensors(task_runner_internal::resize_outputs_for_batch(
      prototype_outputs, effective_batch));

  const auto logical_jobs =
      static_cast<std::size_t>(std::max(1, batch_info.logical_jobs));
  const double efficiency = logical_jobs > 0
                                ? static_cast<double>(effective_batch) /
                                      static_cast<double>(logical_jobs)
                                : 0.0;
  observe_batch_efficiency(efficiency);

  master->set_start_time(earliest_start);
  master->update_timing_info(
      [earliest_enqueued, latest_enqueued = batch_info.latest_enqueued,
       earliest_batch_collect_start](detail::TimingInfo& timing) {
        timing.enqueued_time = earliest_enqueued;
        timing.last_enqueued_time = latest_enqueued == clock::time_point{}
                                        ? earliest_enqueued
                                        : latest_enqueued;
        timing.batch_collect_start_time = earliest_batch_collect_start;
      });

  if (const bool need_materialized_inputs =
          (starpu_ == nullptr || !starpu_->has_input_pool())) {
    if (auto merged_inputs = merge_input_tensors(jobs, effective_batch);
        !merged_inputs.empty()) {
      master->set_input_tensors(merged_inputs);
    }
    task_runner_internal::release_inputs_from_additional_jobs(jobs);
    master->clear_pending_sub_jobs();
  } else {
    std::vector<std::shared_ptr<InferenceJob>> pending_jobs;
    pending_jobs.reserve(jobs.empty() ? 0 : jobs.size() - 1);
    for (size_t idx = 1; idx < jobs.size(); ++idx) {
      pending_jobs.push_back(jobs[idx]);
      jobs[idx]->set_output_tensors({});
    }
    master->set_pending_sub_jobs(std::move(pending_jobs));
  }

  master->set_effective_batch_size(effective_batch);

  auto master_wp = std::weak_ptr<InferenceJob>(master);
  master->set_on_complete(
      [master_wp](
          const std::vector<torch::Tensor>& aggregated_outputs,
          double latency_ms) {
        if (auto master_sp = master_wp.lock()) {
          ResultDispatcher::propagate_completion_to_sub_jobs(
              master_sp, aggregated_outputs, latency_ms);
        }
      });

  if (opts_ != nullptr && should_log(VerbosityLevel::Trace, opts_->verbosity)) {
    const auto queue_size =
        queue_ != nullptr ? static_cast<unsigned long long>(queue_->size()) : 0;
    log_trace(
        opts_->verbosity,
        std::format(
            "Formed batch for job ID {} with {} requests ({} samples); "
            "queue size after dequeue: {}",
            master->get_request_id(), jobs.size(), effective_batch,
            queue_size));
  }

  return master;
}

void
BatchCollector::enqueue_prepared_job(const std::shared_ptr<InferenceJob>& job)
{
  if (prepared_mutex_ == nullptr || prepared_jobs_ == nullptr ||
      prepared_cv_ == nullptr) {
    return;
  }
  if (max_inflight_tasks_ > 0 && inflight_tasks_ != nullptr && job != nullptr) {
    const auto current =
        inflight_tasks_->fetch_add(1, std::memory_order_release) + 1;
    set_inflight_tasks(current);
    if (max_inflight_tasks_ > 0) {
      const double ratio = static_cast<double>(current) /
                           static_cast<double>(max_inflight_tasks_);
      set_starpu_worker_busy_ratio(ratio);
    }
  }
  {
    const std::scoped_lock lock(*prepared_mutex_);
    prepared_jobs_->push_back(job);
    set_starpu_prepared_queue_depth(prepared_jobs_->size());
  }
  prepared_cv_->notify_one();
}

auto
BatchCollector::wait_for_prepared_job() -> std::shared_ptr<InferenceJob>
{
  if (prepared_mutex_ == nullptr || prepared_jobs_ == nullptr ||
      prepared_cv_ == nullptr) {
    return nullptr;
  }
  std::unique_lock lock(*prepared_mutex_);
  prepared_cv_->wait(lock, [this] {
    return !prepared_jobs_->empty() ||
           (batching_done_ != nullptr && *batching_done_);
  });
  if (prepared_jobs_->empty()) {
    return nullptr;
  }
  auto job = prepared_jobs_->front();
  prepared_jobs_->pop_front();
  set_starpu_prepared_queue_depth(prepared_jobs_->size());
  return job;
}

void
BatchCollector::batching_loop()
{
  bool should_stop = false;
  while (!should_stop) {
    auto job = wait_for_next_job();
    if (!job) {
      should_stop = true;
      continue;
    }

    const auto dequeue_time = MonotonicClock::now();
    job->update_timing_info([dequeue_time](detail::TimingInfo& timing) {
      timing.dequeued_time = dequeue_time;
      timing.batch_collect_start_time = dequeue_time;
      timing.batch_collect_end_time = dequeue_time;
    });

    auto jobs = collect_batch(job);
    job = maybe_build_batched_job(jobs);
    if (!job) {
      continue;
    }

    const auto batch_collect_end = MonotonicClock::now();
    job->update_timing_info([batch_collect_end](detail::TimingInfo& timing) {
      timing.batch_collect_end_time = batch_collect_end;
    });

    enqueue_prepared_job(job);
  }

  if (prepared_mutex_ != nullptr && prepared_cv_ != nullptr &&
      batching_done_ != nullptr) {
    {
      const std::scoped_lock lock(*prepared_mutex_);
      *batching_done_ = true;
    }
    prepared_cv_->notify_all();
  }
}

// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
namespace task_runner_internal::testing {

void
validate_tensor_against_prototype(
    const torch::Tensor& tensor, const torch::Tensor& prototype)
{
  validate_tensor_against_prototype_impl(tensor, prototype);
}

void
validate_prototype_tensor(const torch::Tensor& tensor)
{
  validate_prototype_tensor_impl(tensor);
}

auto
batch_collector_job_sample_size(
    const BatchCollector* collector,
    const std::shared_ptr<InferenceJob>& job) -> int64_t
{
  return collector != nullptr ? collector->job_sample_size(job) : -1;
}

auto
batch_collector_exceeds_sample_limit(
    const BatchCollector* collector, int64_t accumulated_samples,
    const std::shared_ptr<InferenceJob>& job, int64_t max_samples_cap) -> bool
{
  return collector != nullptr ? collector->exceeds_sample_limit(
                                    accumulated_samples, job, max_samples_cap)
                              : false;
}

auto
batch_collector_try_acquire_next_job(
    BatchCollector* collector, bool enable_wait,
    Clock::time_point coalesce_deadline) -> std::shared_ptr<InferenceJob>
{
  if (collector == nullptr) {
    return nullptr;
  }
  return collector->try_acquire_next_job(enable_wait, coalesce_deadline);
}

auto
batch_collector_should_hold_job(
    const std::shared_ptr<InferenceJob>& candidate,
    const std::shared_ptr<InferenceJob>& reference,
    const std::optional<int>& target_worker) -> bool
{
  return BatchCollector::should_hold_job(candidate, reference, target_worker);
}

void
batch_collector_disable_prepared_job_sync(BatchCollector* collector)
{
  if (collector == nullptr) {
    return;
  }
  collector->prepared_mutex_ = nullptr;
  collector->prepared_cv_ = nullptr;
  collector->prepared_jobs_ = nullptr;
}

void
batch_collector_set_queue(BatchCollector* collector, InferenceQueue* queue)
{
  if (collector == nullptr) {
    return;
  }
  collector->queue_ = queue;
}

auto
batch_collector_get_queue(const BatchCollector* collector) -> InferenceQueue*
{
  return collector != nullptr ? collector->queue_ : nullptr;
}

void
batch_collector_set_pending_job(
    BatchCollector* collector, const std::shared_ptr<InferenceJob>& job)
{
  if (collector == nullptr || collector->pending_job_ == nullptr) {
    return;
  }
  *collector->pending_job_ = job;
}

}  // namespace task_runner_internal::testing
#endif  // SONAR_IGNORE_END
// GCOVR_EXCL_STOP

}  // namespace starpu_server
