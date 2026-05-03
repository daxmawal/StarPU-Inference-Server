#include "batch_collector_component.hpp"

#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <format>
#include <stdexcept>
#include <utility>

#include "logger.hpp"
#include "monitoring/metrics.hpp"
#include "monitoring/runtime_observability.hpp"
#include "result_dispatcher_component.hpp"
#include "task_runner_internal.hpp"

namespace starpu_server {

using clock = task_runner_internal::Clock;

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
inline namespace batch_collector_component_detail {
struct BatchingLoopTestHooks {
  std::function<void(std::shared_ptr<InferenceJob>&)> after_build_job;
};

auto
batching_loop_test_hooks() -> BatchingLoopTestHooks&
{
  static BatchingLoopTestHooks hooks{};
  return hooks;
}
}  // namespace batch_collector_component_detail
#endif  // SONAR_IGNORE_END

namespace {

auto
active_metrics(const std::shared_ptr<RuntimeObservability>& observability)
    -> MetricsRecorder*
{
  return observability != nullptr ? observability->metrics.get() : nullptr;
}

void
set_batch_pending_jobs_metric(
    const std::shared_ptr<RuntimeObservability>& observability,
    std::size_t pending_jobs)
{
  if (auto* metrics = active_metrics(observability); metrics != nullptr) {
    metrics->set_batch_pending_jobs(pending_jobs);
    return;
  }
  set_batch_pending_jobs(pending_jobs);
}

void
observe_batch_efficiency_metric(
    const std::shared_ptr<RuntimeObservability>& observability,
    double efficiency)
{
  if (auto* metrics = active_metrics(observability); metrics != nullptr) {
    metrics->observe_batch_efficiency(efficiency);
    return;
  }
  observe_batch_efficiency(efficiency);
}

struct AggregatedStartTimes {
  clock::time_point earliest_start;
  clock::time_point earliest_enqueued;
};

auto
resolve_aggregated_start_time(const AggregatedStartTimes& times)
    -> clock::time_point
{
  if (times.earliest_start != clock::time_point{}) {
    return times.earliest_start;
  }
  if (times.earliest_enqueued != clock::time_point{}) {
    return times.earliest_enqueued;
  }
  return clock::now();
}

struct BatchCollectStartTimes {
  clock::time_point earliest_batch_collect_start;
  clock::time_point master_dequeued_time;
};

auto
resolve_batch_collect_start_time(const BatchCollectStartTimes& times)
    -> clock::time_point
{
  if (times.earliest_batch_collect_start != clock::time_point{}) {
    return times.earliest_batch_collect_start;
  }
  return times.master_dequeued_time;
}

void
attach_input_memory_holders(
    const BatchCompositionPolicy& composition_policy,
    const std::shared_ptr<InferenceJob>& master,
    const std::vector<std::shared_ptr<InferenceJob>>& jobs)
{
  if (auto lifetimes = composition_policy.merge_input_memory_holders(jobs);
      !lifetimes.empty()) {
    master->set_input_memory_holders(std::move(lifetimes));
  }
}

auto
resolve_effective_batch_size(
    const task_runner_internal::BatchAggregationInfo& batch_info,
    std::size_t job_count) -> int64_t
{
  return batch_info.total_samples > 0 ? batch_info.total_samples
                                      : static_cast<int64_t>(job_count);
}

struct AggregatedBatchTiming {
  clock::time_point earliest_enqueued;
  clock::time_point latest_enqueued;
  clock::time_point earliest_batch_collect_start;
};

void
update_aggregated_batch_timing(
    const std::shared_ptr<InferenceJob>& master,
    const AggregatedBatchTiming& timing_points)
{
  master->update_timing_info([timing_points](detail::TimingInfo& timing) {
    timing.enqueued_time = timing_points.earliest_enqueued;
    timing.last_enqueued_time =
        timing_points.latest_enqueued == clock::time_point{}
            ? timing_points.earliest_enqueued
            : timing_points.latest_enqueued;
    timing.batch_collect_start_time =
        timing_points.earliest_batch_collect_start;
  });
}

void
attach_materialized_inputs_or_pending_jobs(
    const BatchCompositionPolicy& composition_policy, const StarPUSetup* starpu,
    std::vector<std::shared_ptr<InferenceJob>>& jobs,
    const std::shared_ptr<InferenceJob>& master, int64_t effective_batch)
{
  if (starpu == nullptr || !starpu->has_input_pool()) {
    if (auto merged_inputs =
            composition_policy.merge_input_tensors(jobs, effective_batch);
        !merged_inputs.empty()) {
      master->set_input_tensors(merged_inputs);
    }
    task_runner_internal::release_inputs_from_additional_jobs(jobs);
    master->batch().clear_pending_sub_jobs();
    return;
  }

  std::vector<std::shared_ptr<InferenceJob>> pending_jobs;
  pending_jobs.reserve(jobs.size() - 1);
  for (std::size_t idx = 1; idx < jobs.size(); ++idx) {
    pending_jobs.push_back(jobs[idx]);
    jobs[idx]->set_output_tensors({});
  }
  master->batch().set_pending_sub_jobs(std::move(pending_jobs));
}

void
install_sub_job_completion_callback(const std::shared_ptr<InferenceJob>& master)
{
  auto master_wp = std::weak_ptr<InferenceJob>(master);
  master->completion().set_on_complete(
      [master_wp](
          const std::vector<torch::Tensor>& aggregated_outputs,
          double latency_ms) {
        if (auto master_sp = master_wp.lock()) {
          ResultDispatcher::propagate_completion_to_sub_jobs(
              master_sp, aggregated_outputs, latency_ms);
        }
      });
}

void
trace_formed_batch_if_enabled(
    const RuntimeConfig* opts, const InferenceQueue* queue,
    const std::shared_ptr<InferenceJob>& master, std::size_t request_count,
    int64_t effective_batch)
{
  if (opts == nullptr || !should_log(VerbosityLevel::Trace, opts->verbosity)) {
    return;
  }
  const auto queue_size =
      queue != nullptr ? static_cast<unsigned long long>(queue->size()) : 0;
  log_trace(
      opts->verbosity,
      std::format(
          "Formed batch for job ID {} with {} requests ({} samples); "
          "queue size after dequeue: {}",
          master->get_request_id(), request_count, effective_batch,
          queue_size));
}

}  // namespace

BatchCollector::BatchCollector(
    BatchCollectorRuntimeContext runtime,
    const PreparedBatchingContext& prepared, const InflightContext& inflight,
    BatchCollectorBatchingDependencies batching_dependencies)
    : queue_(runtime.queue), opts_(runtime.opts), starpu_(runtime.starpu),
      pending_job_(runtime.pending_job),
      observability_(std::move(runtime.observability)),
      inflight_tasks_(inflight.inflight_tasks),
      inflight_cv_(inflight.inflight_cv),
      inflight_mutex_(inflight.inflight_mutex),
      max_inflight_tasks_(inflight.max_inflight_tasks),
      prepared_mutex_(prepared.prepared_mutex),
      prepared_cv_(prepared.prepared_cv),
      prepared_jobs_(prepared.prepared_jobs),
      batching_done_(prepared.batching_done),
      batch_capacity_policy_(std::move(batching_dependencies.capacity_policy)),
      batch_composition_policy_(
          std::move(batching_dependencies.composition_policy)),
      batching_strategy_input_provider_(
          std::move(batching_dependencies.strategy_input_provider)),
      batching_strategy_(std::move(batching_dependencies.strategy))
{
  if (batch_capacity_policy_ == nullptr) {
    throw std::invalid_argument(
        "BatchCollector requires a batch capacity policy");
  }
  if (batch_composition_policy_ == nullptr) {
    throw std::invalid_argument(
        "BatchCollector requires a batch composition policy");
  }
  if (batching_strategy_input_provider_ == nullptr) {
    throw std::invalid_argument(
        "BatchCollector requires a batching strategy input provider");
  }
  if (batching_strategy_ == nullptr) {
    throw std::invalid_argument("BatchCollector requires a batching strategy");
  }
}

auto
BatchCollector::wait_for_next_job() -> std::shared_ptr<InferenceJob>
{
  if (max_inflight_tasks_ > 0 && inflight_tasks_ != nullptr &&
      inflight_mutex_ != nullptr && inflight_cv_ != nullptr) {
    constexpr auto kInflightWaitFallback = std::chrono::milliseconds(100);
    std::unique_lock lock(*inflight_mutex_);
    while (inflight_tasks_->load(std::memory_order_acquire) >=
           max_inflight_tasks_) {
      if (should_abort_inflight_wait()) {
        return nullptr;
      }
      static_cast<void>(
          inflight_cv_->wait_for(lock, kInflightWaitFallback, [this] {
            return should_abort_inflight_wait() ||
                   inflight_tasks_->load(std::memory_order_acquire) <
                       max_inflight_tasks_;
          }));
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
    set_batch_pending_jobs_metric(observability_, 0);
    return jobs;
  }

  jobs.push_back(first_job);
  if (first_job->batch().has_aggregated_sub_jobs() ||
      first_job->batch().logical_job_count() > 1) {
    set_batch_pending_jobs_metric(observability_, jobs.size());
    return jobs;
  }

  const auto strategy_input = batching_strategy_input_provider_->sample();
  const auto strategy_decision = batching_strategy_->decide(strategy_input);
  const int max_job_count = std::clamp(
      strategy_decision.target_batch_limit, 1,
      std::max(1, strategy_input.config.batch_limit));
  if (max_job_count <= 1) {
    set_batch_pending_jobs_metric(observability_, jobs.size());
    return jobs;
  }

  const int64_t max_samples_cap =
      batch_capacity_policy_->sample_limit_per_batch(opts_, starpu_);
  int64_t accumulated_samples =
      batch_capacity_policy_->job_sample_size(opts_, first_job);

  const int coalesce_timeout_ms =
      std::max(0, strategy_decision.coalesce_timeout_ms);
  const bool enable_wait = coalesce_timeout_ms > 0;
  const auto batch_coalesce_timeout =
      std::chrono::milliseconds(coalesce_timeout_ms);
  const auto coalesce_deadline =
      enable_wait ? clock::now() + batch_coalesce_timeout : clock::time_point{};
  const auto& target_worker = first_job->get_fixed_worker_id();

  while (jobs.size() < static_cast<size_t>(max_job_count)) {
    auto next = try_acquire_next_job(enable_wait, coalesce_deadline);
    const bool should_break =
        next == nullptr ||
        batch_composition_policy_->should_hold_job(
            next, jobs.front(), target_worker) ||
        batch_capacity_policy_->exceeds_sample_limit(
            opts_, accumulated_samples, next, max_samples_cap);
    if (next && should_break) {
      store_pending_job(next);
    }
    if (should_break) {
      break;
    }
    accumulated_samples += batch_capacity_policy_->job_sample_size(opts_, next);
    jobs.push_back(std::move(next));
  }

  set_batch_pending_jobs_metric(observability_, jobs.size());
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
    const auto samples = std::max<int64_t>(
        1, batch_capacity_policy_->job_sample_size(opts_, master));
    observe_batch_efficiency_metric(
        observability_, static_cast<double>(samples));
    master->batch().set_logical_job_count(1);
    master->batch().set_aggregated_sub_jobs({});
    return master;
  }

  auto batch_info = task_runner_internal::aggregate_batch_metadata(jobs, opts_);
  const auto earliest_enqueued = batch_info.earliest_enqueued;
  const auto earliest_start =
      resolve_aggregated_start_time(AggregatedStartTimes{
          .earliest_start = batch_info.earliest_start,
          .earliest_enqueued = earliest_enqueued,
      });
  const auto earliest_batch_collect_start =
      resolve_batch_collect_start_time(BatchCollectStartTimes{
          .earliest_batch_collect_start =
              batch_info.earliest_batch_collect_start,
          .master_dequeued_time = master->timing_info_snapshot().dequeued_time,
      });

  master->batch().set_logical_job_count(batch_info.logical_jobs);
  master->batch().set_aggregated_sub_jobs(std::move(batch_info.sub_jobs));
  attach_input_memory_holders(*batch_composition_policy_, master, jobs);

  const auto prototype_outputs = master->get_output_tensors();
  const int64_t effective_batch =
      resolve_effective_batch_size(batch_info, jobs.size());
  master->set_output_tensors(task_runner_internal::resize_outputs_for_batch(
      prototype_outputs, effective_batch));

  const auto logical_jobs =
      static_cast<std::size_t>(std::max(1, batch_info.logical_jobs));
  const double efficiency = logical_jobs > 0
                                ? static_cast<double>(effective_batch) /
                                      static_cast<double>(logical_jobs)
                                : 0.0;
  observe_batch_efficiency_metric(observability_, efficiency);

  master->set_start_time(earliest_start);
  update_aggregated_batch_timing(
      master, AggregatedBatchTiming{
                  .earliest_enqueued = earliest_enqueued,
                  .latest_enqueued = batch_info.latest_enqueued,
                  .earliest_batch_collect_start = earliest_batch_collect_start,
              });
  attach_materialized_inputs_or_pending_jobs(
      *batch_composition_policy_, starpu_, jobs, master, effective_batch);

  master->batch().set_effective_batch_size(effective_batch);
  install_sub_job_completion_callback(master);
  trace_formed_batch_if_enabled(
      opts_, queue_, master, jobs.size(), effective_batch);

  return master;
}

void
BatchCollector::reset_prepared_queue_state()
{
  if (batching_strategy_ != nullptr) {
    batching_strategy_->reset();
  }
  if (prepared_mutex_ != nullptr && prepared_jobs_ != nullptr) {
    const std::scoped_lock lock(*prepared_mutex_);
    prepared_jobs_->clear();
    if (batching_done_ != nullptr) {
      *batching_done_ = false;
    }
  } else if (batching_done_ != nullptr) {
    *batching_done_ = false;
  }
  if (auto* metrics = active_metrics(observability_); metrics != nullptr) {
    metrics->set_starpu_prepared_queue_depth(0);
    metrics->set_batch_pending_jobs(0);
  } else {
    set_starpu_prepared_queue_depth(0);
    set_batch_pending_jobs(0);
  }
}

void
BatchCollector::abort_prepared_queue()
{
  if (prepared_mutex_ != nullptr && batching_done_ != nullptr) {
    {
      const std::scoped_lock lock(*prepared_mutex_);
      *batching_done_ = true;
    }
  } else if (batching_done_ != nullptr) {
    *batching_done_ = true;
  }

  if (prepared_cv_ != nullptr) {
    prepared_cv_->notify_all();
  }
}

void
BatchCollector::enqueue_prepared_job(const std::shared_ptr<InferenceJob>& job)
{
  if (prepared_mutex_ == nullptr || prepared_jobs_ == nullptr ||
      prepared_cv_ == nullptr) {
    return;
  }
  {
    const std::scoped_lock lock(*prepared_mutex_);
    prepared_jobs_->push_back(job);
    if (auto* metrics = active_metrics(observability_); metrics != nullptr) {
      metrics->set_starpu_prepared_queue_depth(prepared_jobs_->size());
    } else {
      set_starpu_prepared_queue_depth(prepared_jobs_->size());
    }
  }
  if (job != nullptr && max_inflight_tasks_ > 0 && inflight_tasks_ != nullptr) {
    // Inflight accounting is owned by prepared-queue enqueue/dequeue terminal
    // paths: increment here, decrement in ResultDispatcher terminal handling.
    const auto current =
        inflight_tasks_->fetch_add(1, std::memory_order_release) + 1;
    if (auto* metrics = active_metrics(observability_); metrics != nullptr) {
      metrics->set_inflight_tasks(current);
    } else {
      set_inflight_tasks(current);
    }
    const double ratio =
        static_cast<double>(current) / static_cast<double>(max_inflight_tasks_);
    if (auto* metrics = active_metrics(observability_); metrics != nullptr) {
      metrics->set_starpu_worker_busy_ratio(ratio);
    } else {
      set_starpu_worker_busy_ratio(ratio);
    }
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
  if (auto* metrics = active_metrics(observability_); metrics != nullptr) {
    metrics->set_starpu_prepared_queue_depth(prepared_jobs_->size());
  } else {
    set_starpu_prepared_queue_depth(prepared_jobs_->size());
  }
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
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
    auto& test_hooks = batching_loop_test_hooks();
    if (test_hooks.after_build_job) {
      test_hooks.after_build_job(job);
    }
#endif  // SONAR_IGNORE_END
    if (!job) {
      continue;
    }

    const auto batch_collect_end = MonotonicClock::now();
    job->update_timing_info([batch_collect_end](detail::TimingInfo& timing) {
      timing.batch_collect_end_time = batch_collect_end;
    });

    enqueue_prepared_job(job);
  }

  abort_prepared_queue();
}

// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
namespace task_runner_internal::testing {

void
BatchCollectorTestAdapter::set_after_build_job_hook(
    std::function<void(std::shared_ptr<InferenceJob>&)> hook)
{
  batching_loop_test_hooks().after_build_job = std::move(hook);
}

void
BatchCollectorTestAdapter::reset_after_build_job_hook()
{
  batching_loop_test_hooks().after_build_job = {};
}

void
validate_tensor_against_prototype(
    const torch::Tensor& tensor, const torch::Tensor& prototype)
{
  validate_tensor_against_batch_prototype(tensor, prototype);
}

void
validate_prototype_tensor(const torch::Tensor& tensor)
{
  validate_batch_prototype_tensor(tensor);
}

auto
BatchCollectorTestAdapter::try_acquire_next_job(
    BatchCollector* collector, bool enable_wait,
    Clock::time_point coalesce_deadline) -> std::shared_ptr<InferenceJob>
{
  if (collector == nullptr) {
    return nullptr;
  }
  return collector->try_acquire_next_job(enable_wait, coalesce_deadline);
}

auto
BatchCollectorTestAdapter::is_batching_done(const BatchCollector* collector)
    -> bool
{
  return collector != nullptr ? collector->is_batching_done() : false;
}

auto
BatchCollectorTestAdapter::should_abort_inflight_wait(
    const BatchCollector* collector) -> bool
{
  return collector != nullptr ? collector->should_abort_inflight_wait() : false;
}

void
BatchCollectorTestAdapter::disable_prepared_job_sync(BatchCollector* collector)
{
  if (collector == nullptr) {
    return;
  }
  collector->prepared_mutex_ = nullptr;
  collector->prepared_cv_ = nullptr;
  collector->prepared_jobs_ = nullptr;
}

void
BatchCollectorTestAdapter::set_queue(
    BatchCollector* collector, InferenceQueue* queue)
{
  if (collector == nullptr) {
    return;
  }
  collector->queue_ = queue;
}

auto
BatchCollectorTestAdapter::get_queue(const BatchCollector* collector)
    -> InferenceQueue*
{
  return collector != nullptr ? collector->queue_ : nullptr;
}

void
BatchCollectorTestAdapter::set_batching_done_ptr(
    BatchCollector* collector, bool* batching_done)
{
  if (collector == nullptr) {
    return;
  }
  collector->batching_done_ = batching_done;
}

void
BatchCollectorTestAdapter::set_batching_done_value(
    BatchCollector* collector, bool batching_done)
{
  if (collector == nullptr || collector->batching_done_ == nullptr) {
    return;
  }
  *collector->batching_done_ = batching_done;
}

void
BatchCollectorTestAdapter::set_pending_job(
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
