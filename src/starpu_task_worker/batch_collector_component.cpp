#include "batch_collector_component.hpp"

#include <algorithm>
#include <chrono>
#include <format>
#include <limits>
#include <optional>
#include <utility>

#include "exceptions.hpp"
#include "logger.hpp"
#include "monitoring/congestion_monitor.hpp"
#include "monitoring/metrics.hpp"
#include "monitoring/runtime_observability.hpp"
#include "result_dispatcher_component.hpp"
#include "task_runner_internal.hpp"
#include "utils/monotonic_clock.hpp"

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

auto
active_congestion_monitor(const std::shared_ptr<RuntimeObservability>&
                              observability) -> congestion::Monitor*
{
  return observability != nullptr ? observability->congestion_monitor.get()
                                  : nullptr;
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

struct BatchPressureThresholds {
  double fill_high = 0.0;
  double fill_low = 0.0;
  double rho_high = 0.0;
  double rho_low = 0.0;
};

struct InternalPressureSnapshot {
  bool high = false;
  bool low = false;
  bool severe = false;
};

struct InternalPressureConfig {
  std::size_t max_inflight_tasks = 0;
  int max_batch_size = 1;
};

struct ResolvedBatchPressure {
  bool congested = false;
  bool high = false;
  bool low = false;
  bool severe = false;
  std::optional<clock::time_point> monitor_tick;
};

constexpr double kInternalHighRatio = 0.75;
constexpr double kInternalLowRatio = 0.25;
constexpr double kInternalSevereRatio = 0.95;
constexpr double kPreparedBacklogHighRatio = 1.0;
constexpr double kPreparedBacklogSevereRatio = 2.0;

auto
make_batch_pressure_thresholds(const RuntimeConfig::CongestionSettings&
                                   congestion) -> BatchPressureThresholds
{
  BatchPressureThresholds thresholds{};
  thresholds.fill_high = std::clamp(congestion.fill_high, 0.0, 1.0);
  thresholds.fill_low = std::clamp(
      std::min(congestion.fill_low, congestion.fill_high), 0.0,
      thresholds.fill_high);
  thresholds.rho_high = std::max(0.0, congestion.rho_high);
  thresholds.rho_low = std::clamp(congestion.rho_low, 0.0, thresholds.rho_high);
  return thresholds;
}

auto
load_prepared_depth(
    const std::deque<std::shared_ptr<InferenceJob>>* prepared_jobs,
    std::mutex* prepared_mutex) -> std::size_t
{
  if (prepared_jobs == nullptr) {
    return 0;
  }
  if (prepared_mutex == nullptr) {
    return prepared_jobs->size();
  }

  const std::scoped_lock lock(*prepared_mutex);
  return prepared_jobs->size();
}

auto
load_inflight_tasks(const std::atomic<std::size_t>* inflight_tasks)
    -> std::size_t
{
  if (inflight_tasks == nullptr) {
    return 0;
  }
  return inflight_tasks->load(std::memory_order_acquire);
}

auto
compute_queue_fill(std::size_t queue_size, std::size_t queue_capacity) -> double
{
  if (queue_capacity == 0) {
    return 0.0;
  }
  return static_cast<double>(queue_size) / static_cast<double>(queue_capacity);
}

auto
sample_internal_pressure(
    std::size_t prepared_depth, std::size_t inflight_tasks,
    const InternalPressureConfig& config) -> InternalPressureSnapshot
{
  InternalPressureSnapshot pressure{};
  const std::size_t total_internal_backlog = prepared_depth + inflight_tasks;

  if (config.max_inflight_tasks > 0) {
    const double inflight_ratio =
        static_cast<double>(inflight_tasks) /
        static_cast<double>(config.max_inflight_tasks);
    const double backlog_ratio = static_cast<double>(total_internal_backlog) /
                                 static_cast<double>(config.max_inflight_tasks);
    pressure.high = inflight_ratio >= kInternalHighRatio ||
                    backlog_ratio >= kInternalHighRatio;
    pressure.low = inflight_ratio <= kInternalLowRatio &&
                   backlog_ratio <= kInternalLowRatio;
    pressure.severe = inflight_ratio >= kInternalSevereRatio ||
                      backlog_ratio >= kInternalSevereRatio;
    return pressure;
  }

  const auto local_backlog_ref =
      static_cast<std::size_t>(std::max(1, config.max_batch_size));
  const double prepared_ratio = static_cast<double>(prepared_depth) /
                                static_cast<double>(local_backlog_ref);
  pressure.high = prepared_ratio >= kPreparedBacklogHighRatio;
  pressure.low = prepared_depth == 0;
  pressure.severe = prepared_ratio >= kPreparedBacklogSevereRatio;
  return pressure;
}

auto
sample_monitor_pressure(
    const std::shared_ptr<RuntimeObservability>& observability,
    const BatchPressureThresholds& thresholds,
    const InternalPressureSnapshot& internal_pressure)
    -> std::optional<ResolvedBatchPressure>
{
  const auto* monitor = active_congestion_monitor(observability);
  const auto snapshot =
      monitor != nullptr
          ? std::optional<congestion::Snapshot>(monitor->snapshot())
          : congestion::snapshot();
  if (!snapshot.has_value()) {
    return std::nullopt;
  }

  const auto& values = *snapshot;
  const double queue_fill =
      compute_queue_fill(values.queue_size, values.queue_capacity);
  const bool queue_high =
      values.queue_capacity > 0 && queue_fill >= thresholds.fill_high;
  const bool queue_low =
      values.queue_capacity == 0 || queue_fill <= thresholds.fill_low;

  ResolvedBatchPressure pressure{};
  pressure.monitor_tick = values.last_tick;
  pressure.congested =
      values.congestion ||
      (monitor != nullptr ? monitor->congested() : congestion::is_congested());
  pressure.severe = pressure.congested || values.rejection_rps > 0.0 ||
                    internal_pressure.severe;
  pressure.high = pressure.severe || values.fill_ewma >= thresholds.fill_high ||
                  values.rho_ewma >= thresholds.rho_high || queue_high ||
                  internal_pressure.high;
  pressure.low = !pressure.congested && values.rejection_rps <= 0.0 &&
                 values.fill_ewma <= thresholds.fill_low &&
                 values.rho_ewma <= thresholds.rho_low && queue_low &&
                 internal_pressure.low;
  return pressure;
}

auto
sample_queue_pressure(
    const std::shared_ptr<RuntimeObservability>& observability,
    const InferenceQueue* queue, const BatchPressureThresholds& thresholds,
    const InternalPressureSnapshot& internal_pressure) -> ResolvedBatchPressure
{
  ResolvedBatchPressure pressure{};
  if (queue == nullptr) {
    pressure.high = internal_pressure.high;
    pressure.low = internal_pressure.low;
    pressure.severe = internal_pressure.severe;
    return pressure;
  }

  const double queue_fill =
      compute_queue_fill(queue->size(), queue->capacity());
  if (const auto* monitor = active_congestion_monitor(observability);
      monitor != nullptr) {
    pressure.congested = monitor->congested();
  } else {
    pressure.congested = congestion::is_congested();
  }
  pressure.high = queue_fill >= thresholds.fill_high || internal_pressure.high;
  pressure.low = !pressure.congested && queue_fill <= thresholds.fill_low &&
                 internal_pressure.low;
  pressure.severe = pressure.congested || internal_pressure.severe;
  return pressure;
}

auto
resolve_coalesce_timeout_ms(
    const RuntimeConfig* opts, bool congested, int max_job_count) -> int
{
  int coalesce_timeout_ms =
      std::max(0, opts->batching.batch_coalesce_timeout_ms);
  if (!congested || !opts->congestion.enabled) {
    return coalesce_timeout_ms;
  }

  // Under congestion, keep a short coalescing window even when the configured
  // timeout is 0 so the collector can fill toward max batch size.
  const int tick_interval_ms = std::max(1, opts->congestion.tick_interval_ms);
  const int per_slot_wait_ms =
      std::max(1, tick_interval_ms / std::max(1, max_job_count));
  return std::max(coalesce_timeout_ms, per_slot_wait_ms);
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
    const std::shared_ptr<InferenceJob>& master,
    const std::vector<std::shared_ptr<InferenceJob>>& jobs)
{
  if (auto lifetimes = BatchCollector::merge_input_memory_holders(jobs);
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
    StarPUSetup* starpu, std::vector<std::shared_ptr<InferenceJob>>& jobs,
    const std::shared_ptr<InferenceJob>& master, int64_t effective_batch)
{
  if (starpu == nullptr || !starpu->has_input_pool()) {
    if (auto merged_inputs =
            BatchCollector::merge_input_tensors(jobs, effective_batch);
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
    InferenceQueue* queue, const RuntimeConfig* opts, StarPUSetup* starpu,
    std::shared_ptr<InferenceJob>* pending_job,
    std::shared_ptr<RuntimeObservability> observability,
    const PreparedBatchingContext& prepared, const InflightContext& inflight)
    : queue_(queue), opts_(opts), starpu_(starpu), pending_job_(pending_job),
      observability_(std::move(observability)),
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
  if (!opts_->batching.dynamic_batching) {
    set_batch_pending_jobs_metric(observability_, jobs.size());
    return jobs;
  }
  if (first_job->batch().has_aggregated_sub_jobs() ||
      first_job->batch().logical_job_count() > 1) {
    set_batch_pending_jobs_metric(observability_, jobs.size());
    return jobs;
  }

  const int max_job_count = effective_batch_limit();
  if (max_job_count <= 1) {
    set_batch_pending_jobs_metric(observability_, jobs.size());
    return jobs;
  }

  const int64_t max_samples_cap = sample_limit_per_batch();
  int64_t accumulated_samples = job_sample_size(first_job);

  const auto pressure_sample = sample_batch_pressure();
  const int coalesce_timeout_ms = resolve_coalesce_timeout_ms(
      opts_, pressure_sample.state.congested, max_job_count);
  const bool enable_wait = coalesce_timeout_ms > 0;
  const auto batch_coalesce_timeout =
      std::chrono::milliseconds(coalesce_timeout_ms);
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

[[nodiscard]] auto
BatchCollector::should_hold_job(
    const std::shared_ptr<InferenceJob>& candidate,
    const std::shared_ptr<InferenceJob>& reference,
    const std::optional<int>& target_worker) -> bool
{
  if (!candidate) {
    return false;
  }
  if (candidate->batch().has_aggregated_sub_jobs() ||
      candidate->batch().logical_job_count() > 1) {
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

inline namespace batch_collector_component_detail {
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

}  // namespace batch_collector_component_detail

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
BatchCollector::effective_batch_limit() -> int
{
  const int batch_limit =
      opts_ != nullptr ? std::max(1, opts_->batching.max_batch_size) : 1;
  if (batch_limit <= 1 || opts_ == nullptr ||
      !opts_->batching.dynamic_batching || !opts_->congestion.enabled) {
    return batch_limit;
  }

  if (!adaptive_target_initialized_) {
    adaptive_target_batch_size_ = batch_limit;
    adaptive_target_initialized_ = true;
  }

  update_adaptive_batch_target(batch_limit);
  return std::clamp(adaptive_target_batch_size_, 1, batch_limit);
}

void
BatchCollector::update_adaptive_batch_target(int batch_limit)
{
  if (batch_limit <= 1) {
    adaptive_target_batch_size_ = 1;
    adaptive_target_initialized_ = true;
    low_pressure_streak_ = 0;
    return;
  }

  adaptive_target_batch_size_ =
      std::clamp(adaptive_target_batch_size_, 1, batch_limit);

  const auto pressure_sample = sample_batch_pressure();
  if (!should_refresh_adaptive_target(pressure_sample)) {
    return;
  }
  const auto& pressure = pressure_sample.state;
  if (pressure.congested) {
    adaptive_target_batch_size_ = batch_limit;
    low_pressure_streak_ = 0;
    return;
  }

  if (pressure.high) {
    low_pressure_streak_ = 0;
    const int step = high_pressure_step(batch_limit, pressure.severe);
    adaptive_target_batch_size_ =
        std::min(batch_limit, adaptive_target_batch_size_ + step);
    return;
  }

  if (pressure.low) {
    if (low_pressure_streak_ < std::numeric_limits<int>::max()) {
      ++low_pressure_streak_;
    }
    if (low_pressure_streak_ >= low_pressure_streak_threshold()) {
      adaptive_target_batch_size_ =
          std::max(1, adaptive_target_batch_size_ - 1);
      low_pressure_streak_ = 0;
    }
    return;
  }

  low_pressure_streak_ = 0;
}

auto
BatchCollector::should_refresh_adaptive_target(
    const BatchPressureSample& pressure) -> bool
{
  if (opts_ == nullptr || !opts_->congestion.enabled) {
    return true;
  }

  if (pressure.monitor_tick.has_value()) {
    const auto monitor_tick = *pressure.monitor_tick;
    if (last_adaptive_update_marker_.has_value() &&
        monitor_tick <= *last_adaptive_update_marker_) {
      return false;
    }
    last_adaptive_update_marker_ = monitor_tick;
    return true;
  }

  const auto now = clock::now();

  if (const auto tick_interval = std::chrono::milliseconds(
          std::max(1, opts_->congestion.tick_interval_ms));
      last_adaptive_update_marker_.has_value() &&
      now - *last_adaptive_update_marker_ < tick_interval) {
    return false;
  }
  last_adaptive_update_marker_ = now;
  return true;
}

auto
BatchCollector::high_pressure_step(int batch_limit, bool severe) const -> int
{
  if (batch_limit <= 1 || opts_ == nullptr || !opts_->congestion.enabled) {
    return 1;
  }

  const int tick_interval_ms = std::max(1, opts_->congestion.tick_interval_ms);
  const int entry_horizon_ms =
      std::max(tick_interval_ms, opts_->congestion.entry_horizon_ms);
  const int entry_ticks = std::max(1, entry_horizon_ms / tick_interval_ms);

  const int base_step = std::max(1, batch_limit / entry_ticks);
  if (!severe) {
    return base_step;
  }

  const int low_ticks = low_pressure_streak_threshold();
  const int severe_step = std::max(1, batch_limit / std::max(1, low_ticks));
  return std::max(base_step, severe_step);
}

auto
BatchCollector::low_pressure_streak_threshold() const -> int
{
  if (opts_ == nullptr || !opts_->congestion.enabled) {
    return 1;
  }

  const int tick_interval_ms = std::max(1, opts_->congestion.tick_interval_ms);
  const int exit_horizon_ms =
      std::max(tick_interval_ms, opts_->congestion.exit_horizon_ms);
  return std::max(1, exit_horizon_ms / tick_interval_ms);
}

auto
BatchCollector::sample_batch_pressure() const -> BatchPressureSample
{
  BatchPressureSample sample{};
  if (opts_ == nullptr || !opts_->congestion.enabled) {
    return sample;
  }

  const auto thresholds = make_batch_pressure_thresholds(opts_->congestion);
  InternalPressureConfig internal_pressure_config{};
  internal_pressure_config.max_inflight_tasks = max_inflight_tasks_;
  internal_pressure_config.max_batch_size = opts_->batching.max_batch_size;
  const auto internal_pressure = sample_internal_pressure(
      load_prepared_depth(prepared_jobs_, prepared_mutex_),
      load_inflight_tasks(inflight_tasks_), internal_pressure_config);

  ResolvedBatchPressure pressure =
      sample_monitor_pressure(observability_, thresholds, internal_pressure)
          .value_or(sample_queue_pressure(
              observability_, queue_, thresholds, internal_pressure));
  sample.state.congested = pressure.congested;
  sample.state.high = pressure.high;
  sample.state.low = pressure.low;
  sample.state.severe = pressure.severe;
  sample.monitor_tick = pressure.monitor_tick;
  return sample;
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
  attach_input_memory_holders(master, jobs);

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
      starpu_, jobs, master, effective_batch);

  master->batch().set_effective_batch_size(effective_batch);
  install_sub_job_completion_callback(master);
  trace_formed_batch_if_enabled(
      opts_, queue_, master, jobs.size(), effective_batch);

  return master;
}

void
BatchCollector::reset_prepared_queue_state()
{
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

auto
batch_collector_is_batching_done(const BatchCollector* collector) -> bool
{
  return collector != nullptr ? collector->is_batching_done() : false;
}

auto
batch_collector_should_abort_inflight_wait(const BatchCollector* collector)
    -> bool
{
  return collector != nullptr ? collector->should_abort_inflight_wait() : false;
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
batch_collector_set_batching_done_ptr(
    BatchCollector* collector, bool* batching_done)
{
  if (collector == nullptr) {
    return;
  }
  collector->batching_done_ = batching_done;
}

void
batch_collector_set_batching_done_value(
    BatchCollector* collector, bool batching_done)
{
  if (collector == nullptr || collector->batching_done_ == nullptr) {
    return;
  }
  *collector->batching_done_ = batching_done;
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

void
batch_collector_set_after_build_job_hook(
    std::function<void(std::shared_ptr<InferenceJob>&)> hook)
{
  batching_loop_test_hooks().after_build_job = std::move(hook);
}

void
batch_collector_reset_after_build_job_hook()
{
  batching_loop_test_hooks().after_build_job = {};
}

}  // namespace task_runner_internal::testing
#endif  // SONAR_IGNORE_END
// GCOVR_EXCL_STOP

}  // namespace starpu_server
