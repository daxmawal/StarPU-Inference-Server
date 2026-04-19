#include "batching_strategy_input_provider.hpp"

#include <optional>

#include "inference_queue.hpp"
#include "monitoring/congestion_monitor.hpp"
#include "monitoring/runtime_observability.hpp"

namespace starpu_server {
namespace {

auto
resolve_batching_strategy_config(const RuntimeConfig* opts)
    -> BatchingStrategyConfig
{
  BatchingStrategyConfig config{};
  if (opts == nullptr) {
    return config;
  }

  config.min_batch_limit = resolved_adaptive_min_batch_size(opts->batching);
  config.batch_limit = resolved_batch_capacity(opts->batching);
  config.coalesce_timeout_ms = opts->batching.batch_coalesce_timeout_ms;
  config.congestion_enabled = opts->congestion.enabled;
  config.congestion_fill_high = opts->congestion.fill_high;
  config.congestion_fill_low = opts->congestion.fill_low;
  config.congestion_rho_high = opts->congestion.rho_high;
  config.congestion_rho_low = opts->congestion.rho_low;
  config.congestion_tick_interval_ms = opts->congestion.tick_interval_ms;
  config.congestion_entry_horizon_ms = opts->congestion.entry_horizon_ms;
  config.congestion_exit_horizon_ms = opts->congestion.exit_horizon_ms;
  return config;
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
current_congested(
    const std::shared_ptr<RuntimeObservability>& observability) -> bool
{
  if (observability != nullptr && observability->congestion_monitor != nullptr) {
    return observability->congestion_monitor->congested();
  }
  return congestion::is_congested();
}

auto
sample_monitor_snapshot(
    const std::shared_ptr<RuntimeObservability>& observability)
    -> std::optional<BatchingStrategyMonitorSnapshot>
{
  const auto* monitor =
      observability != nullptr ? observability->congestion_monitor.get()
                               : nullptr;
  const auto snapshot =
      monitor != nullptr
          ? std::optional<congestion::Snapshot>(monitor->snapshot())
          : congestion::snapshot();
  if (!snapshot.has_value()) {
    return std::nullopt;
  }

  const auto& values = *snapshot;
  return BatchingStrategyMonitorSnapshot{
      .fill_ewma = values.fill_ewma,
      .rho_ewma = values.rho_ewma,
      .rejection_rps = values.rejection_rps,
      .queue_size = values.queue_size,
      .queue_capacity = values.queue_capacity,
      .tick = values.last_tick,
  };
}

}  // namespace

RuntimeBatchingStrategyInputProvider::RuntimeBatchingStrategyInputProvider(
    const RuntimeConfig* opts, InferenceQueue* queue,
    std::shared_ptr<RuntimeObservability> observability,
    const std::deque<std::shared_ptr<InferenceJob>>* prepared_jobs,
    std::mutex* prepared_mutex,
    const std::atomic<std::size_t>* inflight_tasks,
    std::size_t max_inflight_tasks)
    : opts_(opts), queue_(queue), observability_(std::move(observability)),
      prepared_jobs_(prepared_jobs), prepared_mutex_(prepared_mutex),
      inflight_tasks_(inflight_tasks),
      max_inflight_tasks_(max_inflight_tasks)
{
}

auto
RuntimeBatchingStrategyInputProvider::sample() const -> BatchingStrategyInput
{
  BatchingStrategyInput input{};
  input.config = resolve_batching_strategy_config(opts_);
  input.runtime.prepared_depth =
      load_prepared_depth(prepared_jobs_, prepared_mutex_);
  input.runtime.inflight_tasks = load_inflight_tasks(inflight_tasks_);
  input.runtime.max_inflight_tasks = max_inflight_tasks_;
  if (queue_ != nullptr) {
    input.runtime.queue_size = queue_->size();
    input.runtime.queue_capacity = queue_->capacity();
  }

  if (!input.config.congestion_enabled) {
    return input;
  }

  input.congested = current_congested(observability_);
  input.monitor = sample_monitor_snapshot(observability_);
  return input;
}

}  // namespace starpu_server
