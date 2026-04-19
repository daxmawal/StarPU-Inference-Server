#include "batching_strategy.hpp"

#include <algorithm>
#include <chrono>
#include <limits>

namespace starpu_server {
namespace {

[[nodiscard]] auto
resolve_adaptive_coalesce_timeout_ms(
    const BatchingStrategyConfig& config, bool congested,
    int target_batch_limit) -> int
{
  const int configured_timeout_ms = std::max(0, config.coalesce_timeout_ms);
  if (!config.congestion_enabled || !congested) {
    return configured_timeout_ms;
  }

  // Under congestion, keep a short coalescing window even when the configured
  // timeout is 0 so the collector can still fill toward the current target.
  const int tick_interval_ms = std::max(1, config.congestion_tick_interval_ms);
  const int per_slot_wait_ms =
      std::max(1, tick_interval_ms / std::max(1, target_batch_limit));
  return std::max(configured_timeout_ms, per_slot_wait_ms);
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

struct ResolvedBatchPressure {
  bool congested = false;
  bool high = false;
  bool low = false;
  bool severe = false;
};

class DisabledBatchingStrategy final : public BatchingStrategy {
 public:
  [[nodiscard]] auto decide(const BatchingStrategyInput& input)
      -> BatchingStrategyDecision override
  {
    static_cast<void>(input);
    return {.target_batch_limit = 1, .coalesce_timeout_ms = 0};
  }

  void reset() override {}
};

constexpr double kInternalHighRatio = 0.75;
constexpr double kInternalLowRatio = 0.25;
constexpr double kInternalSevereRatio = 0.95;
constexpr double kPreparedBacklogHighRatio = 1.0;
constexpr double kPreparedBacklogSevereRatio = 2.0;

auto
compute_queue_fill(std::size_t queue_size, std::size_t queue_capacity) -> double
{
  if (queue_capacity == 0) {
    return 0.0;
  }
  return static_cast<double>(queue_size) / static_cast<double>(queue_capacity);
}

auto
make_batch_pressure_thresholds(const BatchingStrategyConfig& config)
    -> BatchPressureThresholds
{
  BatchPressureThresholds thresholds{};
  thresholds.fill_high = std::clamp(config.congestion_fill_high, 0.0, 1.0);
  thresholds.fill_low = std::clamp(
      std::min(config.congestion_fill_low, config.congestion_fill_high), 0.0,
      thresholds.fill_high);
  thresholds.rho_high = std::max(0.0, config.congestion_rho_high);
  thresholds.rho_low =
      std::clamp(config.congestion_rho_low, 0.0, thresholds.rho_high);
  return thresholds;
}

auto
sample_internal_pressure(const BatchingStrategyInput& input)
    -> InternalPressureSnapshot
{
  InternalPressureSnapshot pressure{};
  const std::size_t total_internal_backlog =
      input.runtime.prepared_depth + input.runtime.inflight_tasks;

  if (input.runtime.max_inflight_tasks > 0) {
    const double inflight_ratio =
        static_cast<double>(input.runtime.inflight_tasks) /
        static_cast<double>(input.runtime.max_inflight_tasks);
    const double backlog_ratio = static_cast<double>(total_internal_backlog) /
                                 static_cast<double>(
                                     input.runtime.max_inflight_tasks);
    pressure.high = inflight_ratio >= kInternalHighRatio ||
                    backlog_ratio >= kInternalHighRatio;
    pressure.low = inflight_ratio <= kInternalLowRatio &&
                   backlog_ratio <= kInternalLowRatio;
    pressure.severe = inflight_ratio >= kInternalSevereRatio ||
                      backlog_ratio >= kInternalSevereRatio;
    return pressure;
  }

  const auto local_backlog_ref =
      static_cast<std::size_t>(std::max(1, input.config.batch_limit));
  const double prepared_ratio = static_cast<double>(input.runtime.prepared_depth) /
                                static_cast<double>(local_backlog_ref);
  pressure.high = prepared_ratio >= kPreparedBacklogHighRatio;
  pressure.low = input.runtime.prepared_depth == 0;
  pressure.severe = prepared_ratio >= kPreparedBacklogSevereRatio;
  return pressure;
}

auto
resolve_monitor_pressure(
    const BatchingStrategyInput& input,
    const BatchingStrategyMonitorSnapshot& monitor,
    const BatchPressureThresholds& thresholds,
    const InternalPressureSnapshot& internal_pressure) -> ResolvedBatchPressure
{
  const double queue_fill =
      compute_queue_fill(monitor.queue_size, monitor.queue_capacity);
  const bool queue_high =
      monitor.queue_capacity > 0 && queue_fill >= thresholds.fill_high;
  const bool queue_low =
      monitor.queue_capacity == 0 || queue_fill <= thresholds.fill_low;

  ResolvedBatchPressure pressure{};
  pressure.congested = input.congested;
  pressure.severe = pressure.congested || monitor.rejection_rps > 0.0 ||
                    internal_pressure.severe;
  pressure.high =
      pressure.severe || monitor.fill_ewma >= thresholds.fill_high ||
      monitor.rho_ewma >= thresholds.rho_high || queue_high ||
      internal_pressure.high;
  pressure.low = !pressure.congested && monitor.rejection_rps <= 0.0 &&
                 monitor.fill_ewma <= thresholds.fill_low &&
                 monitor.rho_ewma <= thresholds.rho_low && queue_low &&
                 internal_pressure.low;
  return pressure;
}

auto
resolve_runtime_pressure(
    const BatchingStrategyInput& input,
    const BatchPressureThresholds& thresholds,
    const InternalPressureSnapshot& internal_pressure) -> ResolvedBatchPressure
{
  const double queue_fill = compute_queue_fill(
      input.runtime.queue_size, input.runtime.queue_capacity);

  ResolvedBatchPressure pressure{};
  pressure.congested = input.congested;
  pressure.high =
      queue_fill >= thresholds.fill_high || internal_pressure.high;
  pressure.low = !pressure.congested && queue_fill <= thresholds.fill_low &&
                 internal_pressure.low;
  pressure.severe = pressure.congested || internal_pressure.severe;
  return pressure;
}

auto
resolve_batch_pressure(const BatchingStrategyInput& input)
    -> ResolvedBatchPressure
{
  if (!input.config.congestion_enabled) {
    return {};
  }

  const auto thresholds = make_batch_pressure_thresholds(input.config);
  const auto internal_pressure = sample_internal_pressure(input);
  if (input.monitor.has_value()) {
    return resolve_monitor_pressure(
        input, *input.monitor, thresholds, internal_pressure);
  }
  return resolve_runtime_pressure(input, thresholds, internal_pressure);
}

}  // namespace

auto
AdaptiveBatchingStrategy::decide(const BatchingStrategyInput& input)
    -> BatchingStrategyDecision
{
  const auto& config = input.config;
  const auto pressure = resolve_batch_pressure(input);
  const int batch_limit = std::max(1, config.batch_limit);
  const int min_batch_limit =
      std::clamp(config.min_batch_limit, 1, batch_limit);
  if (batch_limit <= min_batch_limit) {
    adaptive_target_batch_size_ = min_batch_limit;
    adaptive_target_initialized_ = true;
    low_pressure_streak_ = 0;
    return BatchingStrategyDecision{
        .target_batch_limit = min_batch_limit,
        .coalesce_timeout_ms = resolve_adaptive_coalesce_timeout_ms(
            config, pressure.congested, min_batch_limit),
    };
  }

  if (!config.congestion_enabled) {
    low_pressure_streak_ = 0;
    return BatchingStrategyDecision{
        .target_batch_limit = batch_limit,
        .coalesce_timeout_ms = std::max(0, config.coalesce_timeout_ms),
    };
  }

  if (!adaptive_target_initialized_) {
    adaptive_target_batch_size_ = batch_limit;
    adaptive_target_initialized_ = true;
  }

  update_target_batch_limit(
      input, pressure.congested, pressure.high, pressure.low, pressure.severe);
  const int target_batch_limit =
      std::clamp(adaptive_target_batch_size_, 1, batch_limit);
  return BatchingStrategyDecision{
      .target_batch_limit = target_batch_limit,
      .coalesce_timeout_ms = resolve_adaptive_coalesce_timeout_ms(
          config, pressure.congested, target_batch_limit),
  };
}

void
AdaptiveBatchingStrategy::reset()
{
  adaptive_target_batch_size_ = 1;
  adaptive_target_initialized_ = false;
  low_pressure_streak_ = 0;
  last_adaptive_update_marker_.reset();
}

void
AdaptiveBatchingStrategy::update_target_batch_limit(
    const BatchingStrategyInput& input, bool congested, bool high, bool low,
    bool severe)
{
  const auto& config = input.config;
  const int batch_limit = std::max(1, config.batch_limit);
  const int min_batch_limit =
      std::clamp(config.min_batch_limit, 1, batch_limit);
  adaptive_target_batch_size_ =
      std::clamp(adaptive_target_batch_size_, min_batch_limit, batch_limit);

  if (!should_refresh_target(input)) {
    return;
  }

  if (congested) {
    adaptive_target_batch_size_ = batch_limit;
    low_pressure_streak_ = 0;
    return;
  }

  if (high) {
    low_pressure_streak_ = 0;
    const int step = high_pressure_step(config, batch_limit, severe);
    adaptive_target_batch_size_ =
        std::min(batch_limit, adaptive_target_batch_size_ + step);
    return;
  }

  if (low) {
    if (low_pressure_streak_ < std::numeric_limits<int>::max()) {
      ++low_pressure_streak_;
    }
    if (low_pressure_streak_ >= low_pressure_streak_threshold(config)) {
      adaptive_target_batch_size_ =
          std::max(min_batch_limit, adaptive_target_batch_size_ - 1);
      low_pressure_streak_ = 0;
    }
    return;
  }

  low_pressure_streak_ = 0;
}

auto
AdaptiveBatchingStrategy::should_refresh_target(const BatchingStrategyInput& input)
    -> bool
{
  const auto& config = input.config;
  if (!config.congestion_enabled) {
    return true;
  }

  if (input.monitor.has_value() && input.monitor->tick.has_value()) {
    const auto monitor_tick = *input.monitor->tick;
    if (last_adaptive_update_marker_.has_value() &&
        monitor_tick <= *last_adaptive_update_marker_) {
      return false;
    }
    last_adaptive_update_marker_ = monitor_tick;
    return true;
  }

  const auto now = MonotonicClock::now();
  const auto tick_interval = std::chrono::milliseconds(
      std::max(1, config.congestion_tick_interval_ms));
  if (last_adaptive_update_marker_.has_value() &&
      now - *last_adaptive_update_marker_ < tick_interval) {
    return false;
  }
  last_adaptive_update_marker_ = now;
  return true;
}

auto
AdaptiveBatchingStrategy::high_pressure_step(
    const BatchingStrategyConfig& config, int batch_limit,
    bool severe) -> int
{
  if (batch_limit <= 1 || !config.congestion_enabled) {
    return 1;
  }

  const int tick_interval_ms = std::max(1, config.congestion_tick_interval_ms);
  const int entry_horizon_ms =
      std::max(tick_interval_ms, config.congestion_entry_horizon_ms);
  const int entry_ticks = std::max(1, entry_horizon_ms / tick_interval_ms);

  const int base_step = std::max(1, batch_limit / entry_ticks);
  if (!severe) {
    return base_step;
  }

  const int low_ticks = low_pressure_streak_threshold(config);
  const int severe_step = std::max(1, batch_limit / std::max(1, low_ticks));
  return std::max(base_step, severe_step);
}

auto
AdaptiveBatchingStrategy::low_pressure_streak_threshold(
    const BatchingStrategyConfig& config) -> int
{
  if (!config.congestion_enabled) {
    return 1;
  }

  const int tick_interval_ms = std::max(1, config.congestion_tick_interval_ms);
  const int exit_horizon_ms =
      std::max(tick_interval_ms, config.congestion_exit_horizon_ms);
  return std::max(1, exit_horizon_ms / tick_interval_ms);
}

auto
FixedBatchingStrategy::decide(const BatchingStrategyInput& input)
    -> BatchingStrategyDecision
{
  const auto& config = input.config;
  return BatchingStrategyDecision{
      .target_batch_limit = std::max(1, config.batch_limit),
      .coalesce_timeout_ms = std::max(0, config.coalesce_timeout_ms),
  };
}

auto
make_batching_strategy(BatchingStrategyKind kind)
    -> std::unique_ptr<BatchingStrategy>
{
  switch (kind) {
    case BatchingStrategyKind::Disabled:
      return std::make_unique<DisabledBatchingStrategy>();
    case BatchingStrategyKind::Adaptive:
      return std::make_unique<AdaptiveBatchingStrategy>();
    case BatchingStrategyKind::Fixed:
      return std::make_unique<FixedBatchingStrategy>();
  }
  return std::make_unique<DisabledBatchingStrategy>();
}

}  // namespace starpu_server
