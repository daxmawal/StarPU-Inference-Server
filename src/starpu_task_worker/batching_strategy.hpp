#pragma once

#include <cstddef>
#include <memory>
#include <optional>

#include "utils/monotonic_clock.hpp"
#include "utils/runtime_config.hpp"

namespace starpu_server {

struct BatchingStrategyConfig {
  int min_batch_limit = 1;
  int batch_limit = 1;
  int coalesce_timeout_ms = 0;
  bool congestion_enabled = false;
  double congestion_fill_high = 0.0;
  double congestion_fill_low = 0.0;
  double congestion_rho_high = 0.0;
  double congestion_rho_low = 0.0;
  int congestion_tick_interval_ms = 1;
  int congestion_entry_horizon_ms = 1;
  int congestion_exit_horizon_ms = 1;
};

struct BatchingStrategyRuntimeState {
  std::size_t queue_size = 0;
  std::size_t queue_capacity = 0;
  std::size_t prepared_depth = 0;
  std::size_t inflight_tasks = 0;
  std::size_t max_inflight_tasks = 0;
};

struct BatchingStrategyMonitorSnapshot {
  double fill_ewma = 0.0;
  double rho_ewma = 0.0;
  double rejection_rps = 0.0;
  std::size_t queue_size = 0;
  std::size_t queue_capacity = 0;
  std::optional<MonotonicClock::time_point> tick;
};

struct BatchingStrategyInput {
  BatchingStrategyConfig config{};
  BatchingStrategyRuntimeState runtime{};
  bool congested = false;
  std::optional<BatchingStrategyMonitorSnapshot> monitor;
};

struct BatchingStrategyDecision {
  int target_batch_limit = 1;
  int coalesce_timeout_ms = 0;
};

class BatchingStrategy {
 public:
  BatchingStrategy() = default;
  virtual ~BatchingStrategy() = default;
  BatchingStrategy(const BatchingStrategy&) = default;
  auto operator=(const BatchingStrategy&) -> BatchingStrategy& = default;
  BatchingStrategy(BatchingStrategy&&) = default;
  auto operator=(BatchingStrategy&&) -> BatchingStrategy& = default;

  [[nodiscard]] virtual auto decide(const BatchingStrategyInput& input)
      -> BatchingStrategyDecision = 0;

  virtual void reset() = 0;
};

class AdaptiveBatchingStrategy final : public BatchingStrategy {
 public:
  [[nodiscard]] auto decide(const BatchingStrategyInput& input)
      -> BatchingStrategyDecision override;

  void reset() override;

 private:
  void update_target_batch_limit(
      const BatchingStrategyInput& input, bool congested, bool high, bool low,
      bool severe);

  [[nodiscard]] auto should_refresh_target(const BatchingStrategyInput& input)
      -> bool;

  [[nodiscard]] static auto high_pressure_step(
      const BatchingStrategyConfig& config, int batch_limit,
      bool severe) -> int;

  [[nodiscard]] static auto low_pressure_streak_threshold(
      const BatchingStrategyConfig& config) -> int;

  int adaptive_target_batch_size_ = 1;
  bool adaptive_target_initialized_ = false;
  int low_pressure_streak_ = 0;
  std::optional<MonotonicClock::time_point> last_adaptive_update_marker_;
};

class FixedBatchingStrategy final : public BatchingStrategy {
 public:
  [[nodiscard]] auto decide(const BatchingStrategyInput& input)
      -> BatchingStrategyDecision override;

  void reset() override {}
};

[[nodiscard]] auto make_batching_strategy(BatchingStrategyKind kind)
    -> std::unique_ptr<BatchingStrategy>;

}  // namespace starpu_server
