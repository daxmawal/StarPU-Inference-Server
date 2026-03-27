#pragma once

#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <stop_token>
#include <thread>
#include <vector>

namespace starpu_server {
class InferenceQueue;
class MetricsRecorder;
class BatchingTraceLogger;
}  // namespace starpu_server

namespace starpu_server::congestion {
inline constexpr double kDefaultQueueLatencyBudgetRatio = 0.30;
inline constexpr double kDefaultE2EWarnRatio = 0.90;
inline constexpr double kDefaultE2EOkRatio = 0.80;
inline constexpr double kDefaultFillHigh = 0.80;
inline constexpr double kDefaultFillLow = 0.60;
inline constexpr double kDefaultRhoHigh = 1.05;
inline constexpr double kDefaultRhoLow = 0.90;
inline constexpr double kDefaultEwmaAlpha = 0.2;
inline constexpr std::chrono::seconds kDefaultTickInterval{1};
inline constexpr std::chrono::seconds kDefaultEntryHorizon{5};
inline constexpr std::chrono::seconds kDefaultExitHorizon{15};

struct Config {
  bool enabled{true};
  double latency_slo_ms{0.0};
  double queue_latency_budget_ms{0.0};
  double queue_latency_budget_ratio{kDefaultQueueLatencyBudgetRatio};
  double e2e_warn_ratio{kDefaultE2EWarnRatio};
  double e2e_ok_ratio{kDefaultE2EOkRatio};
  double fill_high{kDefaultFillHigh};
  double fill_low{kDefaultFillLow};
  double rho_high{kDefaultRhoHigh};
  double rho_low{kDefaultRhoLow};
  double alpha{kDefaultEwmaAlpha};
  std::chrono::milliseconds tick_interval{kDefaultTickInterval};
  std::chrono::milliseconds entry_horizon{kDefaultEntryHorizon};
  std::chrono::milliseconds exit_horizon{kDefaultExitHorizon};
};

struct Snapshot {
  bool congestion{false};
  double lambda_rps{0.0};
  double mu_rps{0.0};
  double rho_ewma{0.0};
  double fill_ewma{0.0};
  double dqueue_ewma{0.0};
  double queue_p95_ms{0.0};
  double queue_p99_ms{0.0};
  double e2e_p95_ms{0.0};
  double e2e_p99_ms{0.0};
  double rejection_rps{0.0};
  double score{0.0};
  std::size_t queue_size{0};
  std::size_t queue_capacity{0};
  std::chrono::steady_clock::time_point last_tick;
};

struct CompletionLatencies {
  double queue_latency_ms{0.0};
  double e2e_latency_ms{0.0};
};

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
struct LatencyFlagResult {
  bool danger;
  bool ok;
};

struct NormalizedConfigResult {
  std::chrono::milliseconds tick_interval;
  std::chrono::milliseconds entry_horizon;
  std::chrono::milliseconds exit_horizon;
  std::optional<double> queue_budget_ms;
};

struct QueuePressureScoreTestArgs {
  double fill_high{0.0};
  double fill_low{0.0};
  double fill_smoothed{0.0};
};

struct LatencyPressureScoreTestArgs {
  double latency_slo_ms{0.0};
  double e2e_ok_ratio{0.0};
  std::optional<double> e2e_p95;
};

struct CapacityPressureScoreTestArgs {
  double rho_high{0.0};
  double rho_low{0.0};
  double rho_smoothed{0.0};
};
#endif  // SONAR_IGNORE_END

class Monitor {
 public:
  class Impl {
   public:
    Impl() = default;
    virtual ~Impl() = default;
    Impl(const Impl&) = delete;
    auto operator=(const Impl&) -> Impl& = delete;
    Impl(Impl&&) = delete;
    auto operator=(Impl&&) -> Impl& = delete;
    virtual void start() = 0;
    virtual void shutdown() = 0;
    virtual void record_arrival(std::size_t count) = 0;
    virtual void record_completion(
        std::size_t logical_jobs, CompletionLatencies latencies) = 0;
    virtual void record_rejection(std::size_t count) = 0;
    [[nodiscard]] virtual auto snapshot() const -> Snapshot = 0;
    [[nodiscard]] virtual auto congested() const -> bool = 0;
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
    [[nodiscard]] virtual auto arrivals_for_test() const -> std::size_t = 0;
    [[nodiscard]] virtual auto rejections_for_test() const -> std::size_t = 0;
    virtual auto evaluate_latency_flags_for_test(
        std::optional<double> queue_p95,
        std::optional<double> e2e_p95) -> LatencyFlagResult = 0;
    [[nodiscard]] virtual auto compute_queue_pressure_score_for_test(
        double fill_smoothed) const -> double = 0;
    [[nodiscard]] virtual auto compute_latency_pressure_score_for_test(
        std::optional<double> e2e_p95,
        std::optional<double> queue_p95) const -> double = 0;
    [[nodiscard]] virtual auto normalized_config_for_test() const
        -> NormalizedConfigResult = 0;
    [[nodiscard]] virtual auto compute_capacity_pressure_score_for_test(
        double rho_smoothed) const -> double = 0;
#endif  // SONAR_IGNORE_END
  };

  explicit Monitor(
      InferenceQueue* queue, Config config = {},
      std::shared_ptr<MetricsRecorder> metrics = {},
      std::shared_ptr<BatchingTraceLogger> tracer = {});
  ~Monitor() = default;
  Monitor(const Monitor&) = delete;
  auto operator=(const Monitor&) -> Monitor& = delete;
  Monitor(Monitor&&) = delete;
  auto operator=(Monitor&&) -> Monitor& = delete;

  void start();
  void shutdown();
  void record_arrival(std::size_t count = 1);
  void record_completion(
      std::size_t logical_jobs, CompletionLatencies latencies);
  void record_rejection(std::size_t count = 1);
  [[nodiscard]] auto snapshot() const -> Snapshot;
  [[nodiscard]] auto congested() const -> bool;

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  [[nodiscard]] auto arrivals_for_test() const -> std::size_t;
  [[nodiscard]] auto rejections_for_test() const -> std::size_t;
  auto evaluate_latency_flags_for_test(
      std::optional<double> queue_p95,
      std::optional<double> e2e_p95 = std::nullopt) -> LatencyFlagResult;
  [[nodiscard]] auto compute_queue_pressure_score_for_test(
      double fill_smoothed) const -> double;
  [[nodiscard]] auto compute_latency_pressure_score_for_test(
      std::optional<double> e2e_p95,
      std::optional<double> queue_p95 = std::nullopt) const -> double;
  [[nodiscard]] auto normalized_config_for_test() const
      -> NormalizedConfigResult;
  [[nodiscard]] auto compute_capacity_pressure_score_for_test(
      double rho_smoothed) const -> double;
#endif  // SONAR_IGNORE_END

 private:
  std::shared_ptr<Impl> impl_;
};

auto start(InferenceQueue* queue, Config config = {}) -> bool;
void shutdown();
void record_arrival(std::size_t count = 1);
void record_completion(std::size_t logical_jobs, CompletionLatencies latencies);
void record_rejection(std::size_t count = 1);
auto snapshot() -> std::optional<Snapshot>;
auto is_congested() -> bool;

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
auto percentile_for_test(std::vector<double> samples, double pct)
    -> std::optional<double>;
auto update_ewma_for_test(
    std::optional<double>& state, double sample, double alpha) -> double;
auto record_arrival_for_test(std::size_t count) -> std::size_t;
auto record_rejection_for_test(std::size_t count) -> std::size_t;
auto evaluate_latency_flags_for_test(
    double queue_budget_ms,
    std::optional<double> queue_p95) -> LatencyFlagResult;
auto compute_queue_pressure_score_for_test(
    const QueuePressureScoreTestArgs& args) -> double;
auto compute_latency_pressure_score_for_test(
    const LatencyPressureScoreTestArgs& args) -> double;
auto compute_latency_pressure_score_queue_budget_for_test(
    double queue_budget_ms, std::optional<double> queue_p95) -> double;
auto normalize_config_for_test(Config cfg) -> NormalizedConfigResult;
auto compute_capacity_pressure_score_for_test(
    const CapacityPressureScoreTestArgs& args) -> double;
#endif  // SONAR_IGNORE_END

}  // namespace starpu_server::congestion
