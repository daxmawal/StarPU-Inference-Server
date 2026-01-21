#pragma once

#include <chrono>
#include <cstddef>
#include <optional>

namespace starpu_server {
class InferenceQueue;
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

auto start(InferenceQueue* queue, Config config = {}) -> bool;
void shutdown();
void record_arrival(std::size_t count = 1);
void record_completion(std::size_t logical_jobs, CompletionLatencies latencies);
void record_rejection(std::size_t count = 1);
auto snapshot() -> std::optional<Snapshot>;
auto is_congested() -> bool;

}  // namespace starpu_server::congestion
