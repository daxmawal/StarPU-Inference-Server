#pragma once

#include <chrono>
#include <cstddef>
#include <optional>

namespace starpu_server {
class InferenceQueue;
}  // namespace starpu_server

namespace starpu_server::congestion {

struct Config {
  bool enabled{true};
  double latency_slo_ms{0.0};
  double queue_latency_budget_ms{0.0};
  double queue_latency_budget_ratio{0.30};
  double e2e_warn_ratio{0.90};
  double e2e_ok_ratio{0.80};
  double fill_high{0.80};
  double fill_low{0.60};
  double rho_high{1.05};
  double rho_low{0.90};
  double alpha{0.2};
  std::chrono::milliseconds tick_interval{std::chrono::seconds(1)};
  std::chrono::milliseconds entry_horizon{std::chrono::seconds(5)};
  std::chrono::milliseconds exit_horizon{std::chrono::seconds(15)};
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
  std::chrono::steady_clock::time_point last_tick{};
};

bool start(InferenceQueue* queue, Config config = {});
void shutdown();
void record_arrival(std::size_t count = 1);
void record_completion(
    std::size_t logical_jobs, double queue_latency_ms, double e2e_latency_ms);
void record_rejection(std::size_t count = 1);
auto snapshot() -> std::optional<Snapshot>;
bool is_congested();

}  // namespace starpu_server::congestion
