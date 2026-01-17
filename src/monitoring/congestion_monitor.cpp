#include "monitoring/congestion_monitor.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <stop_token>
#include <thread>
#include <utility>
#include <vector>

#include "logger.hpp"
#include "monitoring/metrics.hpp"
#include "starpu_task_worker/inference_queue.hpp"

namespace starpu_server::congestion {
namespace {

constexpr double kEpsilon = 1e-9;
constexpr double kMaxRho = 1'000.0;

auto
percentile(std::vector<double>& samples, double pct) -> std::optional<double>
{
  if (samples.empty()) {
    return std::nullopt;
  }
  if (pct <= 0.0) {
    return *std::min_element(samples.begin(), samples.end());
  }
  if (pct >= 100.0) {
    return *std::max_element(samples.begin(), samples.end());
  }

  std::ranges::sort(samples);
  const double position =
      (pct / 100.0) * static_cast<double>(samples.size() - 1U);
  const auto lower_index = static_cast<std::size_t>(std::floor(position));
  const auto upper_index = static_cast<std::size_t>(std::ceil(position));
  if (lower_index == upper_index) {
    return samples[lower_index];
  }
  const double fraction = position - static_cast<double>(lower_index);
  return std::lerp(samples[lower_index], samples[upper_index], fraction);
}

auto
update_ewma(std::optional<double>& state, double sample, double alpha) -> double
{
  if (!std::isfinite(sample)) {
    sample = 0.0;
  }
  if (!state.has_value()) {
    state = sample;
    return sample;
  }
  const double smoothed =
      alpha * sample + (1.0 - alpha) * state.value_or(sample);
  state = smoothed;
  return smoothed;
}

auto
update_optional_ewma(
    std::optional<double>& state, std::optional<double> sample,
    double alpha) -> std::optional<double>
{
  if (!sample.has_value()) {
    return state;
  }
  update_ewma(state, *sample, alpha);
  return state;
}

class CongestionMonitor {
 public:
  CongestionMonitor(InferenceQueue* queue, Config cfg)
      : queue_(queue), cfg_(std::move(cfg)),
        last_queue_size_(queue_ != nullptr ? queue_->size() : 0),
        last_tick_(std::chrono::steady_clock::now())
  {
    normalize_config();
    snapshot_.queue_size = last_queue_size_;
    snapshot_.queue_capacity = queue_ != nullptr ? queue_->capacity() : 0;
    snapshot_.last_tick = last_tick_;
  }

  void start()
  {
    worker_ =
        std::jthread([this](const std::stop_token& stop) { tick_loop(stop); });
  }

  void shutdown()
  {
    if (worker_.joinable()) {
      worker_.request_stop();
      worker_.join();
    }
  }

  void record_arrival(std::size_t count)
  {
    if (count == 0) {
      return;
    }
    arrivals_.fetch_add(count, std::memory_order_release);
  }

  void record_completion(
      std::size_t logical_jobs, double queue_latency_ms, double e2e_latency_ms)
  {
    if (logical_jobs > 0) {
      completions_.fetch_add(logical_jobs, std::memory_order_release);
    }
    if (queue_latency_ms > 0.0 || e2e_latency_ms > 0.0) {
      std::scoped_lock lock(samples_mutex_);
      if (queue_latency_ms > 0.0) {
        queue_latency_ms_.push_back(queue_latency_ms);
      }
      if (e2e_latency_ms > 0.0) {
        e2e_latency_ms_.push_back(e2e_latency_ms);
      }
    }
  }

  void record_rejection(std::size_t count)
  {
    if (count == 0) {
      return;
    }
    rejections_.fetch_add(count, std::memory_order_release);
  }

  [[nodiscard]] auto snapshot() const -> Snapshot
  {
    std::scoped_lock lock(snapshot_mutex_);
    return snapshot_;
  }

  [[nodiscard]] auto congested() const -> bool
  {
    return congested_flag_.load(std::memory_order_acquire);
  }

 private:
  void normalize_config()
  {
    if (cfg_.tick_interval <= std::chrono::milliseconds::zero()) {
      cfg_.tick_interval = std::chrono::seconds(1);
    }
    if (cfg_.entry_horizon < cfg_.tick_interval) {
      cfg_.entry_horizon = cfg_.tick_interval;
    }
    if (cfg_.exit_horizon < cfg_.tick_interval) {
      cfg_.exit_horizon = cfg_.tick_interval * 2;
    }

    if (cfg_.latency_slo_ms > 0.0) {
      warn_latency_ms_ = cfg_.latency_slo_ms * cfg_.e2e_warn_ratio;
      ok_latency_ms_ = cfg_.latency_slo_ms * cfg_.e2e_ok_ratio;
    }
    if (cfg_.queue_latency_budget_ms > 0.0) {
      queue_budget_ms_ = cfg_.queue_latency_budget_ms;
    } else if (
        cfg_.latency_slo_ms > 0.0 && cfg_.queue_latency_budget_ratio > 0.0) {
      queue_budget_ms_ = cfg_.latency_slo_ms * cfg_.queue_latency_budget_ratio;
    }
  }

  void tick_loop(const std::stop_token& stop)
  {
    while (!stop.stop_requested()) {
      run_tick();
      const auto sleep_step = std::chrono::milliseconds(50);
      for (auto slept = std::chrono::milliseconds::zero();
           slept < cfg_.tick_interval && !stop.stop_requested();
           slept += sleep_step) {
        std::this_thread::sleep_for(
            std::min(sleep_step, cfg_.tick_interval - slept));
      }
    }
  }

  void run_tick()
  {
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - last_tick_);
    const double dt_seconds = std::max(
        std::chrono::duration<double>(elapsed).count(),
        std::chrono::duration<double>(cfg_.tick_interval).count());
    last_tick_ = now;

    const auto arrivals = arrivals_.exchange(0, std::memory_order_acq_rel);
    const auto completions =
        completions_.exchange(0, std::memory_order_acq_rel);
    const auto rejections = rejections_.exchange(0, std::memory_order_acq_rel);

    const double lambda_rps = dt_seconds > kEpsilon
                                  ? static_cast<double>(arrivals) / dt_seconds
                                  : 0.0;
    const double mu_rps = dt_seconds > kEpsilon
                              ? static_cast<double>(completions) / dt_seconds
                              : 0.0;
    const double rejection_rps =
        dt_seconds > kEpsilon ? static_cast<double>(rejections) / dt_seconds
                              : 0.0;

    const std::size_t queue_capacity =
        queue_ != nullptr ? queue_->capacity() : 0U;
    const std::size_t queue_size = queue_ != nullptr ? queue_->size() : 0U;
    const double fill_ratio = queue_capacity > 0
                                  ? std::clamp(
                                        static_cast<double>(queue_size) /
                                            static_cast<double>(queue_capacity),
                                        0.0, 1.0)
                                  : 0.0;
    const double fill_smoothed =
        update_ewma(fill_ewma_, fill_ratio, cfg_.alpha);

    const double dqueue_dt = dt_seconds > kEpsilon
                                 ? (static_cast<double>(queue_size) -
                                    static_cast<double>(last_queue_size_)) /
                                       dt_seconds
                                 : 0.0;
    last_queue_size_ = queue_size;
    const double dqueue_smoothed =
        update_ewma(dqueue_ewma_, dqueue_dt, cfg_.alpha);

    double rho_sample = 0.0;
    if (mu_rps > 0.0) {
      rho_sample = lambda_rps / mu_rps;
    } else if (lambda_rps > 0.0) {
      rho_sample = kMaxRho;
    }
    const double rho_smoothed =
        update_ewma(rho_ewma_, std::min(rho_sample, kMaxRho), cfg_.alpha);

    std::vector<double> queue_samples;
    std::vector<double> e2e_samples;
    {
      std::scoped_lock lock(samples_mutex_);
      queue_samples.swap(queue_latency_ms_);
      e2e_samples.swap(e2e_latency_ms_);
    }

    const auto queue_p95 = percentile(queue_samples, 95.0);
    const auto queue_p99 = percentile(queue_samples, 99.0);
    const auto e2e_p95 = percentile(e2e_samples, 95.0);
    const auto e2e_p99 = percentile(e2e_samples, 99.0);

    const auto queue_p95_smoothed =
        update_optional_ewma(queue_p95_ewma_, queue_p95, cfg_.alpha);
    const auto queue_p99_smoothed =
        update_optional_ewma(queue_p99_ewma_, queue_p99, cfg_.alpha);
    const auto e2e_p95_smoothed =
        update_optional_ewma(e2e_p95_ewma_, e2e_p95, cfg_.alpha);
    const auto e2e_p99_smoothed =
        update_optional_ewma(e2e_p99_ewma_, e2e_p99, cfg_.alpha);

    const bool panic = rejections > 0;
    const bool under_provisioned = rho_smoothed > cfg_.rho_high;
    const bool queue_pressure =
        fill_smoothed > cfg_.fill_high && dqueue_smoothed > 0.0;
    bool latency_danger = false;
    if (warn_latency_ms_ > 0.0 && e2e_p95_smoothed.has_value()) {
      latency_danger = *e2e_p95_smoothed > warn_latency_ms_;
    } else if (queue_budget_ms_.has_value() && queue_p95_smoothed.has_value()) {
      latency_danger = *queue_p95_smoothed > *queue_budget_ms_;
    }
    const bool entry_condition =
        under_provisioned || queue_pressure || latency_danger;

    bool latency_ok = true;
    if (ok_latency_ms_ > 0.0 && e2e_p95_smoothed.has_value()) {
      latency_ok = *e2e_p95_smoothed < ok_latency_ms_;
    } else if (queue_budget_ms_.has_value() && queue_p95_smoothed.has_value()) {
      latency_ok = *queue_p95_smoothed < *queue_budget_ms_;
    }
    const bool exit_condition = fill_smoothed < cfg_.fill_low &&
                                rho_smoothed < cfg_.rho_low && latency_ok;

    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
    if (panic) {
      set_congested(true);
      entry_accumulator_ = cfg_.entry_horizon;
      exit_accumulator_ = std::chrono::milliseconds::zero();
    } else if (!congested()) {
      if (entry_condition) {
        entry_accumulator_ += elapsed_ms;
      } else {
        entry_accumulator_ = std::chrono::milliseconds::zero();
      }
      if (entry_accumulator_ >= cfg_.entry_horizon) {
        set_congested(true);
        exit_accumulator_ = std::chrono::milliseconds::zero();
      }
    } else {
      if (exit_condition) {
        exit_accumulator_ += elapsed_ms;
      } else {
        exit_accumulator_ = std::chrono::milliseconds::zero();
      }
      if (exit_accumulator_ >= cfg_.exit_horizon) {
        set_congested(false);
        entry_accumulator_ = std::chrono::milliseconds::zero();
      }
    }

    const double queue_pressure_score =
        cfg_.fill_high > cfg_.fill_low
            ? std::clamp(
                  (fill_smoothed - cfg_.fill_low) /
                      std::max(cfg_.fill_high - cfg_.fill_low, kEpsilon),
                  0.0, 1.0)
            : 0.0;

    double latency_pressure_score = 0.0;
    if (warn_latency_ms_ > 0.0 && e2e_p95_smoothed.has_value()) {
      const double upper = cfg_.latency_slo_ms * 1.1;
      const double lower = ok_latency_ms_;
      if (upper > lower) {
        latency_pressure_score =
            std::clamp((*e2e_p95_smoothed - lower) / (upper - lower), 0.0, 1.0);
      }
    } else if (queue_budget_ms_.has_value() && queue_p95_smoothed.has_value()) {
      const double upper = *queue_budget_ms_ * 1.2;
      const double lower = *queue_budget_ms_;
      if (upper > lower) {
        latency_pressure_score = std::clamp(
            (*queue_p95_smoothed - lower) / (upper - lower), 0.0, 1.0);
      }
    }

    const double capacity_pressure_score =
        cfg_.rho_high > cfg_.rho_low
            ? std::clamp(
                  (rho_smoothed - cfg_.rho_low) /
                      std::max(cfg_.rho_high - cfg_.rho_low, kEpsilon),
                  0.0, 1.0)
            : 0.0;

    const double score = std::max(
        {queue_pressure_score, latency_pressure_score,
         capacity_pressure_score});

    Snapshot snap{};
    snap.congestion = congested();
    snap.lambda_rps = lambda_rps;
    snap.mu_rps = mu_rps;
    snap.rho_ewma = rho_smoothed;
    snap.fill_ewma = fill_smoothed;
    snap.dqueue_ewma = dqueue_smoothed;
    snap.queue_p95_ms = queue_p95_smoothed.value_or(0.0);
    snap.queue_p99_ms = queue_p99_smoothed.value_or(0.0);
    snap.e2e_p95_ms = e2e_p95_smoothed.value_or(0.0);
    snap.e2e_p99_ms = e2e_p99_smoothed.value_or(0.0);
    snap.rejection_rps = rejection_rps;
    snap.score = score;
    snap.queue_size = queue_size;
    snap.queue_capacity = queue_capacity;
    snap.last_tick = now;

    {
      std::scoped_lock lock(snapshot_mutex_);
      snapshot_ = snap;
    }

    set_congestion_flag(snap.congestion);
    set_congestion_score(score);
    set_congestion_arrival_rate(lambda_rps);
    set_congestion_completion_rate(mu_rps);
    set_congestion_rejection_rate(rejection_rps);
    set_congestion_rho(rho_smoothed);
    set_congestion_fill_ewma(fill_smoothed);
    set_congestion_queue_growth_rate(dqueue_smoothed);
    if (queue_p95_smoothed.has_value()) {
      set_congestion_queue_latency_p95(*queue_p95_smoothed);
    } else {
      set_congestion_queue_latency_p95(0.0);
    }
    if (queue_p99_smoothed.has_value()) {
      set_congestion_queue_latency_p99(*queue_p99_smoothed);
    } else {
      set_congestion_queue_latency_p99(0.0);
    }
    if (e2e_p95_smoothed.has_value()) {
      set_congestion_e2e_latency_p95(*e2e_p95_smoothed);
    } else {
      set_congestion_e2e_latency_p95(0.0);
    }
    if (e2e_p99_smoothed.has_value()) {
      set_congestion_e2e_latency_p99(*e2e_p99_smoothed);
    } else {
      set_congestion_e2e_latency_p99(0.0);
    }
  }

  void set_congested(bool state)
  {
    congested_flag_.store(state, std::memory_order_release);
  }

  InferenceQueue* queue_;
  Config cfg_;
  std::jthread worker_;
  std::atomic<bool> congested_flag_{false};
  std::atomic<std::uint64_t> arrivals_{0};
  std::atomic<std::uint64_t> completions_{0};
  std::atomic<std::uint64_t> rejections_{0};
  mutable std::mutex samples_mutex_;
  std::vector<double> queue_latency_ms_;
  std::vector<double> e2e_latency_ms_;
  mutable std::mutex snapshot_mutex_;
  Snapshot snapshot_;
  std::optional<double> fill_ewma_;
  std::optional<double> dqueue_ewma_;
  std::optional<double> rho_ewma_;
  std::optional<double> queue_p95_ewma_;
  std::optional<double> queue_p99_ewma_;
  std::optional<double> e2e_p95_ewma_;
  std::optional<double> e2e_p99_ewma_;
  std::size_t last_queue_size_{0};
  std::chrono::steady_clock::time_point last_tick_;
  std::chrono::milliseconds entry_accumulator_{0};
  std::chrono::milliseconds exit_accumulator_{0};
  std::optional<double> queue_budget_ms_;
  double warn_latency_ms_{0.0};
  double ok_latency_ms_{0.0};
};

auto
monitor_atomic() -> std::atomic<std::shared_ptr<CongestionMonitor>>&
{
  static std::atomic<std::shared_ptr<CongestionMonitor>> instance{nullptr};
  return instance;
}

}  // namespace

bool
start(InferenceQueue* queue, Config config)
{
  if (queue == nullptr || !config.enabled) {
    monitor_atomic().store(nullptr, std::memory_order_release);
    if (queue == nullptr && config.enabled) {
      log_warning("Congestion monitor disabled: missing inference queue");
    }
    return false;
  }
  auto monitor = std::make_shared<CongestionMonitor>(queue, std::move(config));
  monitor->start();
  monitor_atomic().store(monitor, std::memory_order_release);
  return true;
}

void
shutdown()
{
  auto monitor = monitor_atomic().exchange(nullptr, std::memory_order_acq_rel);
  if (monitor != nullptr) {
    monitor->shutdown();
  }
}

void
record_arrival(std::size_t count)
{
  auto monitor = monitor_atomic().load(std::memory_order_acquire);
  if (monitor != nullptr) {
    monitor->record_arrival(count);
  }
}

void
record_completion(
    std::size_t logical_jobs, double queue_latency_ms, double e2e_latency_ms)
{
  auto monitor = monitor_atomic().load(std::memory_order_acquire);
  if (monitor != nullptr) {
    monitor->record_completion(logical_jobs, queue_latency_ms, e2e_latency_ms);
  }
}

void
record_rejection(std::size_t count)
{
  auto monitor = monitor_atomic().load(std::memory_order_acquire);
  if (monitor != nullptr) {
    monitor->record_rejection(count);
  }
}

auto
snapshot() -> std::optional<Snapshot>
{
  auto monitor = monitor_atomic().load(std::memory_order_acquire);
  if (monitor == nullptr) {
    return std::nullopt;
  }
  return monitor->snapshot();
}

bool
is_congested()
{
  auto monitor = monitor_atomic().load(std::memory_order_acquire);
  return monitor != nullptr && monitor->congested();
}

}  // namespace starpu_server::congestion
