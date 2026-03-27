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
#include "utils/batching_trace_logger.hpp"

namespace starpu_server::congestion {
inline namespace congestion_monitor_detail {

constexpr double kEpsilon = 1e-9;
constexpr double kMaxRho = 1'000.0;

auto
percentile(std::vector<double>& samples, double pct) -> std::optional<double>
{
  if (samples.empty()) {
    return std::nullopt;
  }
  if (pct <= 0.0) {
    return *std::ranges::min_element(samples);
  }
  if (pct >= 100.0) {
    return *std::ranges::max_element(samples);
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

auto
safe_divide(double numerator, double denominator) -> double
{
  return denominator > kEpsilon ? numerator / denominator : 0.0;
}

auto
compute_rho_sample(double lambda_rps, double mu_rps) -> double
{
  if (mu_rps > 0.0) {
    return lambda_rps / mu_rps;
  }
  if (lambda_rps > 0.0) {
    return kMaxRho;
  }
  return 0.0;
}

template <typename TickStats>
auto
build_congestion_snapshot(
    const TickStats& stats, double score, bool congested_state) -> Snapshot
{
  Snapshot snap{};
  snap.congestion = congested_state;
  snap.lambda_rps = stats.lambda_rps;
  snap.mu_rps = stats.mu_rps;
  snap.rho_ewma = stats.rho_smoothed;
  snap.fill_ewma = stats.fill_smoothed;
  snap.dqueue_ewma = stats.dqueue_smoothed;
  snap.queue_p95_ms = stats.queue_p95_smoothed.value_or(0.0);
  snap.queue_p99_ms = stats.queue_p99_smoothed.value_or(0.0);
  snap.e2e_p95_ms = stats.e2e_p95_smoothed.value_or(0.0);
  snap.e2e_p99_ms = stats.e2e_p99_smoothed.value_or(0.0);
  snap.rejection_rps = stats.rejection_rps;
  snap.score = score;
  snap.queue_size = stats.queue_size;
  snap.queue_capacity = stats.queue_capacity;
  snap.last_tick = stats.now;
  return snap;
}

class MonitorImpl final : public Monitor::Impl {
 public:
  MonitorImpl(const MonitorImpl&) = delete;
  auto operator=(const MonitorImpl&) -> MonitorImpl& = delete;
  MonitorImpl(MonitorImpl&&) = delete;
  auto operator=(MonitorImpl&&) -> MonitorImpl& = delete;
  ~MonitorImpl() override = default;

  MonitorImpl(
      InferenceQueue* queue, const Config& cfg,
      std::shared_ptr<MetricsRecorder> metrics,
      std::shared_ptr<BatchingTraceLogger> tracer)
      : queue_(queue), cfg_(cfg), metrics_(std::move(metrics)),
        tracer_(std::move(tracer))
  {
    runtime_state_.last_queue_size = queue_ != nullptr ? queue_->size() : 0;
    normalize_config();
    snapshot_state_.value.queue_size = runtime_state_.last_queue_size;
    snapshot_state_.value.queue_capacity =
        queue_ != nullptr ? queue_->capacity() : 0;
    snapshot_state_.value.last_tick = runtime_state_.last_tick;
  }

  void start() override
  {
    worker_ =
        std::jthread([this](const std::stop_token& stop) { tick_loop(stop); });
  }

  void shutdown() override
  {
    if (worker_.joinable()) {
      worker_.request_stop();
      worker_.join();
    }
    flush_congestion_span(std::chrono::steady_clock::now());
  }

  void record_arrival(std::size_t count) override
  {
    if (count == 0) {
      return;
    }
    counters_.arrivals.fetch_add(count, std::memory_order_release);
  }

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  [[nodiscard]] auto arrivals_for_test() const -> std::size_t override
  {
    return counters_.arrivals.load(std::memory_order_acquire);
  }

  [[nodiscard]] auto rejections_for_test() const -> std::size_t override
  {
    return counters_.rejections.load(std::memory_order_acquire);
  }
#endif  // SONAR_IGNORE_END

  void record_completion(
      std::size_t logical_jobs, CompletionLatencies latencies) override
  {
    if (logical_jobs > 0) {
      counters_.completions.fetch_add(logical_jobs, std::memory_order_release);
    }
    if (latencies.queue_latency_ms > 0.0 || latencies.e2e_latency_ms > 0.0) {
      std::scoped_lock lock(samples_.mutex);
      if (latencies.queue_latency_ms > 0.0) {
        samples_.queue_latency_ms.push_back(latencies.queue_latency_ms);
      }
      if (latencies.e2e_latency_ms > 0.0) {
        samples_.e2e_latency_ms.push_back(latencies.e2e_latency_ms);
      }
    }
  }

  void record_rejection(std::size_t count) override
  {
    if (count == 0) {
      return;
    }
    counters_.rejections.fetch_add(count, std::memory_order_release);
  }

  [[nodiscard]] auto snapshot() const -> Snapshot override
  {
    std::scoped_lock lock(snapshot_state_.mutex);
    return snapshot_state_.value;
  }

  [[nodiscard]] auto congested() const -> bool override
  {
    return congested_flag_.load(std::memory_order_acquire);
  }

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  auto evaluate_latency_flags_for_test(
      std::optional<double> queue_p95,
      std::optional<double> e2e_p95 = std::nullopt)
      -> LatencyFlagResult override
  {
    TickStats stats{};
    stats.queue_p95_smoothed = queue_p95;
    stats.e2e_p95_smoothed = e2e_p95;
    const auto flags = evaluate_latency_flags(stats);
    return LatencyFlagResult{flags.danger, flags.ok};
  }

  [[nodiscard]] auto compute_queue_pressure_score_for_test(
      double fill_smoothed) const -> double override
  {
    return compute_queue_pressure_score(fill_smoothed);
  }

  [[nodiscard]] auto compute_latency_pressure_score_for_test(
      std::optional<double> e2e_p95,
      std::optional<double> queue_p95 = std::nullopt) const -> double override
  {
    TickStats stats{};
    stats.e2e_p95_smoothed = e2e_p95;
    stats.queue_p95_smoothed = queue_p95;
    return compute_latency_pressure_score(stats);
  }

  [[nodiscard]] auto normalized_config_for_test() const
      -> NormalizedConfigResult override
  {
    return NormalizedConfigResult{
        .tick_interval = cfg_.tick_interval,
        .entry_horizon = cfg_.entry_horizon,
        .exit_horizon = cfg_.exit_horizon,
        .queue_budget_ms = latency_thresholds_.queue_budget_ms,
    };
  }

  [[nodiscard]] auto compute_capacity_pressure_score_for_test(
      double rho_smoothed) const -> double override
  {
    return compute_capacity_pressure_score(rho_smoothed);
  }
#endif  // SONAR_IGNORE_END

 private:
  struct TickStats {
    std::chrono::steady_clock::time_point now;
    std::chrono::milliseconds elapsed;
    double dt_seconds;
    double lambda_rps;
    double mu_rps;
    double rejection_rps;
    std::size_t queue_size;
    std::size_t queue_capacity;
    double fill_smoothed;
    double dqueue_smoothed;
    double rho_smoothed;
    std::optional<double> queue_p95_smoothed;
    std::optional<double> queue_p99_smoothed;
    std::optional<double> e2e_p95_smoothed;
    std::optional<double> e2e_p99_smoothed;
    bool panic;
  };

  struct LatencyFlags {
    bool danger;
    bool ok;
  };

  struct Counters {
    std::atomic<std::uint64_t> arrivals{0};
    std::atomic<std::uint64_t> completions{0};
    std::atomic<std::uint64_t> rejections{0};
  };

  struct Samples {
    mutable std::mutex mutex;
    std::vector<double> queue_latency_ms;
    std::vector<double> e2e_latency_ms;
  };

  struct SnapshotState {
    mutable std::mutex mutex;
    Snapshot value{};
  };

  struct EwmaState {
    std::optional<double> fill;
    std::optional<double> dqueue;
    std::optional<double> rho;
    std::optional<double> queue_p95;
    std::optional<double> queue_p99;
    std::optional<double> e2e_p95;
    std::optional<double> e2e_p99;
  };

  struct RuntimeState {
    std::optional<std::chrono::steady_clock::time_point> congestion_start;
    std::size_t last_queue_size{0};
    std::chrono::steady_clock::time_point last_tick{
        std::chrono::steady_clock::now()};
    std::chrono::milliseconds entry_accumulator{0};
    std::chrono::milliseconds exit_accumulator{0};
  };

  struct LatencyThresholds {
    std::optional<double> queue_budget_ms;
    double warn_latency_ms{0.0};
    double ok_latency_ms{0.0};
  };

  auto capture_tick_stats() -> TickStats
  {
    TickStats stats{};
    stats.now = std::chrono::steady_clock::now();
    stats.elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        stats.now - runtime_state_.last_tick);
    stats.dt_seconds = std::max(
        std::chrono::duration<double>(stats.elapsed).count(),
        std::chrono::duration<double>(cfg_.tick_interval).count());
    runtime_state_.last_tick = stats.now;

    const auto arrivals =
        counters_.arrivals.exchange(0, std::memory_order_acq_rel);
    const auto completions =
        counters_.completions.exchange(0, std::memory_order_acq_rel);
    const auto rejections =
        counters_.rejections.exchange(0, std::memory_order_acq_rel);

    stats.lambda_rps =
        safe_divide(static_cast<double>(arrivals), stats.dt_seconds);
    stats.mu_rps =
        safe_divide(static_cast<double>(completions), stats.dt_seconds);
    stats.rejection_rps =
        safe_divide(static_cast<double>(rejections), stats.dt_seconds);
    stats.panic = rejections > 0;

    stats.queue_capacity = queue_ != nullptr ? queue_->capacity() : 0U;
    stats.queue_size = queue_ != nullptr ? queue_->size() : 0U;
    const double fill_ratio =
        stats.queue_capacity > 0
            ? std::clamp(
                  static_cast<double>(stats.queue_size) /
                      static_cast<double>(stats.queue_capacity),
                  0.0, 1.0)
            : 0.0;
    stats.fill_smoothed = update_ewma(smoothed_.fill, fill_ratio, cfg_.alpha);

    const double dqueue_dt = safe_divide(
        static_cast<double>(stats.queue_size) -
            static_cast<double>(runtime_state_.last_queue_size),
        stats.dt_seconds);
    runtime_state_.last_queue_size = stats.queue_size;
    stats.dqueue_smoothed =
        update_ewma(smoothed_.dqueue, dqueue_dt, cfg_.alpha);

    const double rho_sample =
        compute_rho_sample(stats.lambda_rps, stats.mu_rps);
    stats.rho_smoothed =
        update_ewma(smoothed_.rho, std::min(rho_sample, kMaxRho), cfg_.alpha);

    std::vector<double> queue_samples;
    std::vector<double> e2e_samples;
    {
      std::scoped_lock lock(samples_.mutex);
      queue_samples.swap(samples_.queue_latency_ms);
      e2e_samples.swap(samples_.e2e_latency_ms);
    }

    const auto queue_p95 = percentile(queue_samples, 95.0);
    const auto queue_p99 = percentile(queue_samples, 99.0);
    const auto e2e_p95 = percentile(e2e_samples, 95.0);
    const auto e2e_p99 = percentile(e2e_samples, 99.0);

    stats.queue_p95_smoothed =
        update_optional_ewma(smoothed_.queue_p95, queue_p95, cfg_.alpha);
    stats.queue_p99_smoothed =
        update_optional_ewma(smoothed_.queue_p99, queue_p99, cfg_.alpha);
    stats.e2e_p95_smoothed =
        update_optional_ewma(smoothed_.e2e_p95, e2e_p95, cfg_.alpha);
    stats.e2e_p99_smoothed =
        update_optional_ewma(smoothed_.e2e_p99, e2e_p99, cfg_.alpha);

    return stats;
  }

  auto evaluate_latency_flags(const TickStats& stats) const -> LatencyFlags
  {
    LatencyFlags flags{.danger = false, .ok = true};
    if (latency_thresholds_.warn_latency_ms > 0.0 &&
        stats.e2e_p95_smoothed.has_value()) {
      flags.danger =
          *stats.e2e_p95_smoothed > latency_thresholds_.warn_latency_ms;
    } else if (
        latency_thresholds_.queue_budget_ms.has_value() &&
        stats.queue_p95_smoothed.has_value()) {
      flags.danger =
          *stats.queue_p95_smoothed > *latency_thresholds_.queue_budget_ms;
    }

    if (latency_thresholds_.ok_latency_ms > 0.0 &&
        stats.e2e_p95_smoothed.has_value()) {
      flags.ok = *stats.e2e_p95_smoothed < latency_thresholds_.ok_latency_ms;
    } else if (
        latency_thresholds_.queue_budget_ms.has_value() &&
        stats.queue_p95_smoothed.has_value()) {
      flags.ok =
          *stats.queue_p95_smoothed < *latency_thresholds_.queue_budget_ms;
    }
    return flags;
  }

  auto compute_entry_condition(
      const TickStats& stats, const LatencyFlags& flags) const -> bool
  {
    const bool under_provisioned = stats.rho_smoothed > cfg_.rho_high;
    const bool queue_pressure =
        stats.fill_smoothed > cfg_.fill_high && stats.dqueue_smoothed > 0.0;
    return under_provisioned || queue_pressure || flags.danger;
  }

  auto compute_exit_condition(
      const TickStats& stats, const LatencyFlags& flags) const -> bool
  {
    return stats.fill_smoothed < cfg_.fill_low &&
           stats.rho_smoothed < cfg_.rho_low && flags.ok;
  }

  void update_congestion_state(
      const TickStats& stats, bool entry_condition, bool exit_condition)
  {
    const auto elapsed_ms = stats.elapsed;
    if (stats.panic) {
      set_congested(true, stats.now);
      runtime_state_.entry_accumulator = cfg_.entry_horizon;
      runtime_state_.exit_accumulator = std::chrono::milliseconds::zero();
      return;
    }

    if (!congested()) {
      runtime_state_.entry_accumulator =
          entry_condition ? runtime_state_.entry_accumulator + elapsed_ms
                          : std::chrono::milliseconds::zero();
      if (runtime_state_.entry_accumulator >= cfg_.entry_horizon) {
        set_congested(true, stats.now);
        runtime_state_.exit_accumulator = std::chrono::milliseconds::zero();
      }
      return;
    }

    runtime_state_.exit_accumulator =
        exit_condition ? runtime_state_.exit_accumulator + elapsed_ms
                       : std::chrono::milliseconds::zero();
    if (runtime_state_.exit_accumulator >= cfg_.exit_horizon) {
      set_congested(false, stats.now);
      runtime_state_.entry_accumulator = std::chrono::milliseconds::zero();
    }
  }

  auto compute_queue_pressure_score(double fill_smoothed) const -> double
  {
    if (cfg_.fill_high <= cfg_.fill_low) {
      return 0.0;
    }
    return std::clamp(
        (fill_smoothed - cfg_.fill_low) /
            std::max(cfg_.fill_high - cfg_.fill_low, kEpsilon),
        0.0, 1.0);
  }

  auto compute_latency_pressure_score(const TickStats& stats) const -> double
  {
    if (latency_thresholds_.warn_latency_ms > 0.0 &&
        stats.e2e_p95_smoothed.has_value()) {
      const double upper = cfg_.latency_slo_ms * 1.1;
      const double lower = latency_thresholds_.ok_latency_ms;
      if (upper > lower) {
        return std::clamp(
            (*stats.e2e_p95_smoothed - lower) / (upper - lower), 0.0, 1.0);
      }
    } else if (
        latency_thresholds_.queue_budget_ms.has_value() &&
        stats.queue_p95_smoothed.has_value()) {
      const double upper = *latency_thresholds_.queue_budget_ms * 1.2;
      const double lower = *latency_thresholds_.queue_budget_ms;
      if (upper > lower) {
        return std::clamp(
            (*stats.queue_p95_smoothed - lower) / (upper - lower), 0.0, 1.0);
      }
    }
    return 0.0;
  }

  auto compute_capacity_pressure_score(double rho_smoothed) const -> double
  {
    if (cfg_.rho_high <= cfg_.rho_low) {
      return 0.0;
    }
    return std::clamp(
        (rho_smoothed - cfg_.rho_low) /
            std::max(cfg_.rho_high - cfg_.rho_low, kEpsilon),
        0.0, 1.0);
  }

  auto compute_pressure_score(const TickStats& stats) const -> double
  {
    const double queue_pressure_score =
        compute_queue_pressure_score(stats.fill_smoothed);
    const double latency_pressure_score = compute_latency_pressure_score(stats);
    const double capacity_pressure_score =
        compute_capacity_pressure_score(stats.rho_smoothed);
    return std::max(
        {queue_pressure_score, latency_pressure_score,
         capacity_pressure_score});
  }

  void publish_metrics(
      const TickStats& stats, double score, bool congested_state) const
  {
    if (metrics_ == nullptr) {
      return;
    }
    metrics_->set_congestion_flag(congested_state);
    metrics_->set_congestion_score(score);
    metrics_->set_congestion_arrival_rate(stats.lambda_rps);
    metrics_->set_congestion_completion_rate(stats.mu_rps);
    metrics_->set_congestion_rejection_rate(stats.rejection_rps);
    metrics_->set_congestion_rho(stats.rho_smoothed);
    metrics_->set_congestion_fill_ewma(stats.fill_smoothed);
    metrics_->set_congestion_queue_growth_rate(stats.dqueue_smoothed);
    metrics_->set_congestion_queue_latency_p95(
        stats.queue_p95_smoothed.value_or(0.0));
    metrics_->set_congestion_queue_latency_p99(
        stats.queue_p99_smoothed.value_or(0.0));
    metrics_->set_congestion_e2e_latency_p95(
        stats.e2e_p95_smoothed.value_or(0.0));
    metrics_->set_congestion_e2e_latency_p99(
        stats.e2e_p99_smoothed.value_or(0.0));
  }

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
      latency_thresholds_.warn_latency_ms =
          cfg_.latency_slo_ms * cfg_.e2e_warn_ratio;
      latency_thresholds_.ok_latency_ms =
          cfg_.latency_slo_ms * cfg_.e2e_ok_ratio;
    }
    if (cfg_.queue_latency_budget_ms > 0.0) {
      latency_thresholds_.queue_budget_ms = cfg_.queue_latency_budget_ms;
    } else if (
        cfg_.latency_slo_ms > 0.0 && cfg_.queue_latency_budget_ratio > 0.0) {
      latency_thresholds_.queue_budget_ms =
          cfg_.latency_slo_ms * cfg_.queue_latency_budget_ratio;
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
    const auto stats = capture_tick_stats();
    const auto latency_flags = evaluate_latency_flags(stats);
    const bool entry_condition = compute_entry_condition(stats, latency_flags);
    const bool exit_condition = compute_exit_condition(stats, latency_flags);

    update_congestion_state(stats, entry_condition, exit_condition);
    const bool congested_state = congested();
    const double score = compute_pressure_score(stats);

    const Snapshot snap =
        build_congestion_snapshot(stats, score, congested_state);
    {
      std::scoped_lock lock(snapshot_state_.mutex);
      snapshot_state_.value = snap;
    }

    publish_metrics(stats, score, congested_state);
  }

  void set_congested(bool state, std::chrono::steady_clock::time_point now)
  {
    const bool previous =
        congested_flag_.exchange(state, std::memory_order_acq_rel);
    if (state && !previous) {
      runtime_state_.congestion_start = now;
    } else if (!state && previous) {
      flush_congestion_span(now);
    }
  }

  void flush_congestion_span(std::chrono::steady_clock::time_point end_time)
  {
    if (!runtime_state_.congestion_start.has_value() || tracer_ == nullptr) {
      return;
    }
    tracer_->log_congestion_span(BatchingTraceLogger::TimeRange{
        .start = *runtime_state_.congestion_start,
        .end = end_time,
    });
    runtime_state_.congestion_start.reset();
  }

  InferenceQueue* queue_;
  Config cfg_;
  std::shared_ptr<MetricsRecorder> metrics_;
  std::shared_ptr<BatchingTraceLogger> tracer_;
  std::jthread worker_;
  std::atomic<bool> congested_flag_{false};
  Counters counters_{};
  Samples samples_{};
  SnapshotState snapshot_state_{};
  EwmaState smoothed_{};
  RuntimeState runtime_state_{};
  LatencyThresholds latency_thresholds_{};
};

auto
monitor_atomic() -> std::atomic<std::shared_ptr<Monitor>>&
{
  static std::atomic<std::shared_ptr<Monitor>> instance{nullptr};
  return instance;
}

auto
make_global_metrics_recorder() -> std::shared_ptr<MetricsRecorder>
{
  auto metrics = get_metrics();
  if (metrics == nullptr) {
    return nullptr;
  }
  return std::make_shared<MetricsRecorder>(std::move(metrics));
}

auto
make_global_tracer() -> std::shared_ptr<BatchingTraceLogger>
{
  return {
      &BatchingTraceLogger::instance(),
      [](BatchingTraceLogger* /*tracer*/) noexcept {
        // This shared_ptr aliases the process-wide singleton tracer.
        // Ownership stays with BatchingTraceLogger::instance().
      },
  };
}

}  // namespace congestion_monitor_detail

Monitor::Monitor(
    InferenceQueue* queue, Config config,
    std::shared_ptr<MetricsRecorder> metrics,
    std::shared_ptr<BatchingTraceLogger> tracer)
    : impl_(std::make_shared<MonitorImpl>(
          queue, config, std::move(metrics), std::move(tracer)))
{
}

void
Monitor::start()
{
  if (impl_ != nullptr) {
    impl_->start();
  }
}

void
Monitor::shutdown()
{
  if (impl_ != nullptr) {
    impl_->shutdown();
  }
}

void
Monitor::record_arrival(std::size_t count)
{
  if (impl_ != nullptr) {
    impl_->record_arrival(count);
  }
}

void
Monitor::record_completion(
    std::size_t logical_jobs, CompletionLatencies latencies)
{
  if (impl_ != nullptr) {
    impl_->record_completion(logical_jobs, latencies);
  }
}

void
Monitor::record_rejection(std::size_t count)
{
  if (impl_ != nullptr) {
    impl_->record_rejection(count);
  }
}

auto
Monitor::snapshot() const -> Snapshot
{
  return impl_ != nullptr ? impl_->snapshot() : Snapshot{};
}

auto
Monitor::congested() const -> bool
{
  return impl_ != nullptr && impl_->congested();
}

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
auto
Monitor::arrivals_for_test() const -> std::size_t
{
  return impl_ != nullptr ? impl_->arrivals_for_test() : 0;
}

auto
Monitor::rejections_for_test() const -> std::size_t
{
  return impl_ != nullptr ? impl_->rejections_for_test() : 0;
}

auto
Monitor::evaluate_latency_flags_for_test(
    std::optional<double> queue_p95,
    std::optional<double> e2e_p95) -> LatencyFlagResult
{
  return impl_ != nullptr
             ? impl_->evaluate_latency_flags_for_test(queue_p95, e2e_p95)
             : LatencyFlagResult{false, true};
}

auto
Monitor::compute_queue_pressure_score_for_test(double fill_smoothed) const
    -> double
{
  return impl_ != nullptr
             ? impl_->compute_queue_pressure_score_for_test(fill_smoothed)
             : 0.0;
}

auto
Monitor::compute_latency_pressure_score_for_test(
    std::optional<double> e2e_p95,
    std::optional<double> queue_p95) const -> double
{
  return impl_ != nullptr ? impl_->compute_latency_pressure_score_for_test(
                                e2e_p95, queue_p95)
                          : 0.0;
}

auto
Monitor::normalized_config_for_test() const -> NormalizedConfigResult
{
  return impl_ != nullptr ? impl_->normalized_config_for_test()
                          : NormalizedConfigResult{};
}

auto
Monitor::compute_capacity_pressure_score_for_test(double rho_smoothed) const
    -> double
{
  return impl_ != nullptr
             ? impl_->compute_capacity_pressure_score_for_test(rho_smoothed)
             : 0.0;
}
#endif  // SONAR_IGNORE_END

auto
start(InferenceQueue* queue, Config config) -> bool
{
  if (queue == nullptr || !config.enabled) {
    monitor_atomic().store(nullptr, std::memory_order_release);
    if (queue == nullptr && config.enabled) {
      log_warning("Congestion monitor disabled: missing inference queue");
    }
    return false;
  }
  auto monitor = std::make_shared<Monitor>(
      queue, config, make_global_metrics_recorder(), make_global_tracer());
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
record_completion(std::size_t logical_jobs, CompletionLatencies latencies)
{
  auto monitor = monitor_atomic().load(std::memory_order_acquire);
  if (monitor != nullptr) {
    monitor->record_completion(logical_jobs, latencies);
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

auto
is_congested() -> bool
{
  auto monitor = monitor_atomic().load(std::memory_order_acquire);
  return monitor != nullptr && monitor->congested();
}

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
auto
percentile_for_test(std::vector<double> samples, double pct)
    -> std::optional<double>
{
  return percentile(samples, pct);
}

auto
update_ewma_for_test(std::optional<double>& state, double sample, double alpha)
    -> double
{
  return update_ewma(state, sample, alpha);
}

auto
record_arrival_for_test(std::size_t count) -> std::size_t
{
  InferenceQueue queue(1);
  Config cfg;
  Monitor monitor(&queue, cfg);
  monitor.record_arrival(count);
  return monitor.arrivals_for_test();
}

auto
record_rejection_for_test(std::size_t count) -> std::size_t
{
  InferenceQueue queue(1);
  Config cfg;
  Monitor monitor(&queue, cfg);
  monitor.record_rejection(count);
  return monitor.rejections_for_test();
}

auto
evaluate_latency_flags_for_test(
    double queue_budget_ms,
    std::optional<double> queue_p95) -> LatencyFlagResult
{
  InferenceQueue queue(1);
  Config cfg;
  cfg.latency_slo_ms = 0.0;
  cfg.queue_latency_budget_ms = queue_budget_ms;
  Monitor monitor(&queue, cfg);
  return monitor.evaluate_latency_flags_for_test(queue_p95);
}

auto
compute_queue_pressure_score_for_test(const QueuePressureScoreTestArgs& args)
    -> double
{
  InferenceQueue queue(1);
  Config cfg;
  cfg.fill_high = args.fill_high;
  cfg.fill_low = args.fill_low;
  Monitor monitor(&queue, cfg);
  return monitor.compute_queue_pressure_score_for_test(args.fill_smoothed);
}

auto
compute_latency_pressure_score_for_test(
    const LatencyPressureScoreTestArgs& args) -> double
{
  InferenceQueue queue(1);
  Config cfg;
  cfg.latency_slo_ms = args.latency_slo_ms;
  cfg.e2e_ok_ratio = args.e2e_ok_ratio;
  Monitor monitor(&queue, cfg);
  return monitor.compute_latency_pressure_score_for_test(args.e2e_p95);
}

auto
compute_latency_pressure_score_queue_budget_for_test(
    double queue_budget_ms, std::optional<double> queue_p95) -> double
{
  InferenceQueue queue(1);
  Config cfg;
  cfg.latency_slo_ms = 0.0;
  cfg.queue_latency_budget_ms = queue_budget_ms;
  Monitor monitor(&queue, cfg);
  return monitor.compute_latency_pressure_score_for_test(
      std::nullopt, queue_p95);
}

auto
normalize_config_for_test(Config cfg) -> NormalizedConfigResult
{
  InferenceQueue queue(1);
  Monitor monitor(&queue, cfg);
  return monitor.normalized_config_for_test();
}

auto
compute_capacity_pressure_score_for_test(
    const CapacityPressureScoreTestArgs& args) -> double
{
  InferenceQueue queue(1);
  Config cfg;
  cfg.rho_high = args.rho_high;
  cfg.rho_low = args.rho_low;
  Monitor monitor(&queue, cfg);
  return monitor.compute_capacity_pressure_score_for_test(args.rho_smoothed);
}
#endif  // SONAR_IGNORE_END

}  // namespace starpu_server::congestion
