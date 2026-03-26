#include "monitoring/metrics.hpp"
#if defined(STARPU_TESTING)
#include "support/monitoring/metrics_test_api.hpp"
#endif

#include <prometheus/exposer.h>
#include <prometheus/histogram.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef STARPU_HAVE_NVML
#include <nvml.h>
#endif

#include "metrics_constants.hpp"
#include "metrics_gpu_cpu_providers.hpp"
#include "metrics_labels.hpp"
#include "metrics_registration.hpp"
#include "metrics_sampler.hpp"
#include "utils/logger.hpp"
#include "utils/perf_observer.hpp"

namespace starpu_server {

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
class TestingExposerHandle : public MetricsRegistry::ExposerHandle {
 public:
  void RegisterCollectable(
      const std::shared_ptr<prometheus::Collectable>& collectable) override
  {
  }

  void RemoveCollectable(
      const std::shared_ptr<prometheus::Collectable>& collectable) override
  {
  }
};
#endif  // SONAR_IGNORE_END

class PrometheusExposerHandle : public MetricsRegistry::ExposerHandle {
 public:
  explicit PrometheusExposerHandle(std::unique_ptr<prometheus::Exposer> exposer)
      : exposer_(std::move(exposer))
  {
  }

  void RegisterCollectable(
      const std::shared_ptr<prometheus::Collectable>& collectable) override
  {
    exposer_->RegisterCollectable(collectable);
  }

  void RemoveCollectable(
      const std::shared_ptr<prometheus::Collectable>& collectable) override
  {
    exposer_->RemoveCollectable(collectable);
  }

 private:
  std::unique_ptr<prometheus::Exposer> exposer_;
};

MetricsRegistry::MetricsRegistry(int port)
    : MetricsRegistry(
          port, query_gpu_stats_nvml, make_default_cpu_usage_provider())
{
}

MetricsRegistry::MetricsRegistry(
    int port, GpuStatsProvider gpu_provider, CpuUsageProvider cpu_provider,
    bool start_sampler_thread, std::unique_ptr<ExposerHandle> exposer_handle)
    : registry_state_{std::make_shared<prometheus::Registry>()},
      providers_{std::move(gpu_provider), std::move(cpu_provider)}
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (monitoring::detail::metrics_init_failure_for_test()) {
    throw std::runtime_error("forced metrics initialization failure");
  }
#endif  // SONAR_IGNORE_END
  if (!providers_.gpu_stats_provider) {
    providers_.gpu_stats_provider = query_gpu_stats_nvml;
  }
  if (!providers_.cpu_usage_provider) {
    providers_.cpu_usage_provider = make_default_cpu_usage_provider();
  }
  sampler_ = std::make_unique<Sampler>(*this);
  initialize(port, start_sampler_thread, std::move(exposer_handle));
}

void
MetricsRegistry::initialize(
    [[maybe_unused]] int port, bool start_sampler_thread,
    std::unique_ptr<ExposerHandle> exposer_handle)
{
  try {
    if (!exposer_handle) {
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
      exposer_handle = std::make_unique<TestingExposerHandle>();
#else
      auto exposer = std::make_unique<prometheus::Exposer>(
          std::format("0.0.0.0:{}", port));
      exposer_handle =
          std::make_unique<PrometheusExposerHandle>(std::move(exposer));
#endif  // SONAR_IGNORE_END
    }
    exposer_handle->RegisterCollectable(registry_state_.registry);
    registry_state_.exposer = std::move(exposer_handle);
  }
  catch (const std::exception& e) {
    log_error(std::string("Failed to initialize metrics exposer: ") + e.what());
    throw;
  }

  auto& registry = *registry_state_.registry;
  detail::register_request_counters_and_families(
      registry, counters_, families_);
  detail::register_queue_runtime_and_batching_metrics(
      registry, gauges_, histograms_);
  detail::register_congestion_gauges(registry, gauges_.congestion);
  detail::register_latency_histograms(registry, histograms_);
  detail::register_model_gpu_and_worker_metrics(
      registry, histograms_, families_);

  if (start_sampler_thread) {
    registry_state_.sampler_thread = std::jthread(
        [this](const std::stop_token& stop) { sampler_->sampling_loop(stop); });
  }
}

MetricsRegistry::~MetricsRegistry() noexcept
{
  request_stop();
  if (registry_state_.sampler_thread.joinable()) {
    registry_state_.sampler_thread.join();
  }
  if (registry_state_.exposer && registry_state_.registry) {
    try {
      registry_state_.exposer->RemoveCollectable(registry_state_.registry);
    }
    catch (const std::exception& e) {
      log_error(
          std::string("Failed to remove metrics registry collectable: ") +
          e.what());
    }
  }
}

inline namespace metrics_file_detail {
auto
metrics_atomic() -> std::atomic<std::shared_ptr<MetricsRegistry>>&
{
  static std::atomic<std::shared_ptr<MetricsRegistry>> instance{nullptr};
  return instance;
}

auto
metrics_init_mutex() -> std::mutex&
{
  static std::mutex mutex;
  return mutex;
}

auto
metrics_shutdown_once_flag() -> std::once_flag&
{
  static std::once_flag flag;
  return flag;
}

template <typename Fn>
void
with_metrics_registry(Fn&& callback)
{
  auto metrics_ptr = metrics_atomic().load(std::memory_order_acquire);
  if (metrics_ptr != nullptr) {
    std::forward<Fn>(callback)(*metrics_ptr);
  }
}

template <typename Fn>
void
with_metrics_registry(
    const std::shared_ptr<MetricsRegistry>& metrics_ptr, Fn&& callback)
{
  if (metrics_ptr != nullptr) {
    std::forward<Fn>(callback)(*metrics_ptr);
  }
}

template <auto Method, typename... Args>
void
invoke_metrics_registry(Args&&... args)
{
  with_metrics_registry([&args...](MetricsRegistry& metrics) {
    std::invoke(Method, metrics, std::forward<Args>(args)...);
  });
}

template <auto Method, typename... Args>
void
invoke_metrics_registry(
    const std::shared_ptr<MetricsRegistry>& metrics_ptr, Args&&... args)
{
  with_metrics_registry(metrics_ptr, [&args...](MetricsRegistry& metrics) {
    std::invoke(Method, metrics, std::forward<Args>(args)...);
  });
}

void
set_gauge_if_present(prometheus::Gauge* gauge, double value)
{
  if (gauge != nullptr) {
    gauge->Set(value);
  }
}

void
increment_counter_if_present(prometheus::Counter* counter, double value = 1.0)
{
  if (counter != nullptr) {
    counter->Increment(value);
  }
}

void
observe_histogram_if_non_negative(
    prometheus::Histogram* histogram, double value)
{
  if (histogram != nullptr && value >= 0.0) {
    histogram->Observe(value);
  }
}

using GaugeMetrics = MetricsRegistry::GaugeMetrics;
using CongestionGaugeMetrics = GaugeMetrics::CongestionGaugeMetrics;
using HistogramMetrics = MetricsRegistry::HistogramMetrics;
using CounterMetrics = MetricsRegistry::CounterMetrics;

void record_latency_breakdown(
    MetricsRegistry::HistogramMetrics& histograms,
    const LatencyBreakdownMetrics& breakdown);

template <typename Value, typename Transform = std::identity>
void
set_gauge(
    prometheus::Gauge* GaugeMetrics::*member, Value value,
    Transform transform = {})
{
  with_metrics_registry([member, value, transform = std::move(transform)](
                            MetricsRegistry& metrics) mutable {
    const auto metric_value =
        static_cast<double>(std::invoke(transform, value));
    set_gauge_if_present(metrics.gauges().*member, metric_value);
  });
}

template <typename Value, typename Transform = std::identity>
void
set_gauge(
    const std::shared_ptr<MetricsRegistry>& metrics_ptr,
    prometheus::Gauge* GaugeMetrics::*member, Value value,
    Transform transform = {})
{
  with_metrics_registry(
      metrics_ptr, [member, value, transform = std::move(transform)](
                       MetricsRegistry& metrics) mutable {
        const auto metric_value =
            static_cast<double>(std::invoke(transform, value));
        set_gauge_if_present(metrics.gauges().*member, metric_value);
      });
}

template <typename Value, typename Transform = std::identity>
void
set_gauge(
    CongestionGaugeMetrics GaugeMetrics::*congestion_member,
    prometheus::Gauge* CongestionGaugeMetrics::*member, Value value,
    Transform transform = {})
{
  with_metrics_registry(
      [congestion_member, member, value,
       transform = std::move(transform)](MetricsRegistry& metrics) mutable {
        const auto metric_value =
            static_cast<double>(std::invoke(transform, value));
        const auto& congestion_metrics = metrics.gauges().*congestion_member;
        set_gauge_if_present(congestion_metrics.*member, metric_value);
      });
}

template <typename Value, typename Transform = std::identity>
void
set_gauge(
    const std::shared_ptr<MetricsRegistry>& metrics_ptr,
    CongestionGaugeMetrics GaugeMetrics::*congestion_member,
    prometheus::Gauge* CongestionGaugeMetrics::*member, Value value,
    Transform transform = {})
{
  with_metrics_registry(
      metrics_ptr,
      [congestion_member, member, value,
       transform = std::move(transform)](MetricsRegistry& metrics) mutable {
        const auto metric_value =
            static_cast<double>(std::invoke(transform, value));
        const auto& congestion_metrics = metrics.gauges().*congestion_member;
        set_gauge_if_present(congestion_metrics.*member, metric_value);
      });
}

template <typename Value, typename Transform = std::identity>
void
observe_histogram(
    prometheus::Histogram* HistogramMetrics::*member, Value value,
    Transform transform = {})
{
  with_metrics_registry([member, value, transform = std::move(transform)](
                            MetricsRegistry& metrics) mutable {
    const auto metric_value =
        static_cast<double>(std::invoke(transform, value));
    observe_histogram_if_non_negative(
        metrics.histograms().*member, metric_value);
  });
}

template <typename Value, typename Transform = std::identity>
void
observe_histogram(
    const std::shared_ptr<MetricsRegistry>& metrics_ptr,
    prometheus::Histogram* HistogramMetrics::*member, Value value,
    Transform transform = {})
{
  with_metrics_registry(
      metrics_ptr, [member, value, transform = std::move(transform)](
                       MetricsRegistry& metrics) mutable {
        const auto metric_value =
            static_cast<double>(std::invoke(transform, value));
        observe_histogram_if_non_negative(
            metrics.histograms().*member, metric_value);
      });
}

void
increment_counter(prometheus::Counter* CounterMetrics::*member)
{
  with_metrics_registry([member](MetricsRegistry& metrics) {
    increment_counter_if_present(metrics.counters().*member);
  });
}

void
increment_counter(
    const std::shared_ptr<MetricsRegistry>& metrics_ptr,
    prometheus::Counter* CounterMetrics::*member)
{
  with_metrics_registry(metrics_ptr, [member](MetricsRegistry& metrics) {
    increment_counter_if_present(metrics.counters().*member);
  });
}

template <typename Value, typename Transform = std::identity>
void
increment_counter(
    prometheus::Counter* CounterMetrics::*member, Value value,
    Transform transform = {})
{
  with_metrics_registry([member, value, transform = std::move(transform)](
                            MetricsRegistry& metrics) mutable {
    const auto metric_value =
        static_cast<double>(std::invoke(transform, value));
    increment_counter_if_present(metrics.counters().*member, metric_value);
  });
}

template <typename Value, typename Transform = std::identity>
void
increment_counter(
    const std::shared_ptr<MetricsRegistry>& metrics_ptr,
    prometheus::Counter* CounterMetrics::*member, Value value,
    Transform transform = {})
{
  with_metrics_registry(
      metrics_ptr, [member, value, transform = std::move(transform)](
                       MetricsRegistry& metrics) mutable {
        const auto metric_value =
            static_cast<double>(std::invoke(transform, value));
        increment_counter_if_present(metrics.counters().*member, metric_value);
      });
}

template <typename Key, typename Map, typename Factory>
auto
get_or_create_cached_series(Map& cache, Key lookup_key, Factory&& create_metric)
    -> typename Map::mapped_type
{
  auto entry = cache.find(lookup_key);
  if (entry == cache.end()) {
    const bool overflow = cache.size() >= kMaxLabelSeries;
    Key map_key = overflow ? Key::Overflow() : std::move(lookup_key);
    auto [inserted_it, inserted] =
        cache.try_emplace(std::move(map_key), nullptr);
    entry = inserted_it;
    if (inserted) {
      entry->second = std::forward<Factory>(create_metric)(overflow);
    }
  }
  return entry->second;
}

struct WorkerMetricLabels {
  std::string worker_id;
  std::string device;
  std::string worker_type;
};

auto
make_worker_metric_labels(
    bool overflow, int worker_id, int device_id,
    std::string_view worker_type) -> WorkerMetricLabels
{
  if (overflow) {
    return {kOverflowLabel, kOverflowLabel, kOverflowLabel};
  }
  return {
      std::to_string(worker_id), std::to_string(device_id),
      escape_label_value(worker_type)};
}

struct IoMetricLabels {
  std::string direction;
  WorkerMetricLabels worker;
};

auto
make_io_metric_labels(
    bool overflow, std::string_view direction, int worker_id, int device_id,
    std::string_view worker_type) -> IoMetricLabels
{
  if (overflow) {
    return {kOverflowLabel, {kOverflowLabel, kOverflowLabel, kOverflowLabel}};
  }
  return {
      escape_label_value(direction),
      make_worker_metric_labels(false, worker_id, device_id, worker_type)};
}
}  // namespace metrics_file_detail

auto
init_metrics(int port) -> bool
{
  std::scoped_lock<std::mutex> lock(metrics_init_mutex());
  if (metrics_atomic().load(std::memory_order_acquire) != nullptr) {
    log_warning("Metrics were previously initialized");
    return false;
  }

  try {
    auto new_metrics = std::make_shared<MetricsRegistry>(port);
    std::call_once(
        metrics_shutdown_once_flag(), [] { std::atexit(shutdown_metrics); });

#ifndef STARPU_HAVE_NVML
    std::call_once(nvml_warning_flag(), [] {
      log_warning_critical(
          "NVML support is not available; GPU metrics collection is "
          "disabled.");
    });
#endif

    metrics_atomic().store(new_metrics, std::memory_order_release);

    set_queue_size(0);
    set_inflight_tasks(0);
    set_max_inflight_tasks(0);
    return true;
  }
  catch (const std::exception& e) {
    log_error(std::string("Metrics initialization failed: ") + e.what());
    return false;
  }
}

void
shutdown_metrics()
{
  std::scoped_lock<std::mutex> lock(metrics_init_mutex());
  auto metrics_ptr = metrics_atomic().load(std::memory_order_acquire);
  if (metrics_ptr) {
    metrics_ptr->request_stop();
  }
  metrics_atomic().store(nullptr, std::memory_order_release);
}

auto
get_metrics() -> std::shared_ptr<MetricsRegistry>
{
  return metrics_atomic().load(std::memory_order_acquire);
}

auto
create_metrics_recorder(int port) -> std::shared_ptr<MetricsRecorder>
{
  try {
    auto recorder = std::make_shared<MetricsRecorder>(
        std::make_shared<MetricsRegistry>(port));
    recorder->set_queue_size(0);
    recorder->set_inflight_tasks(0);
    recorder->set_max_inflight_tasks(0);
    return recorder;
  }
  catch (const std::exception& e) {
    log_error(std::string("Metrics initialization failed: ") + e.what());
    return nullptr;
  }
}

void
MetricsRecorder::increment_requests_total() const
{
  increment_counter(registry_, &CounterMetrics::requests_total);
}

void
MetricsRecorder::observe_inference_latency(double latency_ms) const
{
  observe_histogram(
      registry_, &HistogramMetrics::inference_latency, latency_ms);
}

void
MetricsRecorder::set_queue_size(std::size_t size) const
{
  with_metrics_registry(registry_, [size](MetricsRegistry& metrics) {
    auto& gauges = metrics.gauges();
    set_gauge_if_present(gauges.queue_size, static_cast<double>(size));
    const auto capacity = metrics.queue_capacity_value();
    if (capacity > 0 && gauges.queue_fill_ratio != nullptr) {
      const double ratio =
          static_cast<double>(size) / static_cast<double>(capacity);
      gauges.queue_fill_ratio->Set(std::clamp(ratio, 0.0, 1.0));
    }
  });
}

void
MetricsRecorder::set_inflight_tasks(std::size_t size) const
{
  set_gauge(registry_, &GaugeMetrics::inflight_tasks, size);
}

void
MetricsRecorder::set_starpu_worker_busy_ratio(double ratio) const
{
  set_gauge(
      registry_, &GaugeMetrics::starpu_worker_busy_ratio, ratio,
      [](double value) { return std::clamp(value, 0.0, 1.0); });
}

void
MetricsRecorder::set_max_inflight_tasks(std::size_t max_tasks) const
{
  set_gauge(registry_, &GaugeMetrics::max_inflight_tasks, max_tasks);
}

void
MetricsRecorder::set_queue_capacity(std::size_t capacity) const
{
  with_metrics_registry(registry_, [capacity](MetricsRegistry& metrics) {
    auto& gauges = metrics.gauges();
    metrics.set_queue_capacity(capacity);
    set_gauge_if_present(gauges.queue_capacity, static_cast<double>(capacity));
    if (gauges.queue_fill_ratio == nullptr) {
      return;
    }
    if (capacity > 0 && gauges.queue_size != nullptr) {
      const double size = gauges.queue_size->Value();
      gauges.queue_fill_ratio->Set(
          std::clamp(size / static_cast<double>(capacity), 0.0, 1.0));
      return;
    }
    gauges.queue_fill_ratio->Set(0.0);
  });
}

void
MetricsRecorder::set_server_health(bool ready) const
{
  set_gauge(
      registry_, &GaugeMetrics::server_health_state, ready,
      [](bool value) { return value ? 1.0 : 0.0; });
}

void
MetricsRecorder::set_starpu_prepared_queue_depth(std::size_t depth) const
{
  set_gauge(registry_, &GaugeMetrics::starpu_prepared_queue_depth, depth);
}

void
MetricsRecorder::set_batch_pending_jobs(std::size_t pending) const
{
  set_gauge(registry_, &GaugeMetrics::batch_pending_jobs, pending);
}

void
MetricsRecorder::set_congestion_flag(bool congested) const
{
  set_gauge(
      registry_, &GaugeMetrics::congestion, &CongestionGaugeMetrics::flag,
      congested, [](bool value) { return value ? 1.0 : 0.0; });
}

void
MetricsRecorder::set_congestion_score(double score) const
{
  set_gauge(
      registry_, &GaugeMetrics::congestion, &CongestionGaugeMetrics::score,
      score, [](double value) { return std::clamp(value, 0.0, 1.0); });
}

void
MetricsRecorder::set_congestion_arrival_rate(double rps) const
{
  set_gauge(
      registry_, &GaugeMetrics::congestion, &CongestionGaugeMetrics::lambda_rps,
      rps, [](double value) { return std::max(0.0, value); });
}

void
MetricsRecorder::set_congestion_completion_rate(double rps) const
{
  set_gauge(
      registry_, &GaugeMetrics::congestion, &CongestionGaugeMetrics::mu_rps,
      rps, [](double value) { return std::max(0.0, value); });
}

void
MetricsRecorder::set_congestion_rejection_rate(double rps) const
{
  set_gauge(
      registry_, &GaugeMetrics::congestion,
      &CongestionGaugeMetrics::rejection_rps, rps,
      [](double value) { return std::max(0.0, value); });
}

void
MetricsRecorder::set_congestion_rho(double rho) const
{
  set_gauge(
      registry_, &GaugeMetrics::congestion, &CongestionGaugeMetrics::rho_ewma,
      rho, [](double value) {
        const double safe_value = std::isfinite(value) ? value : 0.0;
        return std::max(0.0, safe_value);
      });
}

void
MetricsRecorder::set_congestion_fill_ewma(double fill) const
{
  set_gauge(
      registry_, &GaugeMetrics::congestion,
      &CongestionGaugeMetrics::queue_fill_ewma, fill,
      [](double value) { return std::clamp(value, 0.0, 1.0); });
}

void
MetricsRecorder::set_congestion_queue_growth_rate(double rate) const
{
  set_gauge(
      registry_, &GaugeMetrics::congestion,
      &CongestionGaugeMetrics::queue_growth_rate, rate);
}

void
MetricsRecorder::set_congestion_queue_latency_p95(double latency_ms) const
{
  set_gauge(
      registry_, &GaugeMetrics::congestion,
      &CongestionGaugeMetrics::queue_p95_ms, latency_ms,
      [](double value) { return std::max(0.0, value); });
}

void
MetricsRecorder::set_congestion_queue_latency_p99(double latency_ms) const
{
  set_gauge(
      registry_, &GaugeMetrics::congestion,
      &CongestionGaugeMetrics::queue_p99_ms, latency_ms,
      [](double value) { return std::max(0.0, value); });
}

void
MetricsRecorder::set_congestion_e2e_latency_p95(double latency_ms) const
{
  set_gauge(
      registry_, &GaugeMetrics::congestion, &CongestionGaugeMetrics::e2e_p95_ms,
      latency_ms, [](double value) { return std::max(0.0, value); });
}

void
MetricsRecorder::set_congestion_e2e_latency_p99(double latency_ms) const
{
  set_gauge(
      registry_, &GaugeMetrics::congestion, &CongestionGaugeMetrics::e2e_p99_ms,
      latency_ms, [](double value) { return std::max(0.0, value); });
}

void
MetricsRecorder::increment_request_status(
    int status_code, std::string_view model_name) const
{
  invoke_metrics_registry<&MetricsRegistry::increment_status_counter>(
      registry_,
      MetricsRegistry::StatusCodeLabel{status_code_label(status_code)},
      MetricsRegistry::ModelLabel{model_name});
}

void
MetricsRecorder::increment_requests_received(std::string_view model_name) const
{
  invoke_metrics_registry<&MetricsRegistry::increment_received_counter>(
      registry_, model_name);
}

void
MetricsRecorder::increment_inference_completed(
    std::string_view model_name, std::size_t logical_jobs) const
{
  invoke_metrics_registry<&MetricsRegistry::increment_completed_counter>(
      registry_, model_name, logical_jobs);
}

void
MetricsRecorder::increment_inference_failure(
    std::string_view stage, std::string_view reason,
    std::string_view model_name, std::size_t count) const
{
  invoke_metrics_registry<&MetricsRegistry::increment_failure_counter>(
      registry_, MetricsRegistry::FailureStageLabel{stage},
      MetricsRegistry::FailureReasonLabel{reason},
      MetricsRegistry::ModelLabel{model_name}, count);
}

void
MetricsRecorder::observe_batch_size(std::size_t batch_size) const
{
  observe_histogram(registry_, &HistogramMetrics::batch_size, batch_size);
}

void
MetricsRecorder::observe_logical_batch_size(std::size_t logical_jobs) const
{
  observe_histogram(
      registry_, &HistogramMetrics::logical_batch_size, logical_jobs);
}

void
MetricsRecorder::observe_batch_efficiency(double ratio) const
{
  observe_histogram(registry_, &HistogramMetrics::batch_efficiency, ratio);
}

void
MetricsRecorder::observe_latency_breakdown(
    const LatencyBreakdownMetrics& breakdown) const
{
  with_metrics_registry(registry_, [&breakdown](MetricsRegistry& metrics) {
    record_latency_breakdown(metrics.histograms(), breakdown);
  });
}

void
MetricsRecorder::observe_starpu_task_runtime(double runtime_ms) const
{
  observe_histogram(
      registry_, &HistogramMetrics::starpu_task_runtime, runtime_ms);
}

void
MetricsRecorder::observe_model_load_duration(double duration_ms) const
{
  observe_histogram(
      registry_, &HistogramMetrics::model_load_duration, duration_ms);
}

void
MetricsRecorder::set_model_loaded(
    std::string_view model_name, std::string_view device_label,
    bool loaded) const
{
  invoke_metrics_registry<&MetricsRegistry::set_model_loaded_flag>(
      registry_, MetricsRegistry::ModelLabel{model_name},
      MetricsRegistry::DeviceLabel{device_label}, loaded);
}

void
MetricsRecorder::set_gpu_model_replication_policy(
    std::string_view model_name, std::string_view policy_label) const
{
  invoke_metrics_registry<
      &MetricsRegistry::set_gpu_model_replication_policy_flag>(
      registry_, MetricsRegistry::ModelLabel{model_name}, policy_label);
}

void
MetricsRecorder::set_gpu_model_replicas_total(
    std::string_view model_name, std::size_t replicas) const
{
  invoke_metrics_registry<&MetricsRegistry::set_gpu_model_replicas_total_gauge>(
      registry_, MetricsRegistry::ModelLabel{model_name}, replicas);
}

void
MetricsRecorder::set_starpu_cuda_worker_info(
    int worker_id, int device_id, bool active) const
{
  invoke_metrics_registry<&MetricsRegistry::set_starpu_cuda_worker_info_gauge>(
      registry_, worker_id, device_id, active);
}

void
MetricsRecorder::increment_model_load_failure(std::string_view model_name) const
{
  invoke_metrics_registry<
      &MetricsRegistry::increment_model_load_failure_counter>(
      registry_, model_name);
}

void
MetricsRecorder::increment_rejected_requests() const
{
  increment_counter(registry_, &CounterMetrics::requests_rejected_total);
}

void
MetricsRecorder::observe_compute_latency_by_worker(
    int worker_id, int device_id, std::string_view worker_type,
    double latency_ms) const
{
  invoke_metrics_registry<&MetricsRegistry::observe_compute_latency_by_worker>(
      registry_, worker_id, device_id, worker_type, latency_ms);
}

void
MetricsRecorder::observe_task_runtime_by_worker(
    int worker_id, int device_id, std::string_view worker_type,
    double latency_ms) const
{
  invoke_metrics_registry<&MetricsRegistry::observe_task_runtime_by_worker>(
      registry_, worker_id, device_id, worker_type, latency_ms);
}

void
MetricsRecorder::set_worker_inflight_gauge(
    int worker_id, int device_id, std::string_view worker_type,
    std::size_t value) const
{
  invoke_metrics_registry<&MetricsRegistry::set_worker_inflight_gauge>(
      registry_, worker_id, device_id, worker_type, value);
}

void
MetricsRecorder::observe_io_copy_latency(
    std::string_view direction, int worker_id, int device_id,
    std::string_view worker_type, double duration_ms) const
{
  invoke_metrics_registry<&MetricsRegistry::observe_io_copy_latency>(
      registry_, direction, worker_id, device_id, worker_type, duration_ms);
}

void
MetricsRecorder::increment_transfer_bytes(
    std::string_view direction, int worker_id, int device_id,
    std::string_view worker_type, std::size_t bytes) const
{
  invoke_metrics_registry<&MetricsRegistry::increment_transfer_bytes>(
      registry_, direction, worker_id, device_id, worker_type, bytes);
}

void
set_queue_size(std::size_t size)
{
  with_metrics_registry([size](MetricsRegistry& metrics) {
    auto& gauges = metrics.gauges();
    set_gauge_if_present(gauges.queue_size, static_cast<double>(size));
    const auto capacity = metrics.queue_capacity_value();
    if (capacity > 0 && gauges.queue_fill_ratio != nullptr) {
      const double ratio =
          static_cast<double>(size) / static_cast<double>(capacity);
      gauges.queue_fill_ratio->Set(std::clamp(ratio, 0.0, 1.0));
    }
  });
}

void
set_inflight_tasks(std::size_t size)
{
  set_gauge(&GaugeMetrics::inflight_tasks, size);
}

void
set_starpu_worker_busy_ratio(double ratio)
{
  set_gauge(&GaugeMetrics::starpu_worker_busy_ratio, ratio, [](double value) {
    return std::clamp(value, 0.0, 1.0);
  });
}

void
set_max_inflight_tasks(std::size_t max_tasks)
{
  set_gauge(&GaugeMetrics::max_inflight_tasks, max_tasks);
}

void
set_queue_capacity(std::size_t capacity)
{
  with_metrics_registry([capacity](MetricsRegistry& metrics) {
    auto& gauges = metrics.gauges();
    metrics.set_queue_capacity(capacity);
    set_gauge_if_present(gauges.queue_capacity, static_cast<double>(capacity));
    if (gauges.queue_fill_ratio == nullptr) {
      return;
    }
    if (capacity > 0 && gauges.queue_size != nullptr) {
      const double size = gauges.queue_size->Value();
      gauges.queue_fill_ratio->Set(
          std::clamp(size / static_cast<double>(capacity), 0.0, 1.0));
      return;
    }
    gauges.queue_fill_ratio->Set(0.0);
  });
}

// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
void
set_queue_fill_ratio(std::size_t size, std::size_t capacity)
{
  with_metrics_registry([size, capacity](MetricsRegistry& metrics) {
    if (capacity == 0) {
      return;
    }
    if (auto* gauge = metrics.gauges().queue_fill_ratio; gauge != nullptr) {
      gauge->Set(static_cast<double>(size) / static_cast<double>(capacity));
    }
  });
}
#endif  // SONAR_IGNORE_END
// GCOVR_EXCL_STOP

void
set_starpu_prepared_queue_depth(std::size_t depth)
{
  set_gauge(&GaugeMetrics::starpu_prepared_queue_depth, depth);
}

void
set_batch_pending_jobs(std::size_t pending)
{
  set_gauge(&GaugeMetrics::batch_pending_jobs, pending);
}

void
increment_rejected_requests()
{
  increment_counter(&CounterMetrics::requests_rejected_total);
}

void
set_congestion_flag(bool congested)
{
  set_gauge(
      &GaugeMetrics::congestion, &CongestionGaugeMetrics::flag, congested,
      [](bool value) { return value ? 1.0 : 0.0; });
}

void
set_congestion_score(double score)
{
  set_gauge(
      &GaugeMetrics::congestion, &CongestionGaugeMetrics::score, score,
      [](double value) { return std::clamp(value, 0.0, 1.0); });
}

void
set_congestion_arrival_rate(double rps)
{
  set_gauge(
      &GaugeMetrics::congestion, &CongestionGaugeMetrics::lambda_rps, rps,
      [](double value) { return std::max(0.0, value); });
}

void
set_congestion_completion_rate(double rps)
{
  set_gauge(
      &GaugeMetrics::congestion, &CongestionGaugeMetrics::mu_rps, rps,
      [](double value) { return std::max(0.0, value); });
}

void
set_congestion_rejection_rate(double rps)
{
  set_gauge(
      &GaugeMetrics::congestion, &CongestionGaugeMetrics::rejection_rps, rps,
      [](double value) { return std::max(0.0, value); });
}

void
set_congestion_rho(double rho)
{
  set_gauge(
      &GaugeMetrics::congestion, &CongestionGaugeMetrics::rho_ewma, rho,
      [](double value) {
        const double safe_value = std::isfinite(value) ? value : 0.0;
        return std::max(0.0, safe_value);
      });
}

void
set_congestion_fill_ewma(double fill)
{
  set_gauge(
      &GaugeMetrics::congestion, &CongestionGaugeMetrics::queue_fill_ewma, fill,
      [](double value) { return std::clamp(value, 0.0, 1.0); });
}

void
set_congestion_queue_growth_rate(double rate)
{
  set_gauge(
      &GaugeMetrics::congestion, &CongestionGaugeMetrics::queue_growth_rate,
      rate);
}

void
set_congestion_queue_latency_p95(double latency_ms)
{
  set_gauge(
      &GaugeMetrics::congestion, &CongestionGaugeMetrics::queue_p95_ms,
      latency_ms, [](double value) { return std::max(0.0, value); });
}

void
set_congestion_queue_latency_p99(double latency_ms)
{
  set_gauge(
      &GaugeMetrics::congestion, &CongestionGaugeMetrics::queue_p99_ms,
      latency_ms, [](double value) { return std::max(0.0, value); });
}

void
set_congestion_e2e_latency_p95(double latency_ms)
{
  set_gauge(
      &GaugeMetrics::congestion, &CongestionGaugeMetrics::e2e_p95_ms,
      latency_ms, [](double value) { return std::max(0.0, value); });
}

void
set_congestion_e2e_latency_p99(double latency_ms)
{
  set_gauge(
      &GaugeMetrics::congestion, &CongestionGaugeMetrics::e2e_p99_ms,
      latency_ms, [](double value) { return std::max(0.0, value); });
}

void
set_server_health(bool ready)
{
  set_gauge(&GaugeMetrics::server_health_state, ready, [](bool value) {
    return value ? 1.0 : 0.0;
  });
}

void
increment_request_status(int status_code, std::string_view model_name)
{
  invoke_metrics_registry<&MetricsRegistry::increment_status_counter>(
      MetricsRegistry::StatusCodeLabel{status_code_label(status_code)},
      MetricsRegistry::ModelLabel{model_name});
}

void
increment_requests_received(std::string_view model_name)
{
  invoke_metrics_registry<&MetricsRegistry::increment_received_counter>(
      model_name);
}

void
observe_batch_size(std::size_t batch_size)
{
  observe_histogram(&HistogramMetrics::batch_size, batch_size);
}

void
observe_logical_batch_size(std::size_t logical_jobs)
{
  observe_histogram(&HistogramMetrics::logical_batch_size, logical_jobs);
}

void
observe_batch_efficiency(double ratio)
{
  observe_histogram(&HistogramMetrics::batch_efficiency, ratio);
}

void
observe_inference_latency(double latency_ms)
{
  observe_histogram(&HistogramMetrics::inference_latency, latency_ms);
}

inline namespace metrics_file_detail {

void
record_latency_breakdown(
    MetricsRegistry::HistogramMetrics& histograms,
    const LatencyBreakdownMetrics& breakdown)
{
  observe_histogram_if_non_negative(
      histograms.queue_latency, breakdown.queue_ms);
  observe_histogram_if_non_negative(
      histograms.batch_collect_latency, breakdown.batch_ms);
  observe_histogram_if_non_negative(
      histograms.submit_latency, breakdown.submit_ms);
  observe_histogram_if_non_negative(
      histograms.scheduling_latency, breakdown.scheduling_ms);
  observe_histogram_if_non_negative(
      histograms.codelet_latency, breakdown.codelet_ms);
  observe_histogram_if_non_negative(
      histograms.inference_compute_latency, breakdown.inference_ms);
  observe_histogram_if_non_negative(
      histograms.callback_latency, breakdown.callback_ms);
  observe_histogram_if_non_negative(
      histograms.preprocess_latency, breakdown.preprocess_ms);
  observe_histogram_if_non_negative(
      histograms.postprocess_latency, breakdown.postprocess_ms);
}

}  // namespace metrics_file_detail

void
observe_latency_breakdown(const LatencyBreakdownMetrics& breakdown)
{
  with_metrics_registry([&breakdown](MetricsRegistry& metrics) {
    record_latency_breakdown(metrics.histograms(), breakdown);
  });
}

void
observe_starpu_task_runtime(double runtime_ms)
{
  observe_histogram(&HistogramMetrics::starpu_task_runtime, runtime_ms);
}

void
observe_model_load_duration(double duration_ms)
{
  observe_histogram(&HistogramMetrics::model_load_duration, duration_ms);
}

void
set_model_loaded(
    std::string_view model_name, std::string_view device_label, bool loaded)
{
  invoke_metrics_registry<&MetricsRegistry::set_model_loaded_flag>(
      MetricsRegistry::ModelLabel{model_name},
      MetricsRegistry::DeviceLabel{device_label}, loaded);
}

void
set_gpu_model_replication_policy(
    std::string_view model_name, std::string_view policy_label)
{
  invoke_metrics_registry<
      &MetricsRegistry::set_gpu_model_replication_policy_flag>(
      MetricsRegistry::ModelLabel{model_name}, policy_label);
}

void
set_gpu_model_replicas_total(std::string_view model_name, std::size_t replicas)
{
  invoke_metrics_registry<&MetricsRegistry::set_gpu_model_replicas_total_gauge>(
      MetricsRegistry::ModelLabel{model_name}, replicas);
}

void
set_starpu_cuda_worker_info(int worker_id, int device_id, bool active)
{
  invoke_metrics_registry<&MetricsRegistry::set_starpu_cuda_worker_info_gauge>(
      worker_id, device_id, active);
}

void
increment_model_load_failure(std::string_view model_name)
{
  invoke_metrics_registry<
      &MetricsRegistry::increment_model_load_failure_counter>(model_name);
}

void
observe_compute_latency_by_worker(
    int worker_id, int device_id, std::string_view worker_type,
    double latency_ms)
{
  invoke_metrics_registry<&MetricsRegistry::observe_compute_latency_by_worker>(
      worker_id, device_id, worker_type, latency_ms);
}

void
observe_task_runtime_by_worker(
    int worker_id, int device_id, std::string_view worker_type,
    double latency_ms)
{
  invoke_metrics_registry<&MetricsRegistry::observe_task_runtime_by_worker>(
      worker_id, device_id, worker_type, latency_ms);
}

void
set_worker_inflight_gauge(
    int worker_id, int device_id, std::string_view worker_type,
    std::size_t value)
{
  invoke_metrics_registry<&MetricsRegistry::set_worker_inflight_gauge>(
      worker_id, device_id, worker_type, value);
}

void
observe_io_copy_latency(
    std::string_view direction, int worker_id, int device_id,
    std::string_view worker_type, double duration_ms)
{
  invoke_metrics_registry<&MetricsRegistry::observe_io_copy_latency>(
      direction, worker_id, device_id, worker_type, duration_ms);
}

void
increment_transfer_bytes(
    std::string_view direction, int worker_id, int device_id,
    std::string_view worker_type, std::size_t bytes)
{
  invoke_metrics_registry<&MetricsRegistry::increment_transfer_bytes>(
      direction, worker_id, device_id, worker_type, bytes);
}

auto
MetricsRegistry::registry() const -> std::shared_ptr<prometheus::Registry>
{
  return registry_state_.registry;
}

void
MetricsRegistry::increment_status_counter(
    MetricsRegistry::StatusCodeLabel code_label,
    MetricsRegistry::ModelLabel model_label)
{
  if (families_.requests_by_status == nullptr) {
    return;
  }

  StatusKey key{std::string(code_label.value), std::string(model_label.value)};

  std::scoped_lock lock(mutexes_.status);
  auto* counter = get_or_create_cached_series(
      caches_.status.counters, std::move(key),
      [this, code_label, model_label](bool overflow) {
        const std::string code_label_value =
            overflow ? kOverflowLabel : escape_label_value(code_label.value);
        const std::string model_label_value =
            overflow ? kOverflowLabel : escape_label_value(model_label.value);
        return &families_.requests_by_status->Add(
            {{"code", code_label_value}, {"model", model_label_value}});
      });
  increment_counter_if_present(counter);
}

void
MetricsRegistry::increment_completed_counter(
    std::string_view model_label, std::size_t logical_jobs)
{
  if (families_.inference_completed == nullptr) {
    return;
  }
  std::scoped_lock lock(mutexes_.model_metrics);
  auto* counter = get_or_create_cached_series(
      caches_.model.inference_completed, ModelKey{std::string(model_label)},
      [this, model_label](bool overflow) {
        const std::string model_label_value =
            overflow ? kOverflowLabel : escape_label_value(model_label);
        return &families_.inference_completed->Add(
            {{"model", model_label_value}});
      });
  increment_counter_if_present(counter, static_cast<double>(logical_jobs));
}

void
MetricsRegistry::increment_received_counter(std::string_view model_label)
{
  if (families_.requests_received == nullptr) {
    return;
  }
  std::scoped_lock lock(mutexes_.model_metrics);
  auto* counter = get_or_create_cached_series(
      caches_.model.requests_received, ModelKey{std::string(model_label)},
      [this, model_label](bool overflow) {
        const std::string model_label_value =
            overflow ? kOverflowLabel : escape_label_value(model_label);
        return &families_.requests_received->Add(
            {{"model", model_label_value}});
      });
  increment_counter_if_present(counter);
}

void
MetricsRegistry::increment_failure_counter(
    MetricsRegistry::FailureStageLabel stage_label,
    MetricsRegistry::FailureReasonLabel reason_label,
    MetricsRegistry::ModelLabel model_label, std::size_t count)
{
  if (families_.inference_failures == nullptr) {
    return;
  }
  FailureKey key{
      std::string(stage_label.value), std::string(reason_label.value),
      std::string(model_label.value)};

  std::scoped_lock lock(mutexes_.model_metrics);
  auto* counter = get_or_create_cached_series(
      caches_.model.inference_failures, std::move(key),
      [this, stage_label, reason_label, model_label](bool overflow) {
        const std::string stage_label_value =
            overflow ? kOverflowLabel : escape_label_value(stage_label.value);
        const std::string reason_label_value =
            overflow ? kOverflowLabel : escape_label_value(reason_label.value);
        const std::string model_label_value =
            overflow ? kOverflowLabel : escape_label_value(model_label.value);
        return &families_.inference_failures->Add(
            {{"stage", stage_label_value},
             {"reason", reason_label_value},
             {"model", model_label_value}});
      });
  increment_counter_if_present(counter, static_cast<double>(count));
}

void
MetricsRegistry::increment_model_load_failure_counter(
    std::string_view model_label)
{
  if (families_.model_load_failures == nullptr) {
    return;
  }
  std::scoped_lock lock(mutexes_.model_metrics);
  auto* counter = get_or_create_cached_series(
      caches_.model.model_load_failures, ModelKey{std::string(model_label)},
      [this, model_label](bool overflow) {
        const std::string model_label_value =
            overflow ? kOverflowLabel : escape_label_value(model_label);
        return &families_.model_load_failures->Add(
            {{"model", model_label_value}});
      });
  increment_counter_if_present(counter);
}

void
MetricsRegistry::set_model_loaded_flag(
    MetricsRegistry::ModelLabel model_label,
    MetricsRegistry::DeviceLabel device_label, bool loaded)
{
  if (families_.models_loaded == nullptr) {
    return;
  }
  ModelDeviceKey key{
      std::string(model_label.value), std::string(device_label.value)};

  std::scoped_lock lock(mutexes_.model_metrics);
  auto* gauge = get_or_create_cached_series(
      caches_.model.models_loaded, std::move(key),
      [this, model_label, device_label](bool overflow) {
        const std::string model_label_value =
            overflow ? kOverflowLabel : escape_label_value(model_label.value);
        const std::string device_label_value =
            overflow ? kOverflowLabel : escape_label_value(device_label.value);
        return &families_.models_loaded->Add(
            {{"model", model_label_value}, {"device", device_label_value}});
      });
  set_gauge_if_present(gauge, loaded ? 1.0 : 0.0);
}

void
MetricsRegistry::set_gpu_model_replication_policy_flag(
    MetricsRegistry::ModelLabel model_label, std::string_view policy_label)
{
  if (families_.gpu_model_replication_policy_info == nullptr) {
    return;
  }
  ModelPolicyKey key{std::string(model_label.value), std::string(policy_label)};

  std::scoped_lock lock(mutexes_.model_metrics);
  auto* gauge = get_or_create_cached_series(
      caches_.model.gpu_replication_policy, std::move(key),
      [this, model_label, policy_label](bool overflow) {
        const std::string model_label_value =
            overflow ? kOverflowLabel : escape_label_value(model_label.value);
        const std::string policy_label_value =
            overflow ? kOverflowLabel : escape_label_value(policy_label);
        return &families_.gpu_model_replication_policy_info->Add(
            {{"model", model_label_value}, {"policy", policy_label_value}});
      });
  set_gauge_if_present(gauge, 1.0);
}

void
MetricsRegistry::set_gpu_model_replicas_total_gauge(
    MetricsRegistry::ModelLabel model_label, std::size_t replicas)
{
  if (families_.gpu_model_replicas_total == nullptr) {
    return;
  }

  std::scoped_lock lock(mutexes_.model_metrics);
  auto* gauge = get_or_create_cached_series(
      caches_.model.gpu_replicas_total,
      ModelKey{std::string(model_label.value)},
      [this, model_label](bool overflow) {
        const std::string model_label_value =
            overflow ? kOverflowLabel : escape_label_value(model_label.value);
        return &families_.gpu_model_replicas_total->Add(
            {{"model", model_label_value}});
      });
  set_gauge_if_present(gauge, static_cast<double>(replicas));
}

void
MetricsRegistry::set_starpu_cuda_worker_info_gauge(
    int worker_id, int device_id, bool active)
{
  if (families_.starpu_cuda_worker_info == nullptr) {
    return;
  }

  WorkerKey key{worker_id, device_id, "cuda"};
  std::scoped_lock lock(mutexes_.worker_metrics);
  auto* gauge = get_or_create_cached_series(
      caches_.worker.cuda_worker_info, std::move(key),
      [this, worker_id, device_id](bool overflow) {
        const std::string device_label =
            overflow ? kOverflowLabel : std::to_string(device_id);
        const std::string worker_label =
            overflow ? kOverflowLabel : std::to_string(worker_id);
        return &families_.starpu_cuda_worker_info->Add(
            {{"device", device_label}, {"worker_id", worker_label}});
      });
  set_gauge_if_present(gauge, active ? 1.0 : 0.0);
}

void
MetricsRegistry::observe_compute_latency_by_worker(
    int worker_id, int device_id, std::string_view worker_type,
    double latency_ms)
{
  if (latency_ms < 0.0 ||
      families_.inference_compute_latency_by_worker == nullptr) {
    return;
  }
  WorkerKey key{worker_id, device_id, std::string(worker_type)};
  std::scoped_lock lock(mutexes_.worker_metrics);
  auto* histogram = get_or_create_cached_series(
      caches_.worker.compute_latency, std::move(key),
      [this, worker_id, device_id, worker_type](bool overflow) {
        const auto labels = make_worker_metric_labels(
            overflow, worker_id, device_id, worker_type);
        return &families_.inference_compute_latency_by_worker->Add(
            {{"worker_id", labels.worker_id},
             {"device", labels.device},
             {"worker_type", labels.worker_type}},
            kInferenceLatencyMsBuckets);
      });
  observe_histogram_if_non_negative(histogram, latency_ms);
}

void
MetricsRegistry::observe_task_runtime_by_worker(
    int worker_id, int device_id, std::string_view worker_type,
    double latency_ms)
{
  if (latency_ms < 0.0 || families_.starpu_task_runtime_by_worker == nullptr) {
    return;
  }
  WorkerKey key{worker_id, device_id, std::string(worker_type)};
  std::scoped_lock lock(mutexes_.worker_metrics);
  auto* histogram = get_or_create_cached_series(
      caches_.worker.task_runtime, std::move(key),
      [this, worker_id, device_id, worker_type](bool overflow) {
        const auto labels = make_worker_metric_labels(
            overflow, worker_id, device_id, worker_type);
        return &families_.starpu_task_runtime_by_worker->Add(
            {{"worker_id", labels.worker_id},
             {"device", labels.device},
             {"worker_type", labels.worker_type}},
            kTaskRuntimeMsBuckets);
      });
  observe_histogram_if_non_negative(histogram, latency_ms);
}

void
MetricsRegistry::set_worker_inflight_gauge(
    int worker_id, int device_id, std::string_view worker_type,
    std::size_t value)
{
  if (families_.starpu_worker_inflight == nullptr) {
    return;
  }
  WorkerKey key{worker_id, device_id, std::string(worker_type)};
  std::scoped_lock lock(mutexes_.worker_metrics);
  auto* gauge = get_or_create_cached_series(
      caches_.worker.inflight, std::move(key),
      [this, worker_id, device_id, worker_type](bool overflow) {
        const auto labels = make_worker_metric_labels(
            overflow, worker_id, device_id, worker_type);
        return &families_.starpu_worker_inflight->Add(
            {{"worker_id", labels.worker_id},
             {"device", labels.device},
             {"worker_type", labels.worker_type}});
      });
  set_gauge_if_present(gauge, static_cast<double>(value));
}

void
MetricsRegistry::observe_io_copy_latency(
    std::string_view direction, int worker_id, int device_id,
    std::string_view worker_type, double duration_ms)
{
  if (duration_ms < 0.0 || families_.io_copy_latency == nullptr) {
    return;
  }
  IoKey key{
      std::string(direction), worker_id, device_id, std::string(worker_type)};
  std::scoped_lock lock(mutexes_.io_metrics);
  auto* histogram = get_or_create_cached_series(
      caches_.io.copy_latency, std::move(key),
      [this, direction, worker_id, device_id, worker_type](bool overflow) {
        const auto labels = make_io_metric_labels(
            overflow, direction, worker_id, device_id, worker_type);
        return &families_.io_copy_latency->Add(
            {{"direction", labels.direction},
             {"worker_id", labels.worker.worker_id},
             {"device", labels.worker.device},
             {"worker_type", labels.worker.worker_type}},
            kInferenceLatencyMsBuckets);
      });
  observe_histogram_if_non_negative(histogram, duration_ms);
}

void
MetricsRegistry::increment_transfer_bytes(
    std::string_view direction, int worker_id, int device_id,
    std::string_view worker_type, std::size_t bytes)
{
  if (bytes == 0 || families_.transfer_bytes == nullptr) {
    return;
  }
  IoKey key{
      std::string(direction), worker_id, device_id, std::string(worker_type)};
  std::scoped_lock lock(mutexes_.io_metrics);
  auto* counter = get_or_create_cached_series(
      caches_.io.transfer_bytes, std::move(key),
      [this, direction, worker_id, device_id, worker_type](bool overflow) {
        const auto labels = make_io_metric_labels(
            overflow, direction, worker_id, device_id, worker_type);
        return &families_.transfer_bytes->Add(
            {{"direction", labels.direction},
             {"worker_id", labels.worker.worker_id},
             {"device", labels.worker.device},
             {"worker_type", labels.worker.worker_type}});
      });
  increment_counter_if_present(counter, static_cast<double>(bytes));
}

void
increment_inference_completed(
    std::string_view model_name, std::size_t logical_jobs)
{
  invoke_metrics_registry<&MetricsRegistry::increment_completed_counter>(
      model_name, logical_jobs);
}

void
increment_inference_failure(
    std::string_view stage, std::string_view reason,
    std::string_view model_name, std::size_t count)
{
  invoke_metrics_registry<&MetricsRegistry::increment_failure_counter>(
      MetricsRegistry::FailureStageLabel{stage},
      MetricsRegistry::FailureReasonLabel{reason},
      MetricsRegistry::ModelLabel{model_name}, count);
}

void
MetricsRegistry::set_queue_capacity(std::size_t capacity)
{
  registry_state_.queue_capacity.store(capacity, std::memory_order_release);
}

auto
MetricsRegistry::queue_capacity_value() const -> std::size_t
{
  return registry_state_.queue_capacity.load(std::memory_order_acquire);
}

}  // namespace starpu_server

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
#include "support/monitoring/metrics_test_accessor.hpp"
#endif  // SONAR_IGNORE_END
