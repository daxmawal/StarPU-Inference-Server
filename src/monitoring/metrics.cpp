#include "monitoring/metrics.hpp"
#if defined(STARPU_TESTING)
#include "monitoring/metrics_test_api.hpp"
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

namespace monitoring::detail {

auto
read_total_cpu_times(std::istream& input, CpuTotals& out) -> bool
{
  std::string cpu{};
  if (!(input >> cpu)) {
    return false;
  }
  if (cpu != "cpu") {
    return false;
  }
  if (!(input >> out.user >> out.nice >> out.system >> out.idle >> out.iowait >>
        out.irq >> out.softirq >> out.steal)) {
    return false;
  }
  return true;
}

auto
read_total_cpu_times(const std::filesystem::path& path, CpuTotals& out) -> bool
{
  std::ifstream input{path};
  if (!input.is_open()) {
    return false;
  }
  return read_total_cpu_times(input, out);
}

}  // namespace monitoring::detail

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

auto make_default_cpu_usage_provider() -> MetricsRegistry::CpuUsageProvider;
auto make_default_cpu_usage_provider(
    std::function<bool(monitoring::detail::CpuTotals&)> reader)
    -> MetricsRegistry::CpuUsageProvider;

namespace {
using monitoring::detail::CpuTotals;

const std::filesystem::path kProcStatm{"/proc/self/statm"};

const prometheus::Histogram::BucketBoundaries kInferenceLatencyMsBuckets{
    1, 5, 10, 25, 50, 100, 250, 500, 1000};
const prometheus::Histogram::BucketBoundaries kBatchSizeBuckets{
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
const prometheus::Histogram::BucketBoundaries kBatchEfficiencyBuckets{
    0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0, 8.0};
const prometheus::Histogram::BucketBoundaries kModelLoadDurationMsBuckets{
    10, 50, 100, 200, 500, 1000, 2000, 5000, 10000};
const prometheus::Histogram::BucketBoundaries kTaskRuntimeMsBuckets{
    1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 5000};

constexpr std::size_t kMaxLabelSeries = 10000;
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
constexpr auto kSamplingErrorLogThrottle = std::chrono::seconds(0);
#else
constexpr auto kSamplingErrorLogThrottle = std::chrono::seconds(60);
#endif  // SONAR_IGNORE_END

#include "metrics_gpu_cpu_providers.inl"
#include "metrics_labels.inl"
#include "metrics_registration.inl"
#include "metrics_sampler.inl"

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
  register_request_counters_and_families(registry, counters_, families_);
  register_queue_runtime_and_batching_metrics(registry, gauges_, histograms_);
  register_congestion_gauges(registry, gauges_.congestion);
  register_latency_histograms(registry, histograms_);
  register_model_gpu_and_worker_metrics(registry, histograms_, families_);

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

namespace {
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
      entry->second = create_metric(overflow);
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
}  // namespace

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
  with_metrics_registry([size](MetricsRegistry& metrics) {
    set_gauge_if_present(
        metrics.gauges().inflight_tasks, static_cast<double>(size));
  });
}

void
set_starpu_worker_busy_ratio(double ratio)
{
  with_metrics_registry([ratio](MetricsRegistry& metrics) {
    set_gauge_if_present(
        metrics.gauges().starpu_worker_busy_ratio, std::clamp(ratio, 0.0, 1.0));
  });
}

void
set_max_inflight_tasks(std::size_t max_tasks)
{
  with_metrics_registry([max_tasks](MetricsRegistry& metrics) {
    set_gauge_if_present(
        metrics.gauges().max_inflight_tasks, static_cast<double>(max_tasks));
  });
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
  with_metrics_registry([depth](MetricsRegistry& metrics) {
    set_gauge_if_present(
        metrics.gauges().starpu_prepared_queue_depth,
        static_cast<double>(depth));
  });
}

void
set_batch_pending_jobs(std::size_t pending)
{
  with_metrics_registry([pending](MetricsRegistry& metrics) {
    set_gauge_if_present(
        metrics.gauges().batch_pending_jobs, static_cast<double>(pending));
  });
}

void
increment_rejected_requests()
{
  with_metrics_registry([](MetricsRegistry& metrics) {
    increment_counter_if_present(metrics.counters().requests_rejected_total);
  });
}

void
set_congestion_flag(bool congested)
{
  with_metrics_registry([congested](MetricsRegistry& metrics) {
    set_gauge_if_present(
        metrics.gauges().congestion.flag, congested ? 1.0 : 0.0);
  });
}

void
set_congestion_score(double score)
{
  with_metrics_registry([score](MetricsRegistry& metrics) {
    set_gauge_if_present(
        metrics.gauges().congestion.score, std::clamp(score, 0.0, 1.0));
  });
}

void
set_congestion_arrival_rate(double rps)
{
  with_metrics_registry([rps](MetricsRegistry& metrics) {
    set_gauge_if_present(
        metrics.gauges().congestion.lambda_rps, std::max(0.0, rps));
  });
}

void
set_congestion_completion_rate(double rps)
{
  with_metrics_registry([rps](MetricsRegistry& metrics) {
    set_gauge_if_present(
        metrics.gauges().congestion.mu_rps, std::max(0.0, rps));
  });
}

void
set_congestion_rejection_rate(double rps)
{
  with_metrics_registry([rps](MetricsRegistry& metrics) {
    set_gauge_if_present(
        metrics.gauges().congestion.rejection_rps, std::max(0.0, rps));
  });
}

void
set_congestion_rho(double rho)
{
  with_metrics_registry([rho](MetricsRegistry& metrics) {
    const double value = std::isfinite(rho) ? rho : 0.0;
    set_gauge_if_present(
        metrics.gauges().congestion.rho_ewma, std::max(0.0, value));
  });
}

void
set_congestion_fill_ewma(double fill)
{
  with_metrics_registry([fill](MetricsRegistry& metrics) {
    set_gauge_if_present(
        metrics.gauges().congestion.queue_fill_ewma,
        std::clamp(fill, 0.0, 1.0));
  });
}

void
set_congestion_queue_growth_rate(double rate)
{
  with_metrics_registry([rate](MetricsRegistry& metrics) {
    set_gauge_if_present(metrics.gauges().congestion.queue_growth_rate, rate);
  });
}

void
set_congestion_queue_latency_p95(double latency_ms)
{
  with_metrics_registry([latency_ms](MetricsRegistry& metrics) {
    set_gauge_if_present(
        metrics.gauges().congestion.queue_p95_ms, std::max(0.0, latency_ms));
  });
}

void
set_congestion_queue_latency_p99(double latency_ms)
{
  with_metrics_registry([latency_ms](MetricsRegistry& metrics) {
    set_gauge_if_present(
        metrics.gauges().congestion.queue_p99_ms, std::max(0.0, latency_ms));
  });
}

void
set_congestion_e2e_latency_p95(double latency_ms)
{
  with_metrics_registry([latency_ms](MetricsRegistry& metrics) {
    set_gauge_if_present(
        metrics.gauges().congestion.e2e_p95_ms, std::max(0.0, latency_ms));
  });
}

void
set_congestion_e2e_latency_p99(double latency_ms)
{
  with_metrics_registry([latency_ms](MetricsRegistry& metrics) {
    set_gauge_if_present(
        metrics.gauges().congestion.e2e_p99_ms, std::max(0.0, latency_ms));
  });
}

void
set_server_health(bool ready)
{
  with_metrics_registry([ready](MetricsRegistry& metrics) {
    set_gauge_if_present(
        metrics.gauges().server_health_state, ready ? 1.0 : 0.0);
  });
}

void
increment_request_status(int status_code, std::string_view model_name)
{
  with_metrics_registry([status_code, model_name](MetricsRegistry& metrics) {
    metrics.increment_status_counter(
        MetricsRegistry::StatusCodeLabel{status_code_label(status_code)},
        MetricsRegistry::ModelLabel{model_name});
  });
}

void
increment_requests_received(std::string_view model_name)
{
  with_metrics_registry([model_name](MetricsRegistry& metrics) {
    metrics.increment_received_counter(model_name);
  });
}

void
observe_batch_size(std::size_t batch_size)
{
  with_metrics_registry([batch_size](MetricsRegistry& metrics) {
    observe_histogram_if_non_negative(
        metrics.histograms().batch_size, static_cast<double>(batch_size));
  });
}

void
observe_logical_batch_size(std::size_t logical_jobs)
{
  with_metrics_registry([logical_jobs](MetricsRegistry& metrics) {
    observe_histogram_if_non_negative(
        metrics.histograms().logical_batch_size,
        static_cast<double>(logical_jobs));
  });
}

void
observe_batch_efficiency(double ratio)
{
  with_metrics_registry([ratio](MetricsRegistry& metrics) {
    observe_histogram_if_non_negative(
        metrics.histograms().batch_efficiency, ratio);
  });
}

void
observe_latency_breakdown(const LatencyBreakdownMetrics& breakdown)
{
  with_metrics_registry([&breakdown](MetricsRegistry& metrics) {
    auto& histograms = metrics.histograms();
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
  });
}

void
observe_starpu_task_runtime(double runtime_ms)
{
  with_metrics_registry([runtime_ms](MetricsRegistry& metrics) {
    observe_histogram_if_non_negative(
        metrics.histograms().starpu_task_runtime, runtime_ms);
  });
}

void
observe_model_load_duration(double duration_ms)
{
  with_metrics_registry([duration_ms](MetricsRegistry& metrics) {
    observe_histogram_if_non_negative(
        metrics.histograms().model_load_duration, duration_ms);
  });
}

void
set_model_loaded(
    std::string_view model_name, std::string_view device_label, bool loaded)
{
  with_metrics_registry(
      [model_name, device_label, loaded](MetricsRegistry& metrics) {
        metrics.set_model_loaded_flag(
            MetricsRegistry::ModelLabel{model_name},
            MetricsRegistry::DeviceLabel{device_label}, loaded);
      });
}

void
increment_model_load_failure(std::string_view model_name)
{
  with_metrics_registry([model_name](MetricsRegistry& metrics) {
    metrics.increment_model_load_failure_counter(model_name);
  });
}

void
observe_compute_latency_by_worker(
    int worker_id, int device_id, std::string_view worker_type,
    double latency_ms)
{
  with_metrics_registry([worker_id, device_id, worker_type,
                         latency_ms](MetricsRegistry& metrics) {
    metrics.observe_compute_latency_by_worker(
        worker_id, device_id, worker_type, latency_ms);
  });
}

void
observe_task_runtime_by_worker(
    int worker_id, int device_id, std::string_view worker_type,
    double latency_ms)
{
  with_metrics_registry([worker_id, device_id, worker_type,
                         latency_ms](MetricsRegistry& metrics) {
    metrics.observe_task_runtime_by_worker(
        worker_id, device_id, worker_type, latency_ms);
  });
}

void
set_worker_inflight_gauge(
    int worker_id, int device_id, std::string_view worker_type,
    std::size_t value)
{
  with_metrics_registry([worker_id, device_id, worker_type,
                         value](MetricsRegistry& metrics) {
    metrics.set_worker_inflight_gauge(worker_id, device_id, worker_type, value);
  });
}

void
observe_io_copy_latency(
    std::string_view direction, int worker_id, int device_id,
    std::string_view worker_type, double duration_ms)
{
  with_metrics_registry([direction, worker_id, device_id, worker_type,
                         duration_ms](MetricsRegistry& metrics) {
    metrics.observe_io_copy_latency(
        direction, worker_id, device_id, worker_type, duration_ms);
  });
}

void
increment_transfer_bytes(
    std::string_view direction, int worker_id, int device_id,
    std::string_view worker_type, std::size_t bytes)
{
  with_metrics_registry([direction, worker_id, device_id, worker_type,
                         bytes](MetricsRegistry& metrics) {
    metrics.increment_transfer_bytes(
        direction, worker_id, device_id, worker_type, bytes);
  });
}

void
MetricsRegistry::request_stop()
{
  if (registry_state_.sampler_thread.joinable()) {
    registry_state_.sampler_thread.request_stop();
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
    if (monitoring::detail::metrics_request_stop_skip_join_for_test()) {
      return;
    }
#endif  // SONAR_IGNORE_END
    registry_state_.sampler_thread.join();
  }
}

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
auto
MetricsRegistry::has_gpu_stats_provider() const -> bool
{
  return static_cast<bool>(providers_.gpu_stats_provider);
}

auto
MetricsRegistry::has_cpu_usage_provider() const -> bool
{
  return static_cast<bool>(providers_.cpu_usage_provider);
}
#endif  // SONAR_IGNORE_END

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
      [this, code_label, model_label](bool overflow) -> prometheus::Counter* {
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
      [this, model_label](bool overflow) -> prometheus::Counter* {
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
      [this, model_label](bool overflow) -> prometheus::Counter* {
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
      [this, stage_label, reason_label,
       model_label](bool overflow) -> prometheus::Counter* {
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
      [this, model_label](bool overflow) -> prometheus::Counter* {
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
      [this, model_label, device_label](bool overflow) -> prometheus::Gauge* {
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
      [this, worker_id, device_id,
       worker_type](bool overflow) -> prometheus::Histogram* {
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
      [this, worker_id, device_id,
       worker_type](bool overflow) -> prometheus::Histogram* {
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
      [this, worker_id, device_id,
       worker_type](bool overflow) -> prometheus::Gauge* {
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
      [this, direction, worker_id, device_id,
       worker_type](bool overflow) -> prometheus::Histogram* {
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
      [this, direction, worker_id, device_id,
       worker_type](bool overflow) -> prometheus::Counter* {
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
  with_metrics_registry([model_name, logical_jobs](MetricsRegistry& metrics) {
    metrics.increment_completed_counter(model_name, logical_jobs);
  });
}

void
increment_inference_failure(
    std::string_view stage, std::string_view reason,
    std::string_view model_name, std::size_t count)
{
  with_metrics_registry(
      [stage, reason, model_name, count](MetricsRegistry& metrics) {
        metrics.increment_failure_counter(
            MetricsRegistry::FailureStageLabel{stage},
            MetricsRegistry::FailureReasonLabel{reason},
            MetricsRegistry::ModelLabel{model_name}, count);
      });
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
void
starpu_server::testing::MetricsRegistryTestAccessor::ClearCpuUsageProvider(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.providers_.cpu_usage_provider = {};
}

void
starpu_server::testing::MetricsRegistryTestAccessor::ClearSystemCpuUsageGauge(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.gauges_.system_cpu_usage_percent = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::ClearProcessOpenFdsGauge(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.gauges_.process_open_fds = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    ClearProcessResidentMemoryGauge(starpu_server::MetricsRegistry& metrics)
{
  metrics.gauges_.process_resident_memory_bytes = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    ClearInferenceThroughputGauge(starpu_server::MetricsRegistry& metrics)
{
  metrics.gauges_.inference_throughput = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::ClearGpuStatsProvider(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.providers_.gpu_stats_provider = {};
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::ProcessOpenFdsGauge(
    starpu_server::MetricsRegistry& metrics) -> prometheus::Gauge*
{
  return metrics.gauges_.process_open_fds;
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::ProcessResidentMemoryGauge(
    starpu_server::MetricsRegistry& metrics) -> prometheus::Gauge*
{
  return metrics.gauges_.process_resident_memory_bytes;
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::InferenceThroughputGauge(
    starpu_server::MetricsRegistry& metrics) -> prometheus::Gauge*
{
  return metrics.gauges_.inference_throughput;
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::GpuUtilizationGaugeCount(
    const starpu_server::MetricsRegistry& metrics) -> std::size_t
{
  std::scoped_lock<std::mutex> lock(metrics.mutexes_.sampling);
  return metrics.caches_.gpu.utilization.size();
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::GpuMemoryUsedGaugeCount(
    const starpu_server::MetricsRegistry& metrics) -> std::size_t
{
  std::scoped_lock<std::mutex> lock(metrics.mutexes_.sampling);
  return metrics.caches_.gpu.memory_used.size();
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::GpuMemoryTotalGaugeCount(
    const starpu_server::MetricsRegistry& metrics) -> std::size_t
{
  std::scoped_lock<std::mutex> lock(metrics.mutexes_.sampling);
  return metrics.caches_.gpu.memory_total.size();
}

void
starpu_server::testing::MetricsRegistryTestAccessor::SampleProcessOpenFds(
    starpu_server::MetricsRegistry& metrics)
{
  if (metrics.sampler_ != nullptr) {
    metrics.sampler_->sample_process_open_fds();
  }
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    SampleProcessResidentMemory(starpu_server::MetricsRegistry& metrics)
{
  if (metrics.sampler_ != nullptr) {
    metrics.sampler_->sample_process_resident_memory();
  }
}

void
starpu_server::testing::MetricsRegistryTestAccessor::SampleInferenceThroughput(
    starpu_server::MetricsRegistry& metrics)
{
  if (metrics.sampler_ != nullptr) {
    metrics.sampler_->sample_inference_throughput();
  }
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    ClearStarpuWorkerInflightFamily(starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.starpu_worker_inflight = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    ClearStarpuTaskRuntimeByWorkerFamily(
        starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.starpu_task_runtime_by_worker = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    ClearInferenceComputeLatencyByWorkerFamily(
        starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.inference_compute_latency_by_worker = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::ClearIoCopyLatencyFamily(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.io_copy_latency = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::ClearTransferBytesFamily(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.transfer_bytes = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::ClearModelsLoadedFamily(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.models_loaded = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    ClearModelLoadFailuresFamily(starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.model_load_failures = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    ClearInferenceFailuresFamily(starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.inference_failures = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    ClearInferenceCompletedFamily(starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.inference_completed = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    ClearRequestsReceivedFamily(starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.requests_received = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    ClearRequestsByStatusFamily(starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.requests_by_status = nullptr;
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::FailureKeyOverflowIsEmpty()
    -> bool
{
  const auto key = MetricsRegistry::FailureKey::Overflow();
  return key.overflow && key.stage.empty() && key.reason.empty() &&
         key.model.empty();
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::FailureKeyEquals(
    std::string_view stage_lhs, std::string_view reason_lhs,
    std::string_view model_lhs, bool overflow_lhs, std::string_view stage_rhs,
    std::string_view reason_rhs, std::string_view model_rhs,
    bool overflow_rhs) -> bool
{
  MetricsRegistry::FailureKey lhs{
      std::string(stage_lhs), std::string(reason_lhs), std::string(model_lhs),
      overflow_lhs};
  MetricsRegistry::FailureKey rhs{
      std::string(stage_rhs), std::string(reason_rhs), std::string(model_rhs),
      overflow_rhs};
  return lhs == rhs;
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::ModelKeyOverflowIsEmpty()
    -> bool
{
  const auto key = MetricsRegistry::ModelKey::Overflow();
  return key.overflow && key.model.empty();
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::ModelKeyEquals(
    std::string_view model_lhs, bool overflow_lhs, std::string_view model_rhs,
    bool overflow_rhs) -> bool
{
  MetricsRegistry::ModelKey lhs{std::string(model_lhs), overflow_lhs};
  MetricsRegistry::ModelKey rhs{std::string(model_rhs), overflow_rhs};
  return lhs == rhs;
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::
    ModelDeviceKeyOverflowIsEmpty() -> bool
{
  const auto key = MetricsRegistry::ModelDeviceKey::Overflow();
  return key.overflow && key.model.empty() && key.device.empty();
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::ModelDeviceKeyEquals(
    std::string_view model_lhs, std::string_view device_lhs, bool overflow_lhs,
    std::string_view model_rhs, std::string_view device_rhs,
    bool overflow_rhs) -> bool
{
  MetricsRegistry::ModelDeviceKey lhs{
      std::string(model_lhs), std::string(device_lhs), overflow_lhs};
  MetricsRegistry::ModelDeviceKey rhs{
      std::string(model_rhs), std::string(device_rhs), overflow_rhs};
  return lhs == rhs;
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::IoKeyOverflowIsEmpty()
    -> bool
{
  const auto key = MetricsRegistry::IoKey::Overflow();
  return key.overflow && key.direction.empty() && key.worker_id == 0 &&
         key.device_id == 0 && key.worker_type.empty();
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::IoKeyEquals(
    std::string_view direction_lhs, int worker_id_lhs, int device_id_lhs,
    std::string_view worker_type_lhs, bool overflow_lhs,
    std::string_view direction_rhs, int worker_id_rhs, int device_id_rhs,
    std::string_view worker_type_rhs, bool overflow_rhs) -> bool
{
  MetricsRegistry::IoKey lhs{
      std::string(direction_lhs), worker_id_lhs, device_id_lhs,
      std::string(worker_type_lhs), overflow_lhs};
  MetricsRegistry::IoKey rhs{
      std::string(direction_rhs), worker_id_rhs, device_id_rhs,
      std::string(worker_type_rhs), overflow_rhs};
  return lhs == rhs;
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::WorkerKeyOverflowIsEmpty()
    -> bool
{
  const auto key = MetricsRegistry::WorkerKey::Overflow();
  return key.overflow && key.worker_id == 0 && key.device_id == 0 &&
         key.worker_type.empty();
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::WorkerKeyEquals(
    int worker_id_lhs, int device_id_lhs, std::string_view worker_type_lhs,
    bool overflow_lhs, int worker_id_rhs, int device_id_rhs,
    std::string_view worker_type_rhs, bool overflow_rhs) -> bool
{
  MetricsRegistry::WorkerKey lhs{
      worker_id_lhs, device_id_lhs, std::string(worker_type_lhs), overflow_lhs};
  MetricsRegistry::WorkerKey rhs{
      worker_id_rhs, device_id_rhs, std::string(worker_type_rhs), overflow_rhs};
  return lhs == rhs;
}
#endif  // SONAR_IGNORE_END
