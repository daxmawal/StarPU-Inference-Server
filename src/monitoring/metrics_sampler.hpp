#pragma once

#include <chrono>
#include <exception>
#include <format>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>

#include "metrics_gpu_cpu_providers.hpp"
#include "monitoring/metrics.hpp"
#include "utils/logger.hpp"
#include "utils/perf_observer.hpp"

namespace starpu_server {

class MetricsRegistry::Sampler {
 public:
  explicit Sampler(MetricsRegistry& registry) : registry_(&registry) {}

  void run_sampling_request_nb();
  void sampling_loop(const std::stop_token& stop);
  void sample_process_open_fds();
  void sample_process_resident_memory();
  void sample_inference_throughput();

 private:
  void sample_cpu_usage();
  void sample_gpu_stats();
  void perform_sampling_request_nb();

  MetricsRegistry* registry_;
};

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
void
MetricsRegistry::run_sampling_request_nb()
{
  if (sampler_ != nullptr) {
    sampler_->run_sampling_request_nb();
  }
}
#endif  // SONAR_IGNORE_END

void
MetricsRegistry::Sampler::run_sampling_request_nb()
{
  std::scoped_lock<std::mutex> lock(registry_->mutexes_.sampling);
  perform_sampling_request_nb();
}

void
MetricsRegistry::Sampler::sample_cpu_usage()
{
  auto& gauges = registry_->gauges_;
  const auto& providers = registry_->providers_;
  if (gauges.system_cpu_usage_percent == nullptr) {
    return;
  }
  if (!providers.cpu_usage_provider) {
    gauges.system_cpu_usage_percent->Set(
        std::numeric_limits<double>::quiet_NaN());
    return;
  }

  try {
    auto usage = providers.cpu_usage_provider();
    if (usage.has_value()) {
      gauges.system_cpu_usage_percent->Set(*usage);
    } else {
      gauges.system_cpu_usage_percent->Set(
          std::numeric_limits<double>::quiet_NaN());
    }
  }
  catch (const std::exception& e) {
    if (should_log_sampling_error(cpu_sampling_error_log_ts())) {
      log_error(std::format("CPU metrics sampling failed: {}", e.what()));
    }
    gauges.system_cpu_usage_percent->Set(
        std::numeric_limits<double>::quiet_NaN());
  }
  catch (...) {
    if (should_log_sampling_error(cpu_sampling_error_log_ts())) {
      log_error("CPU metrics sampling failed due to an unknown error");
    }
    gauges.system_cpu_usage_percent->Set(
        std::numeric_limits<double>::quiet_NaN());
  }
}

void
MetricsRegistry::Sampler::sample_inference_throughput()
{
  auto& gauges = registry_->gauges_;
  if (gauges.inference_throughput == nullptr) {
    return;
  }

  if (auto snap = perf_observer::snapshot()) {
    gauges.inference_throughput->Set(snap->throughput);
  }
}

void
MetricsRegistry::Sampler::sample_process_resident_memory()
{
  auto& gauges = registry_->gauges_;
  if (gauges.process_resident_memory_bytes == nullptr) {
    return;
  }

  if (auto rss_bytes = monitoring::detail::read_process_rss_bytes();
      rss_bytes.has_value()) {
    gauges.process_resident_memory_bytes->Set(*rss_bytes);
  } else {
    gauges.process_resident_memory_bytes->Set(
        std::numeric_limits<double>::quiet_NaN());
  }
}

void
MetricsRegistry::Sampler::sample_process_open_fds()
{
  auto& gauges = registry_->gauges_;
  if (gauges.process_open_fds == nullptr) {
    return;
  }

  if (auto fds = monitoring::detail::read_process_open_fds(); fds.has_value()) {
    gauges.process_open_fds->Set(*fds);
  } else {
    gauges.process_open_fds->Set(std::numeric_limits<double>::quiet_NaN());
  }
}

void
MetricsRegistry::Sampler::sample_gpu_stats()
{
  const auto& providers = registry_->providers_;
  auto& caches = registry_->caches_;
  auto& families = registry_->families_;
  if (!providers.gpu_stats_provider) {
    return;
  }

  try {
    auto gstats = providers.gpu_stats_provider();
    std::unordered_set<int> seen_indices;
    seen_indices.reserve(gstats.size());
    for (const auto& stats : gstats) {
      seen_indices.insert(stats.index);
      const std::string label = std::to_string(stats.index);

      ensure_gpu_gauge(
          caches.gpu.utilization, families.gpu_utilization, stats.index, label)
          ->Set(stats.util_percent);
      ensure_gpu_gauge(
          caches.gpu.memory_used, families.gpu_memory_used_bytes, stats.index,
          label)
          ->Set(stats.mem_used_bytes);
      ensure_gpu_gauge(
          caches.gpu.memory_total, families.gpu_memory_total_bytes, stats.index,
          label)
          ->Set(stats.mem_total_bytes);

      set_or_clear_nan(
          caches.gpu.temperature, families.gpu_temperature, stats.index, label,
          stats.temperature_celsius);
      set_or_clear_nan(
          caches.gpu.power, families.gpu_power, stats.index, label,
          stats.power_watts);
    }

    clear_missing_gauges(
        caches.gpu.utilization, families.gpu_utilization, seen_indices);
    clear_missing_gauges(
        caches.gpu.memory_used, families.gpu_memory_used_bytes, seen_indices);
    clear_missing_gauges(
        caches.gpu.memory_total, families.gpu_memory_total_bytes, seen_indices);
    clear_missing_gauges(
        caches.gpu.temperature, families.gpu_temperature, seen_indices);
    clear_missing_gauges(caches.gpu.power, families.gpu_power, seen_indices);
  }
  catch (const std::exception& e) {
    if (should_log_sampling_error(gpu_sampling_error_log_ts())) {
      log_error(std::format("GPU metrics sampling failed: {}", e.what()));
    }
  }
  catch (...) {
    if (should_log_sampling_error(gpu_sampling_error_log_ts())) {
      log_error("GPU metrics sampling failed due to an unknown error");
    }
  }
}

void
MetricsRegistry::Sampler::perform_sampling_request_nb()
{
  sample_cpu_usage();
  sample_inference_throughput();
  sample_process_resident_memory();
  sample_process_open_fds();
  sample_gpu_stats();
}

void
MetricsRegistry::Sampler::sampling_loop(const std::stop_token& stop)
{
  using namespace std::chrono_literals;
  auto next_sleep = 1000ms;
  while (!stop.stop_requested()) {
    {
      std::scoped_lock<std::mutex> lock(registry_->mutexes_.sampling);
      perform_sampling_request_nb();
    }
    for (auto slept = 0ms; slept < next_sleep && !stop.stop_requested();
         slept += 50ms) {
      std::this_thread::sleep_for(50ms);
    }
  }
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

}  // namespace starpu_server
