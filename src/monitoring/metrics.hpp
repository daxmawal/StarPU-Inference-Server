#pragma once

#include <prometheus/counter.h>
#include <prometheus/gauge.h>
#include <prometheus/histogram.h>
#include <prometheus/registry.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>

namespace prometheus {
class Exposer;
template <typename T>
class Family;
}  // namespace prometheus

namespace starpu_server {

class MetricsRegistry {
 public:
  struct GpuSample {
    int index{0};
    double util_percent{0.0};
    double mem_used_bytes{0.0};
    double mem_total_bytes{0.0};
  };

  using GpuStatsProvider = std::function<std::vector<GpuSample>()>;
  using CpuUsageProvider = std::function<std::optional<double>()>;

  explicit MetricsRegistry(int port);
  MetricsRegistry(
      int port, GpuStatsProvider gpu_provider, CpuUsageProvider cpu_provider,
      bool start_sampler_thread = true);
  ~MetricsRegistry() noexcept;

  std::shared_ptr<prometheus::Registry> registry;
  prometheus::Counter* requests_total;
  prometheus::Histogram* inference_latency;
  prometheus::Gauge* queue_size_gauge;

  prometheus::Gauge* system_cpu_usage_percent{nullptr};
  prometheus::Family<prometheus::Gauge>* gpu_utilization_family{nullptr};
  prometheus::Family<prometheus::Gauge>* gpu_memory_used_bytes_family{nullptr};
  prometheus::Family<prometheus::Gauge>* gpu_memory_total_bytes_family{nullptr};

  void run_sampling_iteration();
  void request_stop();

 private:
  void initialize(int port, bool start_sampler_thread);
  void perform_sampling_iteration();
  void sampling_loop(const std::stop_token& stop);

  std::unique_ptr<prometheus::Exposer> exposer_;
  std::jthread sampler_thread_;
  GpuStatsProvider gpu_stats_provider_;
  CpuUsageProvider cpu_usage_provider_;
  std::unordered_map<int, prometheus::Gauge*> gpu_utilization_gauges_;
  std::unordered_map<int, prometheus::Gauge*> gpu_memory_used_gauges_;
  std::unordered_map<int, prometheus::Gauge*> gpu_memory_total_gauges_;
};

bool init_metrics(int port);
void shutdown_metrics();
void set_queue_size(std::size_t size);
auto get_metrics() -> std::shared_ptr<MetricsRegistry>;

}  // namespace starpu_server
