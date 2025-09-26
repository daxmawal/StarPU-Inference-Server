#pragma once

#include <prometheus/counter.h>
#include <prometheus/gauge.h>
#include <prometheus/histogram.h>
#include <prometheus/registry.h>

#include <cstddef>
#include <memory>
#include <thread>
#include <unordered_map>

namespace prometheus {
class Exposer;
template <typename T>
class Family;
}  // namespace prometheus

namespace starpu_server {

class MetricsRegistry {
 public:
  explicit MetricsRegistry(int port);
  ~MetricsRegistry() noexcept;

  std::shared_ptr<prometheus::Registry> registry;
  prometheus::Counter* requests_total;
  prometheus::Histogram* inference_latency;
  prometheus::Gauge* queue_size_gauge;

  prometheus::Gauge* system_cpu_usage_percent{nullptr};
  prometheus::Family<prometheus::Gauge>* gpu_utilization_family{nullptr};
  prometheus::Family<prometheus::Gauge>* gpu_memory_used_bytes_family{nullptr};
  prometheus::Family<prometheus::Gauge>* gpu_memory_total_bytes_family{nullptr};

 private:
  std::unique_ptr<prometheus::Exposer> exposer_;
  std::jthread sampler_thread_;
  std::unordered_map<int, prometheus::Gauge*> gpu_utilization_gauges_;
  std::unordered_map<int, prometheus::Gauge*> gpu_memory_used_gauges_;
  std::unordered_map<int, prometheus::Gauge*> gpu_memory_total_gauges_;

  void sampling_loop(const std::stop_token& stop);
};

bool init_metrics(int port);
void shutdown_metrics();
void set_queue_size(std::size_t size);
auto get_metrics() -> std::shared_ptr<MetricsRegistry>;

}  // namespace starpu_server
