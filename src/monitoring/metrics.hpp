#pragma once

#include <prometheus/counter.h>
#include <prometheus/gauge.h>
#include <prometheus/histogram.h>
#include <prometheus/registry.h>

#include <atomic>
#include <cstddef>
#include <memory>

namespace prometheus {
class Exposer;
}  // namespace prometheus

namespace starpu_server {

class MetricsRegistry {
 public:
  explicit MetricsRegistry(int port);
  ~MetricsRegistry();

  std::shared_ptr<prometheus::Registry> registry;
  prometheus::Counter* requests_total;
  prometheus::Histogram* inference_latency;
  prometheus::Gauge* queue_size_gauge;

 private:
  std::unique_ptr<prometheus::Exposer> exposer_;
};

extern std::atomic<std::shared_ptr<MetricsRegistry>> metrics;

bool init_metrics(int port);
void shutdown_metrics();
void set_queue_size(std::size_t size);

}  // namespace starpu_server
