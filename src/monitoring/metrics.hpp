#pragma once

#include <prometheus/counter.h>
#include <prometheus/gauge.h>
#include <prometheus/histogram.h>
#include <prometheus/registry.h>

#include <cstddef>
#include <memory>

namespace prometheus {
class Exposer;
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

 private:
  std::unique_ptr<prometheus::Exposer> exposer_;
};

bool init_metrics(int port);
void shutdown_metrics();
void set_queue_size(std::size_t size);
auto get_metrics() -> std::shared_ptr<MetricsRegistry>;

}  // namespace starpu_server
