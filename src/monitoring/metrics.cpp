#include "monitoring/metrics.hpp"

#include <prometheus/exposer.h>
#include <prometheus/histogram.h>

#include <memory>
#include <string>

namespace starpu_server {

MetricsRegistry::MetricsRegistry(int port)
    : registry(std::make_shared<prometheus::Registry>()),
      requests_total(nullptr), inference_latency(nullptr),
      queue_size_gauge(nullptr), exposer_(
                                     std::make_unique<prometheus::Exposer>(
                                         "0.0.0.0:" + std::to_string(port)))
{
  exposer_->RegisterCollectable(registry);

  auto& counter_family = prometheus::BuildCounter()
                             .Name("requests_total")
                             .Help("Total requests received")
                             .Register(*registry);
  requests_total = &counter_family.Add({});

  auto& histogram_family = prometheus::BuildHistogram()
                               .Name("inference_latency_ms")
                               .Help("Inference latency in milliseconds")
                               .Register(*registry);
  inference_latency = &histogram_family.Add(
      {}, prometheus::Histogram::BucketBoundaries{
              1, 5, 10, 25, 50, 100, 250, 500, 1000});

  auto& gauge_family = prometheus::BuildGauge()
                           .Name("inference_queue_size")
                           .Help("Number of jobs in the inference queue")
                           .Register(*registry);
  queue_size_gauge = &gauge_family.Add({});
}

MetricsRegistry::~MetricsRegistry()
{
  if (exposer_ && registry) {
    exposer_->RemoveCollectable(registry);
  }
}

std::atomic<std::shared_ptr<MetricsRegistry>> metrics{nullptr};

void
init_metrics(int port)
{
  auto new_metrics = std::make_shared<MetricsRegistry>(port);
  metrics.store(std::move(new_metrics), std::memory_order_release);
}

void
shutdown_metrics()
{
  metrics.store(nullptr, std::memory_order_release);
}

}  // namespace starpu_server
