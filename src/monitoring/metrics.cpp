#include "monitoring/metrics.hpp"

#include <prometheus/counter.h>
#include <prometheus/exposer.h>
#include <prometheus/gauge.h>
#include <prometheus/histogram.h>
#include <prometheus/registry.h>

#include <memory>
#include <string>

namespace starpu_server {

std::shared_ptr<prometheus::Registry> metrics_registry;
prometheus::Counter* requests_total = nullptr;
prometheus::Histogram* inference_latency = nullptr;
prometheus::Gauge* queue_size_gauge = nullptr;

namespace {
std::unique_ptr<prometheus::Exposer> exposer;
}

void
init_metrics(int port)
{
  metrics_registry = std::make_shared<prometheus::Registry>();
  exposer =
      std::make_unique<prometheus::Exposer>("0.0.0.0:" + std::to_string(port));
  exposer->RegisterCollectable(metrics_registry);

  auto& counter_family = prometheus::BuildCounter()
                             .Name("requests_total")
                             .Help("Total requests received")
                             .Register(*metrics_registry);
  requests_total = &counter_family.Add({});

  auto& histogram_family = prometheus::BuildHistogram()
                               .Name("inference_latency_ms")
                               .Help("Inference latency in milliseconds")
                               .Register(*metrics_registry);
  inference_latency = &histogram_family.Add(
      {}, prometheus::Histogram::BucketBoundaries{
              1, 5, 10, 25, 50, 100, 250, 500, 1000});

  auto& gauge_family = prometheus::BuildGauge()
                           .Name("inference_queue_size")
                           .Help("Number of jobs in the inference queue")
                           .Register(*metrics_registry);
  queue_size_gauge = &gauge_family.Add({});
}

}  // namespace starpu_server
