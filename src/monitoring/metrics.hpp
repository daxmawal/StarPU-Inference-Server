#pragma once

#include <prometheus/counter.h>
#include <prometheus/gauge.h>
#include <prometheus/histogram.h>
#include <prometheus/registry.h>

#include <memory>

namespace starpu_server {

extern std::shared_ptr<prometheus::Registry> metrics_registry;
extern prometheus::Counter* requests_total;
extern prometheus::Histogram* inference_latency;
extern prometheus::Gauge* queue_size_gauge;

void init_metrics(int port);
void shutdown_metrics();

}  // namespace starpu_server
