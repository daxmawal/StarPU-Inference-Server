#include "monitoring/metrics.hpp"

#include <prometheus/exposer.h>
#include <prometheus/histogram.h>

#include <memory>
#include <string>

#include "utils/logger.hpp"

namespace starpu_server {

namespace {
const prometheus::Histogram::BucketBoundaries kInferenceLatencyMsBuckets{
    1, 5, 10, 25, 50, 100, 250, 500, 1000};
}  // namespace

MetricsRegistry::MetricsRegistry(int port)
    : registry(std::make_shared<prometheus::Registry>()),
      requests_total(nullptr), inference_latency(nullptr),
      queue_size_gauge(nullptr), exposer_(nullptr)
{
  try {
    exposer_ = std::make_unique<prometheus::Exposer>(
        "0.0.0.0:" + std::to_string(port));
    exposer_->RegisterCollectable(registry);
  }
  catch (const std::exception& e) {
    log_error(std::string("Failed to initialize metrics exposer: ") + e.what());
    throw;
  }

  auto& counter_family = prometheus::BuildCounter()
                             .Name("requests_total")
                             .Help("Total requests received")
                             .Register(*registry);
  requests_total = &counter_family.Add({});

  auto& histogram_family = prometheus::BuildHistogram()
                               .Name("inference_latency_ms")
                               .Help("Inference latency in milliseconds")
                               .Register(*registry);
  inference_latency = &histogram_family.Add({}, kInferenceLatencyMsBuckets);

  auto& gauge_family = prometheus::BuildGauge()
                           .Name("inference_queue_size")
                           .Help("Number of jobs in the inference queue")
                           .Register(*registry);
  queue_size_gauge = &gauge_family.Add({});
}

MetricsRegistry::~MetricsRegistry() noexcept
{
  if (exposer_ && registry) {
    try {
      exposer_->RemoveCollectable(registry);
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
}  // namespace

auto
init_metrics(int port) -> bool
{
  std::shared_ptr<MetricsRegistry> expected{nullptr};

  try {
    auto new_metrics = std::make_shared<MetricsRegistry>(port);

    if (!metrics_atomic().compare_exchange_strong(
            expected, new_metrics, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
      log_warning("Metrics were previously initialized");
      return false;
    }

    set_queue_size(0);
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
  auto metrics_ptr = metrics_atomic().load(std::memory_order_acquire);
  if (metrics_ptr && metrics_ptr->queue_size_gauge != nullptr) {
    metrics_ptr->queue_size_gauge->Set(static_cast<double>(size));
  }
}

}  // namespace starpu_server
