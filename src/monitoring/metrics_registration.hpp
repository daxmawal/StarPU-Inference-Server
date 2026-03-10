#pragma once

#include <string>
#include <string_view>

#include "metrics_constants.hpp"
#include "monitoring/metrics.hpp"

namespace starpu_server::detail {

inline auto
register_counter_metric(
    prometheus::Registry& registry, std::string_view name,
    std::string_view help) -> prometheus::Counter*
{
  auto& family = prometheus::BuildCounter()
                     .Name(std::string(name))
                     .Help(std::string(help))
                     .Register(registry);
  return &family.Add({});
}

inline auto
register_gauge_metric(
    prometheus::Registry& registry, std::string_view name,
    std::string_view help) -> prometheus::Gauge*
{
  auto& family = prometheus::BuildGauge()
                     .Name(std::string(name))
                     .Help(std::string(help))
                     .Register(registry);
  return &family.Add({});
}

inline auto
register_histogram_metric(
    prometheus::Registry& registry, std::string_view name,
    std::string_view help,
    const prometheus::Histogram::BucketBoundaries& buckets)
    -> prometheus::Histogram*
{
  auto& family = prometheus::BuildHistogram()
                     .Name(std::string(name))
                     .Help(std::string(help))
                     .Register(registry);
  return &family.Add({}, buckets);
}

inline void
register_request_counters_and_families(
    prometheus::Registry& registry, MetricsRegistry::CounterMetrics& counters,
    MetricsRegistry::FamilyMetrics& families)
{
  auto& counter_family = prometheus::BuildCounter()
                             .Name("requests_total")
                             .Help("Total requests received")
                             .Register(registry);
  counters.requests_total = &counter_family.Add({});

  auto& status_family = prometheus::BuildCounter()
                            .Name("requests_by_status_total")
                            .Help(
                                "Total requests grouped by gRPC status code "
                                "and model name")
                            .Register(registry);
  families.requests_by_status = &status_family;
  (void)families.requests_by_status->Add(
      {{"code", "unlabeled"}, {"model", "unlabeled"}});

  auto& received_family = prometheus::BuildCounter()
                              .Name("requests_received_total")
                              .Help("Total requests received by model")
                              .Register(registry);
  families.requests_received = &received_family;
  (void)families.requests_received->Add({{"model", "unlabeled"}});

  auto& completed_family = prometheus::BuildCounter()
                               .Name("inference_completed_total")
                               .Help("Total logical inferences completed")
                               .Register(registry);
  families.inference_completed = &completed_family;
  (void)families.inference_completed->Add({{"model", "unlabeled"}});

  auto& failures_family = prometheus::BuildCounter()
                              .Name("inference_failures_total")
                              .Help(
                                  "Inference failures grouped by stage and "
                                  "reason")
                              .Register(registry);
  families.inference_failures = &failures_family;
  (void)families.inference_failures->Add(
      {{"stage", "unlabeled"},
       {"reason", "unlabeled"},
       {"model", "unlabeled"}});

  auto& rejected_family =
      prometheus::BuildCounter()
          .Name("requests_rejected_total")
          .Help("Total requests rejected (e.g., queue full)")
          .Register(registry);
  counters.requests_rejected_total = &rejected_family.Add({});
}

inline void
register_queue_runtime_and_batching_metrics(
    prometheus::Registry& registry, MetricsRegistry::GaugeMetrics& gauges,
    MetricsRegistry::HistogramMetrics& histograms)
{
  histograms.inference_latency = register_histogram_metric(
      registry, "inference_latency_ms", "Inference latency in milliseconds",
      kInferenceLatencyMsBuckets);
  gauges.queue_size = register_gauge_metric(
      registry, "inference_queue_size",
      "Number of jobs in the inference queue");
  gauges.queue_capacity = register_gauge_metric(
      registry, "inference_max_queue_size",
      "Configured maximum inference queue capacity");
  gauges.queue_fill_ratio = register_gauge_metric(
      registry, "inference_queue_fill_ratio",
      "Queue occupancy ratio (queue_size / max_queue_size)");
  gauges.inflight_tasks = register_gauge_metric(
      registry, "inference_inflight_tasks",
      "Number of StarPU tasks currently submitted and not yet completed");
  gauges.max_inflight_tasks = register_gauge_metric(
      registry, "inference_max_inflight_tasks",
      "Configured cap on inflight StarPU tasks (0 means unbounded)");
  gauges.starpu_worker_busy_ratio = register_gauge_metric(
      registry, "starpu_worker_busy_ratio",
      "Approximate ratio of inflight tasks to max inflight limit (0-1, 0 when "
      "unbounded)");
  gauges.starpu_prepared_queue_depth = register_gauge_metric(
      registry, "starpu_prepared_queue_depth",
      "Number of batched jobs waiting for StarPU submission");
  gauges.system_cpu_usage_percent = register_gauge_metric(
      registry, "system_cpu_usage_percent",
      "System-wide CPU utilization percentage (0-100)");
  gauges.inference_throughput = register_gauge_metric(
      registry, "inference_throughput_rps",
      "Rolling throughput of logical inferences/s based on completed jobs");
  gauges.process_resident_memory_bytes = register_gauge_metric(
      registry, "process_resident_memory_bytes",
      "Resident Set Size of the server process");
  gauges.process_open_fds = register_gauge_metric(
      registry, "process_open_fds", "Number of open file descriptors");
  gauges.server_health_state = register_gauge_metric(
      registry, "server_health_state",
      "Server health state: 1=ready, 0=not ready or shutting down");
  histograms.queue_latency = register_histogram_metric(
      registry, "inference_queue_latency_ms", "Time spent waiting in the queue",
      kInferenceLatencyMsBuckets);
  histograms.batch_efficiency = register_histogram_metric(
      registry, "inference_batch_efficiency_ratio",
      "Ratio of effective batch size to logical request count",
      kBatchEfficiencyBuckets);
  gauges.batch_pending_jobs = register_gauge_metric(
      registry, "inference_batch_collect_pending_jobs",
      "Number of requests aggregated in the current batch collection");
  histograms.batch_collect_latency = register_histogram_metric(
      registry, "inference_batch_collect_ms", "Time spent collecting a batch",
      kInferenceLatencyMsBuckets);
}

inline void
register_congestion_gauges(
    prometheus::Registry& registry,
    MetricsRegistry::GaugeMetrics::CongestionGaugeMetrics& congestion_gauges)
{
  congestion_gauges.flag = register_gauge_metric(
      registry, "inference_congestion_flag",
      "1 when congestion detector reports congestion, 0 otherwise");
  congestion_gauges.score = register_gauge_metric(
      registry, "inference_congestion_score",
      "Composite congestion pressure score (0-1, heuristic)");
  congestion_gauges.lambda_rps = register_gauge_metric(
      registry, "inference_lambda_rps",
      "Arrival rate (requests/s) over congestion tick");
  congestion_gauges.mu_rps = register_gauge_metric(
      registry, "inference_mu_rps",
      "Completion rate (requests/s) over congestion tick");
  congestion_gauges.rho_ewma = register_gauge_metric(
      registry, "inference_rho_ewma", "Smoothed utilization ratio lambda/mu");
  congestion_gauges.queue_fill_ewma = register_gauge_metric(
      registry, "inference_queue_fill_ratio_ewma",
      "Smoothed queue fill ratio (0-1)");
  congestion_gauges.queue_growth_rate = register_gauge_metric(
      registry, "inference_queue_growth_rate",
      "Queue growth rate dQ/dt (jobs per second)");
  congestion_gauges.queue_p95_ms = register_gauge_metric(
      registry, "inference_queue_latency_p95_ms",
      "p95 queue latency over congestion tick");
  congestion_gauges.queue_p99_ms = register_gauge_metric(
      registry, "inference_queue_latency_p99_ms",
      "p99 queue latency over congestion tick");
  congestion_gauges.e2e_p95_ms = register_gauge_metric(
      registry, "inference_e2e_latency_p95_ms",
      "p95 end-to-end latency over congestion tick");
  congestion_gauges.e2e_p99_ms = register_gauge_metric(
      registry, "inference_e2e_latency_p99_ms",
      "p99 end-to-end latency over congestion tick");
  congestion_gauges.rejection_rps = register_gauge_metric(
      registry, "inference_rejection_rate_rps",
      "Request rejection rate (requests/s)");
}

inline void
register_latency_histograms(
    prometheus::Registry& registry,
    MetricsRegistry::HistogramMetrics& histograms)
{
  histograms.submit_latency = register_histogram_metric(
      registry, "inference_submit_latency_ms",
      "Time spent between dequeue and submission into StarPU",
      kInferenceLatencyMsBuckets);
  histograms.scheduling_latency = register_histogram_metric(
      registry, "inference_scheduling_latency_ms",
      "Time spent waiting for scheduling on a StarPU worker",
      kInferenceLatencyMsBuckets);
  histograms.codelet_latency = register_histogram_metric(
      registry, "inference_codelet_latency_ms",
      "Duration of the StarPU codelet execution", kInferenceLatencyMsBuckets);
  histograms.inference_compute_latency = register_histogram_metric(
      registry, "inference_compute_latency_ms",
      "Model compute time (inference)", kInferenceLatencyMsBuckets);
  histograms.callback_latency = register_histogram_metric(
      registry, "inference_callback_latency_ms",
      "Callback/response handling latency", kInferenceLatencyMsBuckets);
  histograms.preprocess_latency = register_histogram_metric(
      registry, "inference_preprocess_latency_ms",
      "Server-side preprocessing latency", kInferenceLatencyMsBuckets);
  histograms.postprocess_latency = register_histogram_metric(
      registry, "inference_postprocess_latency_ms",
      "Server-side postprocessing latency", kInferenceLatencyMsBuckets);
  histograms.batch_size = register_histogram_metric(
      registry, "inference_batch_size", "Effective batch size executed",
      kBatchSizeBuckets);
  histograms.logical_batch_size = register_histogram_metric(
      registry, "inference_logical_batch_size",
      "Number of logical requests aggregated into a batch", kBatchSizeBuckets);
}

inline void
register_model_gpu_and_worker_metrics(
    prometheus::Registry& registry,
    MetricsRegistry::HistogramMetrics& histograms,
    MetricsRegistry::FamilyMetrics& families)
{
  auto& model_load_hist_family = prometheus::BuildHistogram()
                                     .Name("model_load_duration_ms")
                                     .Help("Duration of model load and wiring")
                                     .Register(registry);
  histograms.model_load_duration =
      &model_load_hist_family.Add({}, kModelLoadDurationMsBuckets);

  auto& model_load_fail_family = prometheus::BuildCounter()
                                     .Name("model_load_failures_total")
                                     .Help("Total failed model load attempts")
                                     .Register(registry);
  families.model_load_failures = &model_load_fail_family;
  (void)families.model_load_failures->Add({{"model", "unlabeled"}});

  auto& models_loaded_family =
      prometheus::BuildGauge()
          .Name("models_loaded")
          .Help("Flag indicating a model is loaded on a device")
          .Register(registry);
  families.models_loaded = &models_loaded_family;
  (void)families.models_loaded->Add(
      {{"model", "unlabeled"}, {"device", "unknown"}});

  families.gpu_utilization =
      &prometheus::BuildGauge()
           .Name("gpu_utilization_percent")
           .Help("GPU utilization percentage per GPU (0-100)")
           .Register(registry);
  families.gpu_memory_used_bytes =
      &prometheus::BuildGauge()
           .Name("gpu_memory_used_bytes")
           .Help("Used GPU memory in bytes per GPU")
           .Register(registry);
  families.gpu_memory_total_bytes =
      &prometheus::BuildGauge()
           .Name("gpu_memory_total_bytes")
           .Help("Total GPU memory in bytes per GPU")
           .Register(registry);

  families.gpu_temperature = &prometheus::BuildGauge()
                                  .Name("gpu_temperature_celsius")
                                  .Help("Reported GPU temperature in Celsius")
                                  .Register(registry);

  families.gpu_power = &prometheus::BuildGauge()
                            .Name("gpu_power_watts")
                            .Help("Reported GPU power draw in Watts")
                            .Register(registry);

  auto& starpu_runtime_family = prometheus::BuildHistogram()
                                    .Name("starpu_task_runtime_ms")
                                    .Help("Wall-clock runtime of a StarPU task")
                                    .Register(registry);
  histograms.starpu_task_runtime =
      &starpu_runtime_family.Add({}, kTaskRuntimeMsBuckets);

  auto& compute_latency_by_worker_family =
      prometheus::BuildHistogram()
          .Name("inference_compute_latency_ms_by_worker")
          .Help(
              "Model compute latency by worker and device "
              "(callback start - inference start)")
          .Register(registry);
  families.inference_compute_latency_by_worker =
      &compute_latency_by_worker_family;

  auto& starpu_runtime_by_worker_family =
      prometheus::BuildHistogram()
          .Name("starpu_task_runtime_ms_by_worker")
          .Help("StarPU codelet runtime by worker/device")
          .Register(registry);
  families.starpu_task_runtime_by_worker = &starpu_runtime_by_worker_family;

  auto& worker_inflight_family =
      prometheus::BuildGauge()
          .Name("starpu_worker_inflight_tasks")
          .Help("Inflight StarPU tasks per worker/device")
          .Register(registry);
  families.starpu_worker_inflight = &worker_inflight_family;

  auto& io_copy_latency_family =
      prometheus::BuildHistogram()
          .Name("inference_io_copy_ms")
          .Help("Host/device copy latency by direction/device/worker")
          .Register(registry);
  families.io_copy_latency = &io_copy_latency_family;

  auto& transfer_bytes_family =
      prometheus::BuildCounter()
          .Name("inference_transfer_bytes_total")
          .Help("Total bytes transferred by direction/device/worker")
          .Register(registry);
  families.transfer_bytes = &transfer_bytes_family;
}
}  // namespace starpu_server::detail
