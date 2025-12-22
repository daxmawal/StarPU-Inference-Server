#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <prometheus/client_metric.h>
#include <prometheus/metric_family.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "monitoring/metrics.hpp"

using namespace starpu_server;

namespace {
class TemporaryStatmFile {
 public:
  explicit TemporaryStatmFile(std::string_view contents)
  {
    const auto timestamp =
        std::chrono::steady_clock::now().time_since_epoch().count();
    path_ = std::filesystem::temp_directory_path() /
            std::filesystem::path(
                "starpu_rss_statm_" + std::to_string(timestamp) + ".test");
    std::ofstream out{path_};
    out << contents;
  }

  ~TemporaryStatmFile()
  {
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }

  auto path() const -> const std::filesystem::path& { return path_; }

 private:
  std::filesystem::path path_;
};

class StatmPathGuard {
 public:
  explicit StatmPathGuard(std::filesystem::path path)
  {
    monitoring::detail::set_process_rss_bytes_path_for_test(std::move(path));
  }

  ~StatmPathGuard()
  {
    monitoring::detail::reset_process_rss_bytes_path_for_test();
  }
};

class PageSizeProviderGuard {
 public:
  explicit PageSizeProviderGuard(std::function<long()> provider)
  {
    monitoring::detail::set_process_page_size_provider_for_test(
        std::move(provider));
  }

  ~PageSizeProviderGuard()
  {
    monitoring::detail::reset_process_page_size_provider_for_test();
  }
};

void
AssertMetricsInitialized(const std::shared_ptr<MetricsRegistry>& metrics)
{
  ASSERT_NE(metrics, nullptr);
  ASSERT_NE(metrics->registry(), nullptr);
  ASSERT_NE(metrics->requests_total(), nullptr);
  ASSERT_NE(metrics->requests_rejected_total(), nullptr);
  ASSERT_NE(metrics->inference_latency(), nullptr);
  ASSERT_NE(metrics->queue_size_gauge(), nullptr);
  ASSERT_NE(metrics->inflight_tasks_gauge(), nullptr);
  ASSERT_NE(metrics->max_inflight_tasks_gauge(), nullptr);
  ASSERT_NE(metrics->starpu_worker_busy_ratio_gauge(), nullptr);
  ASSERT_NE(metrics->starpu_prepared_queue_depth_gauge(), nullptr);
  ASSERT_NE(metrics->batch_pending_jobs_gauge(), nullptr);
  ASSERT_NE(metrics->batch_efficiency_histogram(), nullptr);
  ASSERT_NE(metrics->starpu_task_runtime_histogram(), nullptr);
  ASSERT_NE(metrics->inference_compute_latency_by_worker_family(), nullptr);
  ASSERT_NE(metrics->starpu_task_runtime_by_worker_family(), nullptr);
  ASSERT_NE(metrics->starpu_worker_inflight_family(), nullptr);
  ASSERT_NE(metrics->io_copy_latency_family(), nullptr);
  ASSERT_NE(metrics->transfer_bytes_family(), nullptr);
  ASSERT_NE(metrics->server_health_state_gauge(), nullptr);
  ASSERT_NE(metrics->inference_throughput_gauge(), nullptr);
  ASSERT_NE(metrics->process_resident_memory_gauge(), nullptr);
  ASSERT_NE(metrics->process_open_fds_gauge(), nullptr);
  ASSERT_NE(metrics->queue_fill_ratio_gauge(), nullptr);
  ASSERT_NE(metrics->queue_capacity_gauge(), nullptr);
  ASSERT_NE(metrics->queue_latency_histogram(), nullptr);
  ASSERT_NE(metrics->batch_collect_latency_histogram(), nullptr);
  ASSERT_NE(metrics->submit_latency_histogram(), nullptr);
  ASSERT_NE(metrics->scheduling_latency_histogram(), nullptr);
  ASSERT_NE(metrics->codelet_latency_histogram(), nullptr);
  ASSERT_NE(metrics->inference_compute_latency_histogram(), nullptr);
  ASSERT_NE(metrics->callback_latency_histogram(), nullptr);
  ASSERT_NE(metrics->preprocess_latency_histogram(), nullptr);
  ASSERT_NE(metrics->postprocess_latency_histogram(), nullptr);
  ASSERT_NE(metrics->batch_size_histogram(), nullptr);
  ASSERT_NE(metrics->logical_batch_size_histogram(), nullptr);
  ASSERT_NE(metrics->requests_by_status_family(), nullptr);
  ASSERT_NE(metrics->inference_completed_family(), nullptr);
  ASSERT_NE(metrics->inference_failures_family(), nullptr);
  ASSERT_NE(metrics->model_load_failures_family(), nullptr);
  ASSERT_NE(metrics->models_loaded_family(), nullptr);
  ASSERT_NE(metrics->gpu_temperature_family(), nullptr);
  ASSERT_NE(metrics->gpu_power_family(), nullptr);
}

auto
HasMetric(
    const std::vector<prometheus::MetricFamily>& families,
    std::string_view name) -> bool
{
  return std::ranges::any_of(
      families, [name](const prometheus::MetricFamily& family) {
        return family.name == name;
      });
}

auto
FindFamily(
    const std::vector<prometheus::MetricFamily>& families,
    std::string_view name) -> const prometheus::MetricFamily*
{
  for (const auto& family : families) {
    if (family.name == name) {
      return &family;
    }
  }
  return nullptr;
}

auto
MetricMatchesLabels(
    const prometheus::ClientMetric& metric,
    const std::vector<std::pair<std::string_view, std::string_view>>& labels)
    -> bool
{
  for (const auto& [label_name, label_value] : labels) {
    bool matched = false;
    for (const auto& label : metric.label) {
      if (label.name == label_name && label.value == label_value) {
        matched = true;
        break;
      }
    }
    if (!matched) {
      return false;
    }
  }
  return true;
}

auto
FindMetric(
    const prometheus::MetricFamily& family,
    const std::vector<std::pair<std::string_view, std::string_view>>& labels)
    -> const prometheus::ClientMetric*
{
  for (const auto& metric : family.metric) {
    if (MetricMatchesLabels(metric, labels)) {
      return &metric;
    }
  }
  return nullptr;
}

auto
FindGaugeValue(
    const std::vector<prometheus::MetricFamily>& families,
    std::string_view family_name,
    const std::vector<std::pair<std::string_view, std::string_view>>& labels)
    -> std::optional<double>
{
  const auto* family = FindFamily(families, family_name);
  if (family == nullptr) {
    return std::nullopt;
  }
  const auto* metric = FindMetric(*family, labels);
  if (metric == nullptr) {
    return std::nullopt;
  }
  return metric->gauge.value;
}

auto
FindCounterValue(
    const std::vector<prometheus::MetricFamily>& families,
    std::string_view family_name,
    const std::vector<std::pair<std::string_view, std::string_view>>& labels)
    -> std::optional<double>
{
  const auto* family = FindFamily(families, family_name);
  if (family == nullptr) {
    return std::nullopt;
  }
  const auto* metric = FindMetric(*family, labels);
  if (metric == nullptr) {
    return std::nullopt;
  }
  return metric->counter.value;
}

auto
FindHistogramMetric(
    const std::vector<prometheus::MetricFamily>& families,
    std::string_view family_name,
    const std::vector<std::pair<std::string_view, std::string_view>>& labels)
    -> const prometheus::ClientMetric*
{
  const auto* family = FindFamily(families, family_name);
  if (family == nullptr) {
    return nullptr;
  }
  return FindMetric(*family, labels);
}
}  // namespace

TEST(Metrics, InitializesPointersAndRegistry)
{
  ASSERT_TRUE(init_metrics(0));

  auto metrics = get_metrics();
  AssertMetricsInitialized(metrics);

  const auto families = metrics->registry()->Collect();
  EXPECT_TRUE(HasMetric(families, "requests_total"));
  EXPECT_TRUE(HasMetric(families, "requests_rejected_total"));
  EXPECT_TRUE(HasMetric(families, "requests_by_status_total"));
  EXPECT_TRUE(HasMetric(families, "inference_latency_ms"));
  EXPECT_TRUE(HasMetric(families, "inference_queue_size"));
  EXPECT_TRUE(HasMetric(families, "inference_max_queue_size"));
  EXPECT_TRUE(HasMetric(families, "inference_queue_fill_ratio"));
  EXPECT_TRUE(HasMetric(families, "inference_inflight_tasks"));
  EXPECT_TRUE(HasMetric(families, "inference_max_inflight_tasks"));
  EXPECT_TRUE(HasMetric(families, "server_health_state"));
  EXPECT_TRUE(HasMetric(families, "inference_batch_size"));
  EXPECT_TRUE(HasMetric(families, "inference_logical_batch_size"));
  EXPECT_TRUE(HasMetric(families, "inference_queue_latency_ms"));
  EXPECT_TRUE(HasMetric(families, "inference_batch_collect_ms"));
  EXPECT_TRUE(HasMetric(families, "inference_submit_latency_ms"));
  EXPECT_TRUE(HasMetric(families, "inference_scheduling_latency_ms"));
  EXPECT_TRUE(HasMetric(families, "inference_compute_latency_ms"));
  EXPECT_TRUE(HasMetric(families, "inference_callback_latency_ms"));
  EXPECT_TRUE(HasMetric(families, "inference_preprocess_latency_ms"));
  EXPECT_TRUE(HasMetric(families, "inference_postprocess_latency_ms"));

  shutdown_metrics();
  EXPECT_EQ(get_metrics(), nullptr);
}

TEST(Metrics, RepeatedInitDoesNotAllocateRegistry)
{
  ASSERT_TRUE(init_metrics(0));
  auto first = get_metrics();

  EXPECT_FALSE(init_metrics(0));
  auto second = get_metrics();
  EXPECT_EQ(first, second);

  shutdown_metrics();
  EXPECT_EQ(get_metrics(), nullptr);
}

TEST(Metrics, InitFailsWhenMetricsRegistryThrows)
{
  struct FailureGuard {
    FailureGuard()
    {
      monitoring::detail::set_metrics_init_failure_for_test(true);
    }

    ~FailureGuard()
    {
      monitoring::detail::set_metrics_init_failure_for_test(false);
    }
  } guard;

  EXPECT_FALSE(init_metrics(0));
  EXPECT_EQ(get_metrics(), nullptr);
}

TEST(Metrics, MetricsDestructionLogsRemovalFailure)
{
  class ThrowingRemoveHandle : public MetricsRegistry::ExposerHandle {
   public:
    void RegisterCollectable(
        const std::shared_ptr<prometheus::Collectable>&) override
    {
    }

    void RemoveCollectable(
        const std::shared_ptr<prometheus::Collectable>&) override
    {
      throw std::runtime_error("remove failed");
    }
  };

  testing::internal::CaptureStderr();
  {
    auto handle = std::make_unique<ThrowingRemoveHandle>();
    MetricsRegistry registry(
        0, [] { return std::vector<MetricsRegistry::GpuSample>{}; },
        [] { return std::optional<double>{}; }, false, std::move(handle));
  }
  const std::string log = testing::internal::GetCapturedStderr();
  EXPECT_NE(
      log.find("Failed to remove metrics registry collectable"),
      std::string::npos);
}

TEST(Metrics, AccessorsReturnAllocatedFamiliesAndGauges)
{
  ASSERT_TRUE(init_metrics(0));

  auto metrics = get_metrics();
  AssertMetricsInitialized(metrics);

  EXPECT_NE(metrics->system_cpu_usage_percent(), nullptr);
  EXPECT_NE(metrics->gpu_utilization_family(), nullptr);
  EXPECT_NE(metrics->gpu_memory_used_bytes_family(), nullptr);
  EXPECT_NE(metrics->gpu_memory_total_bytes_family(), nullptr);

  shutdown_metrics();
}

TEST(MetricsRegistry, RecordsWorkerAndTransferMetrics)
{
  MetricsRegistry metrics(
      0, [] { return std::vector<MetricsRegistry::GpuSample>{}; },
      [] { return std::optional<double>{}; }, false);

  metrics.observe_compute_latency_by_worker(3, 1, "cpu", 12.5);
  metrics.observe_task_runtime_by_worker(3, 1, "cpu", 4.0);
  metrics.set_worker_inflight_gauge(3, 1, "cpu", 2);
  metrics.observe_io_copy_latency("h2d", 3, 1, "cpu", 7.0);
  metrics.increment_transfer_bytes("h2d", 3, 1, "cpu", 1024);
  metrics.increment_completed_counter("model-x", 5);

  const auto families = metrics.registry()->Collect();

  const auto* compute_metric = FindHistogramMetric(
      families, "inference_compute_latency_ms_by_worker",
      {{"worker_id", "3"}, {"device", "1"}, {"worker_type", "cpu"}});
  ASSERT_NE(compute_metric, nullptr);
  EXPECT_EQ(compute_metric->histogram.sample_count, 1);
  EXPECT_DOUBLE_EQ(compute_metric->histogram.sample_sum, 12.5);

  const auto* runtime_metric = FindHistogramMetric(
      families, "starpu_task_runtime_ms_by_worker",
      {{"worker_id", "3"}, {"device", "1"}, {"worker_type", "cpu"}});
  ASSERT_NE(runtime_metric, nullptr);
  EXPECT_EQ(runtime_metric->histogram.sample_count, 1);
  EXPECT_DOUBLE_EQ(runtime_metric->histogram.sample_sum, 4.0);

  const auto inflight_value = FindGaugeValue(
      families, "starpu_worker_inflight_tasks",
      {{"worker_id", "3"}, {"device", "1"}, {"worker_type", "cpu"}});
  ASSERT_TRUE(inflight_value.has_value());
  EXPECT_DOUBLE_EQ(*inflight_value, 2.0);

  const auto* io_metric = FindHistogramMetric(
      families, "inference_io_copy_ms",
      {{"direction", "h2d"},
       {"worker_id", "3"},
       {"device", "1"},
       {"worker_type", "cpu"}});
  ASSERT_NE(io_metric, nullptr);
  EXPECT_EQ(io_metric->histogram.sample_count, 1);
  EXPECT_DOUBLE_EQ(io_metric->histogram.sample_sum, 7.0);

  const auto transfer_value = FindCounterValue(
      families, "inference_transfer_bytes_total",
      {{"direction", "h2d"},
       {"worker_id", "3"},
       {"device", "1"},
       {"worker_type", "cpu"}});
  ASSERT_TRUE(transfer_value.has_value());
  EXPECT_DOUBLE_EQ(*transfer_value, 1024.0);

  const auto completed_value = FindCounterValue(
      families, "inference_completed_total", {{"model", "model-x"}});
  ASSERT_TRUE(completed_value.has_value());
  EXPECT_DOUBLE_EQ(*completed_value, 5.0);
}

TEST(Metrics, IncrementRequestStatusUsesExpectedLabels)
{
  ASSERT_TRUE(init_metrics(0));
  struct MetricsGuard {
    ~MetricsGuard() { shutdown_metrics(); }
  } guard;

  const std::string model_name = "status-model";
  struct StatusCase {
    int code;
    std::string_view label;
  };
  const std::vector<StatusCase> cases{
      {3, "INVALID_ARGUMENT"},
      {4, "DEADLINE_EXCEEDED"},
      {5, "NOT_FOUND"},
      {7, "PERMISSION_DENIED"},
      {8, "RESOURCE_EXHAUSTED"},
      {9, "FAILED_PRECONDITION"},
      {10, "ABORTED"},
      {11, "OUT_OF_RANGE"},
      {12, "UNIMPLEMENTED"},
      {13, "INTERNAL"},
      {14, "UNAVAILABLE"},
      {16, "UNAUTHENTICATED"},
      {42, "42"},
  };

  for (const auto& entry : cases) {
    increment_request_status(entry.code, model_name);
  }

  const auto metrics = get_metrics();
  ASSERT_NE(metrics, nullptr);
  const auto families = metrics->registry()->Collect();

  for (const auto& entry : cases) {
    SCOPED_TRACE(std::to_string(entry.code));
    const auto value = FindCounterValue(
        families, "requests_by_status_total",
        {{"code", entry.label}, {"model", model_name}});
    ASSERT_TRUE(value.has_value());
    EXPECT_DOUBLE_EQ(*value, 1.0);
  }
}

TEST(Metrics, SetQueueFillRatioUpdatesGauge)
{
  ASSERT_TRUE(init_metrics(0));
  struct MetricsGuard {
    ~MetricsGuard() { shutdown_metrics(); }
  } guard;

  set_queue_fill_ratio(3, 10);

  const auto metrics = get_metrics();
  ASSERT_NE(metrics, nullptr);
  const auto families = metrics->registry()->Collect();
  const auto ratio = FindGaugeValue(families, "inference_queue_fill_ratio", {});
  ASSERT_TRUE(ratio.has_value());
  EXPECT_DOUBLE_EQ(*ratio, 0.3);
}

TEST(Metrics, ObserveBatchAndHealthMetrics)
{
  ASSERT_TRUE(init_metrics(0));
  struct MetricsGuard {
    ~MetricsGuard() { shutdown_metrics(); }
  } guard;

  set_server_health(true);
  observe_batch_size(12);
  observe_logical_batch_size(24);
  observe_batch_efficiency(0.85);
  observe_starpu_task_runtime(5.5);

  const auto metrics = get_metrics();
  ASSERT_NE(metrics, nullptr);
  const auto families = metrics->registry()->Collect();

  const auto health = FindGaugeValue(families, "server_health_state", {});
  ASSERT_TRUE(health.has_value());
  EXPECT_DOUBLE_EQ(*health, 1.0);

  const auto* batch_metric =
      FindHistogramMetric(families, "inference_batch_size", {});
  ASSERT_NE(batch_metric, nullptr);
  EXPECT_EQ(batch_metric->histogram.sample_count, 1);
  EXPECT_DOUBLE_EQ(batch_metric->histogram.sample_sum, 12.0);

  const auto* logical_metric =
      FindHistogramMetric(families, "inference_logical_batch_size", {});
  ASSERT_NE(logical_metric, nullptr);
  EXPECT_EQ(logical_metric->histogram.sample_count, 1);
  EXPECT_DOUBLE_EQ(logical_metric->histogram.sample_sum, 24.0);

  const auto* efficiency_metric =
      FindHistogramMetric(families, "inference_batch_efficiency_ratio", {});
  ASSERT_NE(efficiency_metric, nullptr);
  EXPECT_EQ(efficiency_metric->histogram.sample_count, 1);
  EXPECT_DOUBLE_EQ(efficiency_metric->histogram.sample_sum, 0.85);

  const auto* runtime_metric =
      FindHistogramMetric(families, "starpu_task_runtime_ms", {});
  ASSERT_NE(runtime_metric, nullptr);
  EXPECT_EQ(runtime_metric->histogram.sample_count, 1);
  EXPECT_DOUBLE_EQ(runtime_metric->histogram.sample_sum, 5.5);
}

TEST(Metrics, SetQueueFillAndStarpuGauges)
{
  ASSERT_TRUE(init_metrics(0));
  struct MetricsGuard {
    ~MetricsGuard() { shutdown_metrics(); }
  } guard;

  // trigger zero-capacity guard; gauge should remain at default (0)
  set_queue_fill_ratio(4, 0);
  const auto metrics = get_metrics();
  ASSERT_NE(metrics, nullptr);
  const auto families = metrics->registry()->Collect();
  const auto ratio_zero =
      FindGaugeValue(families, "inference_queue_fill_ratio", {});
  ASSERT_TRUE(ratio_zero.has_value());
  EXPECT_DOUBLE_EQ(*ratio_zero, 0.0);

  set_queue_fill_ratio(2, 4);
  const auto ratio_half = FindGaugeValue(
      metrics->registry()->Collect(), "inference_queue_fill_ratio", {});
  ASSERT_TRUE(ratio_half.has_value());
  EXPECT_DOUBLE_EQ(*ratio_half, 0.5);

  set_starpu_worker_busy_ratio(0.4);
  const auto busy_value = FindGaugeValue(
      metrics->registry()->Collect(), "starpu_worker_busy_ratio", {});
  ASSERT_TRUE(busy_value.has_value());
  EXPECT_DOUBLE_EQ(*busy_value, 0.4);

  set_starpu_prepared_queue_depth(7);
  const auto pending_depth = FindGaugeValue(
      metrics->registry()->Collect(), "starpu_prepared_queue_depth", {});
  ASSERT_TRUE(pending_depth.has_value());
  EXPECT_DOUBLE_EQ(*pending_depth, 7.0);

  set_batch_pending_jobs(13);
  const auto pending_gauge = metrics->batch_pending_jobs_gauge();
  ASSERT_NE(pending_gauge, nullptr);
  EXPECT_DOUBLE_EQ(pending_gauge->Value(), 13.0);
}

TEST(Metrics, InitializeThrowsWhenExposerRegisterFails)
{
  class ThrowingHandle : public MetricsRegistry::ExposerHandle {
   public:
    void RegisterCollectable(
        const std::shared_ptr<prometheus::Collectable>&) override
    {
      throw std::runtime_error("register failure");
    }

    void RemoveCollectable(
        const std::shared_ptr<prometheus::Collectable>&) override
    {
    }
  };

  EXPECT_THROW(
      MetricsRegistry registry(
          0, [] { return std::vector<MetricsRegistry::GpuSample>{}; },
          [] { return std::optional<double>{}; }, false,
          std::make_unique<ThrowingHandle>()),
      std::runtime_error);
}

TEST(Metrics, WorkersMetricsViaGlobalWrappers)
{
  ASSERT_TRUE(init_metrics(0));
  struct MetricsGuard {
    ~MetricsGuard() { shutdown_metrics(); }
  } guard;

  const int worker_id = 5;
  const int device_id = 1;
  const std::string worker_type = "cpu";

  observe_compute_latency_by_worker(worker_id, device_id, worker_type, 8.5);
  observe_task_runtime_by_worker(worker_id, device_id, worker_type, 3.75);
  set_worker_inflight_gauge(worker_id, device_id, worker_type, 4);
  observe_io_copy_latency("d2h", worker_id, device_id, worker_type, 2.25);
  increment_transfer_bytes("d2h", worker_id, device_id, worker_type, 128);

  const auto metrics = get_metrics();
  ASSERT_NE(metrics, nullptr);
  const auto families = metrics->registry()->Collect();

  const auto* compute_metric = FindHistogramMetric(
      families, "inference_compute_latency_ms_by_worker",
      {{"worker_id", std::to_string(worker_id)},
       {"device", std::to_string(device_id)},
       {"worker_type", worker_type}});
  ASSERT_NE(compute_metric, nullptr);
  EXPECT_EQ(compute_metric->histogram.sample_count, 1);
  EXPECT_DOUBLE_EQ(compute_metric->histogram.sample_sum, 8.5);

  const auto* runtime_metric = FindHistogramMetric(
      families, "starpu_task_runtime_ms_by_worker",
      {{"worker_id", std::to_string(worker_id)},
       {"device", std::to_string(device_id)},
       {"worker_type", worker_type}});
  ASSERT_NE(runtime_metric, nullptr);
  EXPECT_EQ(runtime_metric->histogram.sample_count, 1);
  EXPECT_DOUBLE_EQ(runtime_metric->histogram.sample_sum, 3.75);

  const auto inflight_value = FindGaugeValue(
      families, "starpu_worker_inflight_tasks",
      {{"worker_id", std::to_string(worker_id)},
       {"device", std::to_string(device_id)},
       {"worker_type", worker_type}});
  ASSERT_TRUE(inflight_value.has_value());
  EXPECT_DOUBLE_EQ(*inflight_value, 4.0);

  const auto* io_metric = FindHistogramMetric(
      families, "inference_io_copy_ms",
      {{"direction", "d2h"},
       {"worker_id", std::to_string(worker_id)},
       {"device", std::to_string(device_id)},
       {"worker_type", worker_type}});
  ASSERT_NE(io_metric, nullptr);
  EXPECT_EQ(io_metric->histogram.sample_count, 1);
  EXPECT_DOUBLE_EQ(io_metric->histogram.sample_sum, 2.25);

  const auto transfer_value = FindCounterValue(
      families, "inference_transfer_bytes_total",
      {{"direction", "d2h"},
       {"worker_id", std::to_string(worker_id)},
       {"device", std::to_string(device_id)},
       {"worker_type", worker_type}});
  ASSERT_TRUE(transfer_value.has_value());
  EXPECT_DOUBLE_EQ(*transfer_value, 128.0);
}

TEST(MetricsRegistry, ClearStarpuTaskRuntimeFamilySkipsObservation)
{
  MetricsRegistry metrics(
      0, [] { return std::vector<MetricsRegistry::GpuSample>{}; },
      [] { return std::optional<double>{}; }, false);
  MetricsRegistry::TestAccessor::ClearStarpuTaskRuntimeByWorkerFamily(metrics);

  metrics.observe_task_runtime_by_worker(2, 1, "cpu", 2.5);

  const auto* runtime_metric = FindHistogramMetric(
      metrics.registry()->Collect(), "starpu_task_runtime_ms_by_worker",
      {{"worker_id", "2"}, {"device", "1"}, {"worker_type", "cpu"}});
  EXPECT_EQ(runtime_metric, nullptr);
}

TEST(MetricsRegistry, ClearInferenceComputeLatencyFamilySkipsObservation)
{
  MetricsRegistry metrics(
      0, [] { return std::vector<MetricsRegistry::GpuSample>{}; },
      [] { return std::optional<double>{}; }, false);
  MetricsRegistry::TestAccessor::ClearInferenceComputeLatencyByWorkerFamily(
      metrics);

  metrics.observe_compute_latency_by_worker(2, 1, "cpu", 5.1);

  const auto* compute_metric = FindHistogramMetric(
      metrics.registry()->Collect(), "inference_compute_latency_ms_by_worker",
      {{"worker_id", "2"}, {"device", "1"}, {"worker_type", "cpu"}});
  EXPECT_EQ(compute_metric, nullptr);
}

TEST(MetricsRegistry, ClearIoCopyLatencyFamilySkipsObservation)
{
  MetricsRegistry metrics(
      0, [] { return std::vector<MetricsRegistry::GpuSample>{}; },
      [] { return std::optional<double>{}; }, false);
  MetricsRegistry::TestAccessor::ClearIoCopyLatencyFamily(metrics);

  metrics.observe_io_copy_latency("h2d", 2, 1, "cpu", 3.0);

  const auto* io_metric = FindHistogramMetric(
      metrics.registry()->Collect(), "inference_io_copy_ms",
      {{"direction", "h2d"},
       {"worker_id", "2"},
       {"device", "1"},
       {"worker_type", "cpu"}});
  EXPECT_EQ(io_metric, nullptr);
}

TEST(MetricsRegistry, ClearTransferBytesFamilySkipsIncrement)
{
  MetricsRegistry metrics(
      0, [] { return std::vector<MetricsRegistry::GpuSample>{}; },
      [] { return std::optional<double>{}; }, false);
  MetricsRegistry::TestAccessor::ClearTransferBytesFamily(metrics);

  metrics.increment_transfer_bytes("h2d", 2, 1, "cpu", 256);

  const auto value = FindCounterValue(
      metrics.registry()->Collect(), "inference_transfer_bytes_total",
      {{"direction", "h2d"},
       {"worker_id", "2"},
       {"device", "1"},
       {"worker_type", "cpu"}});
  EXPECT_FALSE(value.has_value());
}

TEST(MetricsDetail, ReadProcessOpenFdsHandlesMissingDirectory)
{
  struct PathGuard {
    PathGuard()
    {
      monitoring::detail::set_process_fd_path_for_test("/does/not/exist");
    }

    ~PathGuard() { monitoring::detail::reset_process_fd_path_for_test(); }
  } guard;

  const auto value = monitoring::detail::read_process_open_fds();
  EXPECT_FALSE(value.has_value());
}

TEST(MetricsDetail, ReadProcessOpenFdsCatchesDirectoryIteratorExceptions)
{
  struct FactoryGuard {
    FactoryGuard()
    {
      monitoring::detail::set_process_fd_directory_iterator_for_test(
          [](const std::filesystem::path&)
              -> std::filesystem::directory_iterator {
            throw std::runtime_error("iter failure");
          });
    }

    ~FactoryGuard()
    {
      monitoring::detail::reset_process_fd_directory_iterator_for_test();
    }
  } guard;

  const auto value = monitoring::detail::read_process_open_fds();
  EXPECT_FALSE(value.has_value());
}

TEST(MetricsDetail, ReadProcessRssBytesHandlesMissingStatm)
{
  const auto missing_path =
      std::filesystem::temp_directory_path() / "starpu_metrics_missing_statm";
  StatmPathGuard guard(missing_path);

  const auto value = monitoring::detail::read_process_rss_bytes();
  EXPECT_FALSE(value.has_value());
}

TEST(MetricsDetail, ReadProcessRssBytesHandlesZeroResident)
{
  TemporaryStatmFile statm("256 0");
  StatmPathGuard guard(statm.path());

  const auto value = monitoring::detail::read_process_rss_bytes();
  EXPECT_FALSE(value.has_value());
}

TEST(MetricsDetail, ReadProcessRssBytesHandlesNonPositivePageSize)
{
  TemporaryStatmFile statm("256 64");
  StatmPathGuard guard(statm.path());
  PageSizeProviderGuard page_guard([] { return 0L; });

  const auto value = monitoring::detail::read_process_rss_bytes();
  EXPECT_FALSE(value.has_value());
}

TEST(MetricsDetail, CpuUsagePercentClampsAboveOneHundred)
{
  const auto prev = monitoring::detail::CpuTotals{
      /*user=*/10,
      /*nice=*/0,
      /*system=*/0,
      /*idle=*/20,
      /*iowait=*/30,
      /*irq=*/0,
      /*softirq=*/0,
      /*steal=*/0};
  const auto curr = monitoring::detail::CpuTotals{
      /*user=*/20,
      /*nice=*/0,
      /*system=*/0,
      /*idle=*/10,
      /*iowait=*/10,
      /*irq=*/10,
      /*softirq=*/10,
      /*steal=*/10};

  const auto usage = monitoring::detail::cpu_usage_percent(prev, curr);
  EXPECT_DOUBLE_EQ(usage, 100.0);
}

TEST(MetricsRegistry, RunSamplingSkipsCpuUsageWhenProviderMissing)
{
  MetricsRegistry metrics(
      0, [] { return std::vector<MetricsRegistry::GpuSample>{}; },
      [] { return std::optional<double>{}; }, false);
  metrics.system_cpu_usage_percent()->Set(7.0);
  MetricsRegistry::TestAccessor::ClearCpuUsageProvider(metrics);

  metrics.run_sampling_request_nb();

  const auto value = FindGaugeValue(
      metrics.registry()->Collect(), "system_cpu_usage_percent", {});
  ASSERT_TRUE(value.has_value());
  EXPECT_DOUBLE_EQ(*value, 7.0);
}

TEST(Metrics, IncrementInferenceCompletedUpdatesCounter)
{
  ASSERT_TRUE(init_metrics(0));
  struct MetricsGuard {
    ~MetricsGuard() { shutdown_metrics(); }
  } guard;

  const std::string model_name = "global-complete";
  increment_inference_completed(model_name, 3);

  const auto metrics = get_metrics();
  ASSERT_NE(metrics, nullptr);
  const auto families = metrics->registry()->Collect();
  const auto count = FindCounterValue(
      families, "inference_completed_total", {{"model", model_name}});
  ASSERT_TRUE(count.has_value());
  EXPECT_DOUBLE_EQ(*count, 3.0);
}

TEST(MetricsRegistry, IncrementTransferBytesSkipsZero)
{
  MetricsRegistry metrics(
      0, [] { return std::vector<MetricsRegistry::GpuSample>{}; },
      [] { return std::optional<double>{}; }, false);

  metrics.increment_transfer_bytes("h2d", 1, 1, "cpu", 0);

  const auto value = FindCounterValue(
      metrics.registry()->Collect(), "inference_transfer_bytes_total",
      {{"direction", "h2d"},
       {"worker_id", "1"},
       {"device", "1"},
       {"worker_type", "cpu"}});
  EXPECT_FALSE(value.has_value());
}

TEST(MetricsRegistry, ObserveIoCopyLatencySkipsNegativeDuration)
{
  MetricsRegistry metrics(
      0, [] { return std::vector<MetricsRegistry::GpuSample>{}; },
      [] { return std::optional<double>{}; }, false);

  metrics.observe_io_copy_latency("h2d", 2, 3, "cpu", -0.1);

  const auto* metric = FindHistogramMetric(
      metrics.registry()->Collect(), "inference_io_copy_ms",
      {{"direction", "h2d"},
       {"worker_id", "2"},
       {"device", "3"},
       {"worker_type", "cpu"}});
  EXPECT_EQ(metric, nullptr);
}

TEST(MetricsRegistry, SetWorkerInflightGaugeSkipsWhenFamilyMissing)
{
  MetricsRegistry metrics(
      0, [] { return std::vector<MetricsRegistry::GpuSample>{}; },
      [] { return std::optional<double>{}; }, false);

  MetricsRegistry::TestAccessor::ClearStarpuWorkerInflightFamily(metrics);
  metrics.set_worker_inflight_gauge(3, 2, "cpu", 5);

  const auto value = FindGaugeValue(
      metrics.registry()->Collect(), "starpu_worker_inflight_tasks",
      {{"worker_id", "3"}, {"device", "2"}, {"worker_type", "cpu"}});
  EXPECT_FALSE(value.has_value());
}

TEST(MetricsRegistry, ObserveTaskRuntimeByWorkerSkipsNegativeLatency)
{
  MetricsRegistry metrics(
      0, [] { return std::vector<MetricsRegistry::GpuSample>{}; },
      [] { return std::optional<double>{}; }, false);

  metrics.observe_task_runtime_by_worker(5, 6, "gpu", -1.0);

  const auto* metric = FindHistogramMetric(
      metrics.registry()->Collect(), "starpu_task_runtime_ms_by_worker",
      {{"worker_id", "5"}, {"device", "6"}, {"worker_type", "gpu"}});
  EXPECT_EQ(metric, nullptr);
}

TEST(MetricsRegistry, ObserveComputeLatencyByWorkerSkipsNegativeLatency)
{
  MetricsRegistry metrics(
      0, [] { return std::vector<MetricsRegistry::GpuSample>{}; },
      [] { return std::optional<double>{}; }, false);

  metrics.observe_compute_latency_by_worker(5, 6, "gpu", -1.0);

  const auto* metric = FindHistogramMetric(
      metrics.registry()->Collect(), "inference_compute_latency_ms_by_worker",
      {{"worker_id", "5"}, {"device", "6"}, {"worker_type", "gpu"}});
  EXPECT_EQ(metric, nullptr);
}

TEST(MetricsRegistry, SetModelLoadedFlagSkipsWhenFamilyMissing)
{
  MetricsRegistry metrics(
      0, [] { return std::vector<MetricsRegistry::GpuSample>{}; },
      [] { return std::optional<double>{}; }, false);

  MetricsRegistry::TestAccessor::ClearModelsLoadedFamily(metrics);
  metrics.set_model_loaded_flag(
      MetricsRegistry::ModelLabel{"model-x"},
      MetricsRegistry::DeviceLabel{"dev-x"}, true);

  const auto value = FindGaugeValue(
      metrics.registry()->Collect(), "models_loaded",
      {{"model", "model-x"}, {"device", "dev-x"}});
  EXPECT_FALSE(value.has_value());
}

TEST(MetricsRegistry, IncrementModelLoadFailureCounterSkipsWhenFamilyMissing)
{
  MetricsRegistry metrics(
      0, [] { return std::vector<MetricsRegistry::GpuSample>{}; },
      [] { return std::optional<double>{}; }, false);

  MetricsRegistry::TestAccessor::ClearModelLoadFailuresFamily(metrics);
  metrics.increment_model_load_failure_counter("model-y");

  const auto value = FindCounterValue(
      metrics.registry()->Collect(), "model_load_failures_total",
      {{"model", "model-y"}});
  EXPECT_FALSE(value.has_value());
}

TEST(MetricsRegistry, IncrementFailureCounterSkipsWhenFamilyMissing)
{
  MetricsRegistry metrics(
      0, [] { return std::vector<MetricsRegistry::GpuSample>{}; },
      [] { return std::optional<double>{}; }, false);

  MetricsRegistry::TestAccessor::ClearInferenceFailuresFamily(metrics);
  metrics.increment_failure_counter(
      MetricsRegistry::FailureStageLabel{"stage"},
      MetricsRegistry::FailureReasonLabel{"reason"},
      MetricsRegistry::ModelLabel{"model-z"}, 2);

  const auto value = FindCounterValue(
      metrics.registry()->Collect(), "inference_failures_total",
      {{"stage", "stage"}, {"reason", "reason"}, {"model", "model-z"}});
  EXPECT_FALSE(value.has_value());
}

TEST(MetricsRegistry, IncrementCompletedCounterSkipsWhenFamilyMissing)
{
  MetricsRegistry metrics(
      0, [] { return std::vector<MetricsRegistry::GpuSample>{}; },
      [] { return std::optional<double>{}; }, false);

  MetricsRegistry::TestAccessor::ClearInferenceCompletedFamily(metrics);
  metrics.increment_completed_counter("model-w", 4);

  const auto value = FindCounterValue(
      metrics.registry()->Collect(), "inference_completed_total",
      {{"model", "model-w"}});
  EXPECT_FALSE(value.has_value());
}

TEST(MetricsRegistry, IncrementStatusCounterSkipsWhenFamilyMissing)
{
  MetricsRegistry metrics(
      0, [] { return std::vector<MetricsRegistry::GpuSample>{}; },
      [] { return std::optional<double>{}; }, false);

  MetricsRegistry::TestAccessor::ClearRequestsByStatusFamily(metrics);
  metrics.increment_status_counter(
      MetricsRegistry::StatusCodeLabel{"5"},
      MetricsRegistry::ModelLabel{"model-status"});

  const auto value = FindCounterValue(
      metrics.registry()->Collect(), "requests_by_status_total",
      {{"code", "5"}, {"model", "model-status"}});
  EXPECT_FALSE(value.has_value());
}
