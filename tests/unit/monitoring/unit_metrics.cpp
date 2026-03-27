#include <arpa/inet.h>
#include <grpcpp/support/status_code_enum.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <prometheus/client_metric.h>
#include <prometheus/metric_family.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
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
#include "support/monitoring/metrics_test_api.hpp"

using namespace starpu_server;

const prometheus::Histogram::BucketBoundaries kInferenceLatencyMsBuckets{
    1, 5, 10, 25, 50, 100, 250, 500, 1000};
const prometheus::Histogram::BucketBoundaries kBatchSizeBuckets{
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
const prometheus::Histogram::BucketBoundaries kBatchEfficiencyBuckets{
    0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0, 8.0};
const prometheus::Histogram::BucketBoundaries kModelLoadDurationMsBuckets{
    10, 50, 100, 200, 500, 1000, 2000, 5000, 10000};
const prometheus::Histogram::BucketBoundaries kTaskRuntimeMsBuckets{
    1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 5000};

#include "monitoring/metrics_registration.hpp"

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

class RequestStopJoinGuard {
 public:
  explicit RequestStopJoinGuard(bool skip_join_in_request_stop)
  {
    monitoring::detail::set_metrics_request_stop_skip_join_for_test(
        skip_join_in_request_stop);
  }

  ~RequestStopJoinGuard()
  {
    monitoring::detail::set_metrics_request_stop_skip_join_for_test(false);
  }
};

void
AssertMetricsInitialized(const std::shared_ptr<MetricsRegistry>& metrics)
{
  ASSERT_NE(metrics, nullptr);
  ASSERT_NE(metrics->registry(), nullptr);
  auto& counters = metrics->counters();
  auto& gauges = metrics->gauges();
  auto& histograms = metrics->histograms();
  auto& families = metrics->families();
  ASSERT_NE(counters.requests_total, nullptr);
  ASSERT_NE(counters.requests_rejected_total, nullptr);
  ASSERT_NE(histograms.inference_latency, nullptr);
  ASSERT_NE(gauges.queue_size, nullptr);
  ASSERT_NE(gauges.inflight_tasks, nullptr);
  ASSERT_NE(gauges.max_inflight_tasks, nullptr);
  ASSERT_NE(gauges.starpu_worker_busy_ratio, nullptr);
  ASSERT_NE(gauges.starpu_prepared_queue_depth, nullptr);
  ASSERT_NE(gauges.batch_pending_jobs, nullptr);
  ASSERT_NE(histograms.batch_efficiency, nullptr);
  ASSERT_NE(histograms.starpu_task_runtime, nullptr);
  ASSERT_NE(families.inference_compute_latency_by_worker, nullptr);
  ASSERT_NE(families.starpu_task_runtime_by_worker, nullptr);
  ASSERT_NE(families.starpu_worker_inflight, nullptr);
  ASSERT_NE(families.io_copy_latency, nullptr);
  ASSERT_NE(families.transfer_bytes, nullptr);
  ASSERT_NE(gauges.server_health_state, nullptr);
  ASSERT_NE(gauges.inference_throughput, nullptr);
  ASSERT_NE(gauges.process_resident_memory_bytes, nullptr);
  ASSERT_NE(gauges.process_open_fds, nullptr);
  ASSERT_NE(gauges.queue_fill_ratio, nullptr);
  ASSERT_NE(gauges.queue_capacity, nullptr);
  ASSERT_NE(histograms.queue_latency, nullptr);
  ASSERT_NE(histograms.batch_collect_latency, nullptr);
  ASSERT_NE(histograms.submit_latency, nullptr);
  ASSERT_NE(histograms.scheduling_latency, nullptr);
  ASSERT_NE(histograms.codelet_latency, nullptr);
  ASSERT_NE(histograms.inference_compute_latency, nullptr);
  ASSERT_NE(histograms.callback_latency, nullptr);
  ASSERT_NE(histograms.preprocess_latency, nullptr);
  ASSERT_NE(histograms.postprocess_latency, nullptr);
  ASSERT_NE(histograms.batch_size, nullptr);
  ASSERT_NE(histograms.logical_batch_size, nullptr);
  ASSERT_NE(families.requests_by_status, nullptr);
  ASSERT_NE(families.requests_received, nullptr);
  ASSERT_NE(families.inference_completed, nullptr);
  ASSERT_NE(families.inference_failures, nullptr);
  ASSERT_NE(families.model_load_failures, nullptr);
  ASSERT_NE(families.models_loaded, nullptr);
  ASSERT_NE(families.gpu_model_replication_policy_info, nullptr);
  ASSERT_NE(families.gpu_model_replicas_total, nullptr);
  ASSERT_NE(families.starpu_cuda_worker_info, nullptr);
  ASSERT_NE(families.gpu_temperature, nullptr);
  ASSERT_NE(families.gpu_power, nullptr);
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

TEST(MetricsRegistration, RegisterCounterMetricBuildsCounterFamily)
{
  prometheus::Registry registry;

  auto* counter = detail::register_counter_metric(
      registry, "unit_test_counter_total",
      "Counter used by unit test for registration helper");
  ASSERT_NE(counter, nullptr);
  counter->Increment();

  const auto families = registry.Collect();
  const auto value =
      FindCounterValue(families, "unit_test_counter_total", /*labels=*/{});
  ASSERT_TRUE(value.has_value());
  EXPECT_DOUBLE_EQ(*value, 1.0);
}

TEST(Metrics, InitializesPointersAndRegistry)
{
  ASSERT_TRUE(init_metrics(0));

  auto metrics = get_metrics();
  AssertMetricsInitialized(metrics);

  const auto families = metrics->registry()->Collect();
  EXPECT_TRUE(HasMetric(families, "requests_total"));
  EXPECT_TRUE(HasMetric(families, "requests_rejected_total"));
  EXPECT_TRUE(HasMetric(families, "requests_by_status_total"));
  EXPECT_TRUE(HasMetric(families, "requests_received_total"));
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

  ::testing::internal::CaptureStderr();
  {
    auto handle = std::make_unique<ThrowingRemoveHandle>();
    MetricsRegistry registry(
        0, [] { return std::vector<MetricsRegistry::GpuSample>{}; },
        [] { return std::optional<double>{}; }, false, std::move(handle));
  }
  const std::string log = ::testing::internal::GetCapturedStderr();
  EXPECT_NE(
      log.find("Failed to remove metrics registry collectable"),
      std::string::npos);
}

TEST(Metrics, MetricsDestructionJoinsSamplerThreadWhenStillJoinable)
{
  RequestStopJoinGuard guard{/*skip_join_in_request_stop=*/true};

  EXPECT_NO_THROW({
    MetricsRegistry registry(
        0, [] { return std::vector<MetricsRegistry::GpuSample>{}; },
        [] { return std::optional<double>{}; },
        /*start_sampler_thread=*/true);
  });
}

TEST(Metrics, AccessorsReturnAllocatedFamiliesAndGauges)
{
  ASSERT_TRUE(init_metrics(0));

  auto metrics = get_metrics();
  AssertMetricsInitialized(metrics);

  EXPECT_NE(metrics->gauges().system_cpu_usage_percent, nullptr);
  EXPECT_NE(metrics->families().gpu_utilization, nullptr);
  EXPECT_NE(metrics->families().gpu_memory_used_bytes, nullptr);
  EXPECT_NE(metrics->families().gpu_memory_total_bytes, nullptr);

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

TEST(Metrics, IncrementRequestsReceivedUsesModelLabel)
{
  ASSERT_TRUE(init_metrics(0));
  struct MetricsGuard {
    ~MetricsGuard() { shutdown_metrics(); }
  } guard;

  increment_requests_received("model-a");
  increment_requests_received("model-a");
  increment_requests_received("model-b");

  const auto metrics = get_metrics();
  ASSERT_NE(metrics, nullptr);
  const auto families = metrics->registry()->Collect();

  const auto model_a = FindCounterValue(
      families, "requests_received_total", {{"model", "model-a"}});
  ASSERT_TRUE(model_a.has_value());
  EXPECT_DOUBLE_EQ(*model_a, 2.0);

  const auto model_b = FindCounterValue(
      families, "requests_received_total", {{"model", "model-b"}});
  ASSERT_TRUE(model_b.has_value());
  EXPECT_DOUBLE_EQ(*model_b, 1.0);
}

TEST(MetricsRegistry, EscapesReservedStatusLabels)
{
  MetricsRegistry metrics(
      0, [] { return std::vector<MetricsRegistry::GpuSample>{}; },
      [] { return std::optional<double>{}; },
      /*start_sampler_thread=*/false);

  metrics.increment_status_counter(
      MetricsRegistry::StatusCodeLabel{"__overflow__"},
      MetricsRegistry::ModelLabel{"__label__model"});

  const auto families = metrics.registry()->Collect();
  const auto value = FindCounterValue(
      families, "requests_by_status_total",
      {{"code", "__label____overflow__"},
       {"model", "__label____label__model"}});
  ASSERT_TRUE(value.has_value());
  EXPECT_DOUBLE_EQ(*value, 1.0);
}

TEST(MetricsRegistry, StatusLabelOverflowUsesReservedBucket)
{
  MetricsRegistry metrics(
      0, [] { return std::vector<MetricsRegistry::GpuSample>{}; },
      [] { return std::optional<double>{}; },
      /*start_sampler_thread=*/false);

  constexpr std::size_t kMaxLabelSeriesForTest = 10000;
  constexpr std::size_t kOverflowSamples = 2;
  for (std::size_t i = 0; i < kMaxLabelSeriesForTest + kOverflowSamples; ++i) {
    const std::string model_label = "model-" + std::to_string(i);
    metrics.increment_status_counter(
        MetricsRegistry::StatusCodeLabel{"OK"},
        MetricsRegistry::ModelLabel{model_label});
  }

  const auto families = metrics.registry()->Collect();
  const auto overflow_value = FindCounterValue(
      families, "requests_by_status_total",
      {{"code", "__overflow__"}, {"model", "__overflow__"}});
  ASSERT_TRUE(overflow_value.has_value());
  EXPECT_DOUBLE_EQ(*overflow_value, static_cast<double>(kOverflowSamples));

  const std::string overflowed_label =
      "model-" + std::to_string(kMaxLabelSeriesForTest);
  const auto unexpected_value = FindCounterValue(
      families, "requests_by_status_total",
      {{"code", "OK"}, {"model", overflowed_label}});
  EXPECT_FALSE(unexpected_value.has_value());
}

TEST(MetricsRegistry, FailureKeyOverflowSetsFlag)
{
  EXPECT_TRUE(starpu_server::testing::MetricsRegistryTestAccessor::
                  FailureKeyOverflowIsEmpty());
}

TEST(MetricsRegistry, FailureKeyEqualityComparesAllFields)
{
  EXPECT_TRUE(
      starpu_server::testing::MetricsRegistryTestAccessor::FailureKeyEquals(
          "stage-a", "reason-a", "model-a", /*overflow_lhs=*/false, "stage-a",
          "reason-a", "model-a", /*overflow_rhs=*/false));

  EXPECT_FALSE(
      starpu_server::testing::MetricsRegistryTestAccessor::FailureKeyEquals(
          "stage-a", "reason-a", "model-a", /*overflow_lhs=*/false, "stage-b",
          "reason-a", "model-a", /*overflow_rhs=*/false));
  EXPECT_FALSE(
      starpu_server::testing::MetricsRegistryTestAccessor::FailureKeyEquals(
          "stage-a", "reason-a", "model-a", /*overflow_lhs=*/false, "stage-a",
          "reason-b", "model-a", /*overflow_rhs=*/false));
  EXPECT_FALSE(
      starpu_server::testing::MetricsRegistryTestAccessor::FailureKeyEquals(
          "stage-a", "reason-a", "model-a", /*overflow_lhs=*/false, "stage-a",
          "reason-a", "model-b", /*overflow_rhs=*/false));
  EXPECT_FALSE(
      starpu_server::testing::MetricsRegistryTestAccessor::FailureKeyEquals(
          "stage-a", "reason-a", "model-a", /*overflow_lhs=*/false, "stage-a",
          "reason-a", "model-a", /*overflow_rhs=*/true));
}

TEST(MetricsRegistry, ModelKeyOverflowSetsFlag)
{
  EXPECT_TRUE(starpu_server::testing::MetricsRegistryTestAccessor::
                  ModelKeyOverflowIsEmpty());
}

TEST(MetricsRegistry, ModelKeyEqualityComparesAllFields)
{
  EXPECT_TRUE(
      starpu_server::testing::MetricsRegistryTestAccessor::ModelKeyEquals(
          "model-a", /*overflow_lhs=*/false, "model-a",
          /*overflow_rhs=*/false));

  EXPECT_FALSE(
      starpu_server::testing::MetricsRegistryTestAccessor::ModelKeyEquals(
          "model-a", /*overflow_lhs=*/false, "model-b",
          /*overflow_rhs=*/false));
  EXPECT_FALSE(
      starpu_server::testing::MetricsRegistryTestAccessor::ModelKeyEquals(
          "model-a", /*overflow_lhs=*/false, "model-a", /*overflow_rhs=*/true));
}

TEST(MetricsRegistry, ModelDeviceKeyOverflowSetsFlag)
{
  EXPECT_TRUE(starpu_server::testing::MetricsRegistryTestAccessor::
                  ModelDeviceKeyOverflowIsEmpty());
}

TEST(MetricsRegistry, ModelDeviceKeyEqualityComparesAllFields)
{
  EXPECT_TRUE(
      starpu_server::testing::MetricsRegistryTestAccessor::ModelDeviceKeyEquals(
          "model-a", "gpu0", /*overflow_lhs=*/false, "model-a", "gpu0",
          /*overflow_rhs=*/false));

  EXPECT_FALSE(
      starpu_server::testing::MetricsRegistryTestAccessor::ModelDeviceKeyEquals(
          "model-a", "gpu0", /*overflow_lhs=*/false, "model-b", "gpu0",
          /*overflow_rhs=*/false));
  EXPECT_FALSE(
      starpu_server::testing::MetricsRegistryTestAccessor::ModelDeviceKeyEquals(
          "model-a", "gpu0", /*overflow_lhs=*/false, "model-a", "gpu1",
          /*overflow_rhs=*/false));
  EXPECT_FALSE(
      starpu_server::testing::MetricsRegistryTestAccessor::ModelDeviceKeyEquals(
          "model-a", "gpu0", /*overflow_lhs=*/false, "model-a", "gpu0",
          /*overflow_rhs=*/true));
}

TEST(MetricsRegistry, ModelPolicyKeyOverflowSetsFlag)
{
  EXPECT_TRUE(starpu_server::testing::MetricsRegistryTestAccessor::
                  ModelPolicyKeyOverflowIsEmpty());
}

TEST(MetricsRegistry, ModelPolicyKeyEqualityComparesAllFields)
{
  EXPECT_TRUE(
      starpu_server::testing::MetricsRegistryTestAccessor::ModelPolicyKeyEquals(
          "model-a", "per_worker", /*overflow_lhs=*/false, "model-a",
          "per_worker", /*overflow_rhs=*/false));

  EXPECT_FALSE(
      starpu_server::testing::MetricsRegistryTestAccessor::ModelPolicyKeyEquals(
          "model-a", "per_worker", /*overflow_lhs=*/false, "model-b",
          "per_worker", /*overflow_rhs=*/false));
  EXPECT_FALSE(
      starpu_server::testing::MetricsRegistryTestAccessor::ModelPolicyKeyEquals(
          "model-a", "per_worker", /*overflow_lhs=*/false, "model-a", "shared",
          /*overflow_rhs=*/false));
  EXPECT_FALSE(
      starpu_server::testing::MetricsRegistryTestAccessor::ModelPolicyKeyEquals(
          "model-a", "per_worker", /*overflow_lhs=*/false, "model-a",
          "per_worker", /*overflow_rhs=*/true));
}

TEST(MetricsRegistry, IoKeyOverflowSetsFlag)
{
  EXPECT_TRUE(starpu_server::testing::MetricsRegistryTestAccessor::
                  IoKeyOverflowIsEmpty());
}

TEST(MetricsRegistry, IoKeyEqualityComparesAllFields)
{
  EXPECT_TRUE(starpu_server::testing::MetricsRegistryTestAccessor::IoKeyEquals(
      "h2d", 1, 2, "cpu", /*overflow_lhs=*/false, "h2d", 1, 2, "cpu",
      /*overflow_rhs=*/false));

  EXPECT_FALSE(starpu_server::testing::MetricsRegistryTestAccessor::IoKeyEquals(
      "h2d", 1, 2, "cpu", /*overflow_lhs=*/false, "d2h", 1, 2, "cpu",
      /*overflow_rhs=*/false));
  EXPECT_FALSE(starpu_server::testing::MetricsRegistryTestAccessor::IoKeyEquals(
      "h2d", 1, 2, "cpu", /*overflow_lhs=*/false, "h2d", 3, 2, "cpu",
      /*overflow_rhs=*/false));
  EXPECT_FALSE(starpu_server::testing::MetricsRegistryTestAccessor::IoKeyEquals(
      "h2d", 1, 2, "cpu", /*overflow_lhs=*/false, "h2d", 1, 4, "cpu",
      /*overflow_rhs=*/false));
  EXPECT_FALSE(starpu_server::testing::MetricsRegistryTestAccessor::IoKeyEquals(
      "h2d", 1, 2, "cpu", /*overflow_lhs=*/false, "h2d", 1, 2, "gpu",
      /*overflow_rhs=*/false));
  EXPECT_FALSE(starpu_server::testing::MetricsRegistryTestAccessor::IoKeyEquals(
      "h2d", 1, 2, "cpu", /*overflow_lhs=*/false, "h2d", 1, 2, "cpu",
      /*overflow_rhs=*/true));
}

TEST(MetricsRegistry, WorkerKeyOverflowSetsFlag)
{
  EXPECT_TRUE(starpu_server::testing::MetricsRegistryTestAccessor::
                  WorkerKeyOverflowIsEmpty());
}

TEST(MetricsRegistry, WorkerKeyEqualityComparesAllFields)
{
  EXPECT_TRUE(
      starpu_server::testing::MetricsRegistryTestAccessor::WorkerKeyEquals(
          1, 2, "cpu", /*overflow_lhs=*/false, 1, 2, "cpu",
          /*overflow_rhs=*/false));

  EXPECT_FALSE(
      starpu_server::testing::MetricsRegistryTestAccessor::WorkerKeyEquals(
          1, 2, "cpu", /*overflow_lhs=*/false, 3, 2, "cpu",
          /*overflow_rhs=*/false));
  EXPECT_FALSE(
      starpu_server::testing::MetricsRegistryTestAccessor::WorkerKeyEquals(
          1, 2, "cpu", /*overflow_lhs=*/false, 1, 4, "cpu",
          /*overflow_rhs=*/false));
  EXPECT_FALSE(
      starpu_server::testing::MetricsRegistryTestAccessor::WorkerKeyEquals(
          1, 2, "cpu", /*overflow_lhs=*/false, 1, 2, "gpu",
          /*overflow_rhs=*/false));
  EXPECT_FALSE(
      starpu_server::testing::MetricsRegistryTestAccessor::WorkerKeyEquals(
          1, 2, "cpu", /*overflow_lhs=*/false, 1, 2, "cpu",
          /*overflow_rhs=*/true));
}

TEST(Metrics, SetQueueFillRatioUpdatesGauge)
{
  ASSERT_TRUE(init_metrics(0));
  struct MetricsGuard {
    ~MetricsGuard() { shutdown_metrics(); }
  } guard;

  set_queue_capacity(10);
  set_queue_size(3);

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

TEST(Metrics, ObserveLatencyBreakdownUpdatesHistograms)
{
  ASSERT_TRUE(init_metrics(0));
  struct MetricsGuard {
    ~MetricsGuard() { shutdown_metrics(); }
  } guard;

  observe_latency_breakdown(LatencyBreakdownMetrics{
      1.0,
      2.0,
      3.0,
      4.0,
      5.0,
      6.0,
      7.0,
      8.0,
      9.0,
  });

  const auto metrics = get_metrics();
  ASSERT_NE(metrics, nullptr);
  const auto families = metrics->registry()->Collect();

  const auto* queue_metric =
      FindHistogramMetric(families, "inference_queue_latency_ms", {});
  ASSERT_NE(queue_metric, nullptr);
  EXPECT_EQ(queue_metric->histogram.sample_count, 1);
  EXPECT_DOUBLE_EQ(queue_metric->histogram.sample_sum, 1.0);

  const auto* batch_metric =
      FindHistogramMetric(families, "inference_batch_collect_ms", {});
  ASSERT_NE(batch_metric, nullptr);
  EXPECT_EQ(batch_metric->histogram.sample_count, 1);
  EXPECT_DOUBLE_EQ(batch_metric->histogram.sample_sum, 2.0);

  const auto* submit_metric =
      FindHistogramMetric(families, "inference_submit_latency_ms", {});
  ASSERT_NE(submit_metric, nullptr);
  EXPECT_EQ(submit_metric->histogram.sample_count, 1);
  EXPECT_DOUBLE_EQ(submit_metric->histogram.sample_sum, 3.0);

  const auto* scheduling_metric =
      FindHistogramMetric(families, "inference_scheduling_latency_ms", {});
  ASSERT_NE(scheduling_metric, nullptr);
  EXPECT_EQ(scheduling_metric->histogram.sample_count, 1);
  EXPECT_DOUBLE_EQ(scheduling_metric->histogram.sample_sum, 4.0);

  const auto* codelet_metric =
      FindHistogramMetric(families, "inference_codelet_latency_ms", {});
  ASSERT_NE(codelet_metric, nullptr);
  EXPECT_EQ(codelet_metric->histogram.sample_count, 1);
  EXPECT_DOUBLE_EQ(codelet_metric->histogram.sample_sum, 5.0);

  const auto* inference_metric =
      FindHistogramMetric(families, "inference_compute_latency_ms", {});
  ASSERT_NE(inference_metric, nullptr);
  EXPECT_EQ(inference_metric->histogram.sample_count, 1);
  EXPECT_DOUBLE_EQ(inference_metric->histogram.sample_sum, 6.0);

  const auto* callback_metric =
      FindHistogramMetric(families, "inference_callback_latency_ms", {});
  ASSERT_NE(callback_metric, nullptr);
  EXPECT_EQ(callback_metric->histogram.sample_count, 1);
  EXPECT_DOUBLE_EQ(callback_metric->histogram.sample_sum, 7.0);

  const auto* preprocess_metric =
      FindHistogramMetric(families, "inference_preprocess_latency_ms", {});
  ASSERT_NE(preprocess_metric, nullptr);
  EXPECT_EQ(preprocess_metric->histogram.sample_count, 1);
  EXPECT_DOUBLE_EQ(preprocess_metric->histogram.sample_sum, 8.0);

  const auto* postprocess_metric =
      FindHistogramMetric(families, "inference_postprocess_latency_ms", {});
  ASSERT_NE(postprocess_metric, nullptr);
  EXPECT_EQ(postprocess_metric->histogram.sample_count, 1);
  EXPECT_DOUBLE_EQ(postprocess_metric->histogram.sample_sum, 9.0);
}

TEST(Metrics, SetQueueFillAndStarpuGauges)
{
  ASSERT_TRUE(init_metrics(0));
  struct MetricsGuard {
    ~MetricsGuard() { shutdown_metrics(); }
  } guard;

  // trigger zero-capacity guard; gauge should remain at default (0)
  set_queue_capacity(0);
  set_queue_size(4);
  const auto metrics = get_metrics();
  ASSERT_NE(metrics, nullptr);
  const auto families = metrics->registry()->Collect();
  const auto ratio_zero =
      FindGaugeValue(families, "inference_queue_fill_ratio", {});
  ASSERT_TRUE(ratio_zero.has_value());
  EXPECT_DOUBLE_EQ(*ratio_zero, 0.0);

  set_queue_capacity(4);
  set_queue_size(2);
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
  const auto pending_gauge = metrics->gauges().batch_pending_jobs;
  ASSERT_NE(pending_gauge, nullptr);
  EXPECT_DOUBLE_EQ(pending_gauge->Value(), 13.0);
}

TEST(Metrics, SetQueueCapacityReturnsWhenQueueFillRatioGaugeMissing)
{
  ASSERT_TRUE(init_metrics(0));
  struct MetricsGuard {
    ~MetricsGuard() { shutdown_metrics(); }
  } guard;

  const auto metrics = get_metrics();
  ASSERT_NE(metrics, nullptr);
  starpu_server::testing::MetricsRegistryTestAccessor::ClearQueueFillRatioGauge(
      *metrics);

  set_queue_capacity(8);

  EXPECT_EQ(metrics->queue_capacity_value(), 8U);
}

TEST(Metrics, CongestionSettersNoOpWhenMetricsMissing)
{
  shutdown_metrics();
  EXPECT_EQ(get_metrics(), nullptr);

  set_congestion_flag(true);
  set_congestion_score(0.4);
  set_congestion_arrival_rate(3.0);
  set_congestion_completion_rate(4.0);
  set_congestion_rejection_rate(0.5);
  set_congestion_rho(0.75);
  set_congestion_fill_ewma(0.2);
  set_congestion_queue_growth_rate(1.1);
  set_congestion_queue_latency_p95(2.2);
  set_congestion_queue_latency_p99(3.3);
  set_congestion_e2e_latency_p95(4.4);
  set_congestion_e2e_latency_p99(5.5);

  EXPECT_EQ(get_metrics(), nullptr);
}

TEST(Metrics, CongestionSettersUpdateGauges)
{
  ASSERT_TRUE(init_metrics(0));
  struct MetricsGuard {
    ~MetricsGuard() { shutdown_metrics(); }
  } guard;

  set_congestion_flag(true);
  set_congestion_score(1.5);
  set_congestion_arrival_rate(-2.0);
  set_congestion_completion_rate(4.25);
  set_congestion_rejection_rate(-0.5);
  set_congestion_rho(std::nan(""));
  set_congestion_fill_ewma(1.2);
  set_congestion_queue_growth_rate(3.5);
  set_congestion_queue_latency_p95(-7.0);
  set_congestion_queue_latency_p99(9.25);
  set_congestion_e2e_latency_p95(-1.0);
  set_congestion_e2e_latency_p99(12.5);

  const auto metrics = get_metrics();
  ASSERT_NE(metrics, nullptr);
  const auto families = metrics->registry()->Collect();

  auto expect_gauge = [&](std::string_view name, double expected) {
    const auto value = FindGaugeValue(families, name, {});
    ASSERT_TRUE(value.has_value()) << "Missing gauge " << name;
    EXPECT_DOUBLE_EQ(*value, expected);
  };

  expect_gauge("inference_congestion_flag", 1.0);
  expect_gauge("inference_congestion_score", 1.0);
  expect_gauge("inference_lambda_rps", 0.0);
  expect_gauge("inference_mu_rps", 4.25);
  expect_gauge("inference_rejection_rate_rps", 0.0);
  expect_gauge("inference_rho_ewma", 0.0);
  expect_gauge("inference_queue_fill_ratio_ewma", 1.0);
  expect_gauge("inference_queue_growth_rate", 3.5);
  expect_gauge("inference_queue_latency_p95_ms", 0.0);
  expect_gauge("inference_queue_latency_p99_ms", 9.25);
  expect_gauge("inference_e2e_latency_p95_ms", 0.0);
  expect_gauge("inference_e2e_latency_p99_ms", 12.5);
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
  starpu_server::testing::MetricsRegistryTestAccessor::
      ClearStarpuTaskRuntimeByWorkerFamily(metrics);

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
  starpu_server::testing::MetricsRegistryTestAccessor::
      ClearInferenceComputeLatencyByWorkerFamily(metrics);

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
  starpu_server::testing::MetricsRegistryTestAccessor::ClearIoCopyLatencyFamily(
      metrics);

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
  starpu_server::testing::MetricsRegistryTestAccessor::ClearTransferBytesFamily(
      metrics);

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

TEST(MetricsDetail, ReadProcessRssBytesHandlesMalformedStatm)
{
  TemporaryStatmFile statm("256");
  StatmPathGuard guard(statm.path());

  const auto value = monitoring::detail::read_process_rss_bytes();
  EXPECT_FALSE(value.has_value());
}

TEST(MetricsDetail, ReadProcessRssBytesAllowsZeroResident)
{
  TemporaryStatmFile statm("256 0");
  StatmPathGuard guard(statm.path());

  const auto value = monitoring::detail::read_process_rss_bytes();
  ASSERT_TRUE(value.has_value());
  EXPECT_DOUBLE_EQ(*value, 0.0);
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

TEST(MetricsDetail, CpuUsagePercentReturnsZeroWhenTotalDeltaNonPositive)
{
  const auto prev = monitoring::detail::CpuTotals{
      /*user=*/100,
      /*nice=*/10,
      /*system=*/5,
      /*idle=*/200,
      /*iowait=*/20,
      /*irq=*/1,
      /*softirq=*/2,
      /*steal=*/3};
  const auto curr = monitoring::detail::CpuTotals{
      /*user=*/50,
      /*nice=*/5,
      /*system=*/2,
      /*idle=*/100,
      /*iowait=*/10,
      /*irq=*/1,
      /*softirq=*/1,
      /*steal=*/1};

  const auto usage = monitoring::detail::cpu_usage_percent(prev, curr);
  EXPECT_DOUBLE_EQ(usage, 0.0);
}

TEST(MetricsDetail, StatusCodeLabelMapsAdditionalStatuses)
{
  EXPECT_EQ(
      monitoring::detail::status_code_label_for_test(
          static_cast<int>(grpc::StatusCode::CANCELLED)),
      "CANCELLED");
  EXPECT_EQ(
      monitoring::detail::status_code_label_for_test(
          static_cast<int>(grpc::StatusCode::UNKNOWN)),
      "UNKNOWN");
  EXPECT_EQ(
      monitoring::detail::status_code_label_for_test(
          static_cast<int>(grpc::StatusCode::ALREADY_EXISTS)),
      "ALREADY_EXISTS");
  EXPECT_EQ(
      monitoring::detail::status_code_label_for_test(
          static_cast<int>(grpc::StatusCode::DATA_LOSS)),
      "DATA_LOSS");
}

TEST(MetricsDetail, ShouldLogSamplingErrorSkipsWhenThrottleNotElapsed)
{
  const auto now = std::chrono::duration_cast<std::chrono::seconds>(
                       std::chrono::steady_clock::now().time_since_epoch())
                       .count();
  const auto future = now + 3600;
  std::atomic<std::int64_t> last_log{future};

  const auto should_log =
      monitoring::detail::should_log_sampling_error_for_test(last_log);

  EXPECT_FALSE(should_log);
  EXPECT_EQ(last_log.load(std::memory_order_relaxed), future);
}

TEST(MetricsRegistry, RunSamplingMarksCpuUsageUnknownWhenProviderMissing)
{
  MetricsRegistry metrics(
      0, [] { return std::vector<MetricsRegistry::GpuSample>{}; },
      [] { return std::optional<double>{}; }, false);
  metrics.gauges().system_cpu_usage_percent->Set(7.0);
  starpu_server::testing::MetricsRegistryTestAccessor::ClearCpuUsageProvider(
      metrics);

  metrics.run_sampling_request_nb();

  const auto value = FindGaugeValue(
      metrics.registry()->Collect(), "system_cpu_usage_percent", {});
  ASSERT_TRUE(value.has_value());
  EXPECT_TRUE(std::isnan(*value));
}

TEST(MetricsRegistry, RunSamplingSkipsCpuProviderWhenGaugeMissing)
{
  auto gpu_provider = []() {
    return std::vector<MetricsRegistry::GpuSample>{};
  };
  bool cpu_called = false;
  auto cpu_provider = [&cpu_called]() {
    cpu_called = true;
    return std::optional<double>{50.0};
  };

  MetricsRegistry metrics(
      0, std::move(gpu_provider), std::move(cpu_provider),
      /*start_sampler_thread=*/false);

  starpu_server::testing::MetricsRegistryTestAccessor::ClearSystemCpuUsageGauge(
      metrics);
  EXPECT_EQ(metrics.gauges().system_cpu_usage_percent, nullptr);

  metrics.run_sampling_request_nb();

  EXPECT_FALSE(cpu_called);
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

TEST(Metrics, GlobalModelMetricWrappersRecordExpectedSeries)
{
  ASSERT_TRUE(init_metrics(0));
  struct MetricsGuard {
    ~MetricsGuard() { shutdown_metrics(); }
  } guard;

  const std::string model_name = "global-model";
  increment_inference_failure("submit", "timeout", model_name, 4);
  set_model_loaded(model_name, "gpu:1", true);
  set_gpu_model_replication_policy(model_name, "shared");
  set_gpu_model_replicas_total(model_name, 2);
  set_starpu_cuda_worker_info(5, 1, true);

  const auto metrics = get_metrics();
  ASSERT_NE(metrics, nullptr);
  const auto families = metrics->registry()->Collect();

  const auto failure_count = FindCounterValue(
      families, "inference_failures_total",
      {{"stage", "submit"}, {"reason", "timeout"}, {"model", model_name}});
  ASSERT_TRUE(failure_count.has_value());
  EXPECT_DOUBLE_EQ(*failure_count, 4.0);

  const auto model_loaded = FindGaugeValue(
      families, "models_loaded", {{"model", model_name}, {"device", "gpu:1"}});
  ASSERT_TRUE(model_loaded.has_value());
  EXPECT_DOUBLE_EQ(*model_loaded, 1.0);

  const auto replication_policy = FindGaugeValue(
      families, "gpu_model_replication_policy_info",
      {{"model", model_name}, {"policy", "shared"}});
  ASSERT_TRUE(replication_policy.has_value());
  EXPECT_DOUBLE_EQ(*replication_policy, 1.0);

  const auto replicas = FindGaugeValue(
      families, "gpu_model_replicas_total", {{"model", model_name}});
  ASSERT_TRUE(replicas.has_value());
  EXPECT_DOUBLE_EQ(*replicas, 2.0);

  const auto worker = FindGaugeValue(
      families, "starpu_cuda_worker_info",
      {{"device", "1"}, {"worker_id", "5"}});
  ASSERT_TRUE(worker.has_value());
  EXPECT_DOUBLE_EQ(*worker, 1.0);
}

TEST(Metrics, IncrementInferenceFailureWrapperRecordsCounter)
{
  ASSERT_TRUE(init_metrics(0));
  struct MetricsGuard {
    ~MetricsGuard() { shutdown_metrics(); }
  } guard;

  increment_inference_failure("dispatch", "retry", "wrapper-model", 2);

  const auto metrics = get_metrics();
  ASSERT_NE(metrics, nullptr);
  const auto value = FindCounterValue(
      metrics->registry()->Collect(), "inference_failures_total",
      {{"stage", "dispatch"}, {"reason", "retry"}, {"model", "wrapper-model"}});
  ASSERT_TRUE(value.has_value());
  EXPECT_DOUBLE_EQ(*value, 2.0);
}

TEST(Metrics, SetModelLoadedWrapperRecordsGauge)
{
  ASSERT_TRUE(init_metrics(0));
  struct MetricsGuard {
    ~MetricsGuard() { shutdown_metrics(); }
  } guard;

  set_model_loaded("wrapper-model", "gpu:2", true);

  const auto metrics = get_metrics();
  ASSERT_NE(metrics, nullptr);
  const auto value = FindGaugeValue(
      metrics->registry()->Collect(), "models_loaded",
      {{"model", "wrapper-model"}, {"device", "gpu:2"}});
  ASSERT_TRUE(value.has_value());
  EXPECT_DOUBLE_EQ(*value, 1.0);
}

TEST(MetricsRecorder, DefaultRecorderIsDisabledAndNoopsWithoutRegistry)
{
  const MetricsRecorder recorder;

  EXPECT_FALSE(recorder.enabled());
  EXPECT_EQ(recorder.registry(), nullptr);

  recorder.increment_requests_total();
  recorder.observe_inference_latency(1.0);
  recorder.set_queue_size(1);
  recorder.set_inflight_tasks(2);
  recorder.set_starpu_worker_busy_ratio(0.5);
  recorder.set_max_inflight_tasks(3);
  recorder.set_queue_capacity(4);
  recorder.set_server_health(true);
  recorder.set_starpu_prepared_queue_depth(5);
  recorder.set_batch_pending_jobs(6);
  recorder.set_congestion_flag(true);
  recorder.set_congestion_score(0.7);
  recorder.set_congestion_arrival_rate(8.0);
  recorder.set_congestion_completion_rate(9.0);
  recorder.set_congestion_rejection_rate(1.0);
  recorder.set_congestion_rho(0.8);
  recorder.set_congestion_fill_ewma(0.9);
  recorder.set_congestion_queue_growth_rate(1.1);
  recorder.set_congestion_queue_latency_p95(12.0);
  recorder.set_congestion_queue_latency_p99(13.0);
  recorder.set_congestion_e2e_latency_p95(14.0);
  recorder.set_congestion_e2e_latency_p99(15.0);
  recorder.increment_request_status(
      static_cast<int>(grpc::StatusCode::UNAVAILABLE), "null-model");
  recorder.increment_requests_received("null-model");
  recorder.increment_inference_completed("null-model", 2);
  recorder.increment_inference_failure("submit", "no-registry", "null-model");
  recorder.observe_batch_size(7);
  recorder.observe_logical_batch_size(8);
  recorder.observe_batch_efficiency(0.5);
  recorder.observe_latency_breakdown(
      LatencyBreakdownMetrics{1, 2, 3, 4, 5, 6, 7, 8, 9});
  recorder.observe_starpu_task_runtime(16.0);
  recorder.observe_model_load_duration(17.0);
  recorder.set_model_loaded("null-model", "gpu:0", true);
  recorder.set_gpu_model_replication_policy("null-model", "per_worker");
  recorder.set_gpu_model_replicas_total("null-model", 4);
  recorder.set_starpu_cuda_worker_info(1, 0, true);
  recorder.increment_model_load_failure("null-model");
  recorder.increment_rejected_requests();
  recorder.observe_compute_latency_by_worker(1, 0, "cpu", 18.0);
  recorder.observe_task_runtime_by_worker(1, 0, "cpu", 19.0);
  recorder.set_worker_inflight_gauge(1, 0, "cpu", 3);
  recorder.observe_io_copy_latency("h2d", 1, 0, "cpu", 20.0);
  recorder.increment_transfer_bytes("h2d", 1, 0, "cpu", 256);
}

TEST(MetricsRecorder, CreateMetricsRecorderInitializesAndRecordsMetrics)
{
  auto recorder = create_metrics_recorder(0);
  ASSERT_NE(recorder, nullptr);
  ASSERT_TRUE(recorder->enabled());

  auto metrics = recorder->registry();
  ASSERT_NE(metrics, nullptr);
  EXPECT_DOUBLE_EQ(metrics->gauges().queue_size->Value(), 0.0);
  EXPECT_DOUBLE_EQ(metrics->gauges().inflight_tasks->Value(), 0.0);
  EXPECT_DOUBLE_EQ(metrics->gauges().max_inflight_tasks->Value(), 0.0);

  const std::string model_name = "recorder-model";
  recorder->increment_requests_total();
  recorder->observe_inference_latency(11.5);
  recorder->set_queue_capacity(4);
  recorder->set_queue_size(2);
  recorder->set_inflight_tasks(3);
  recorder->set_starpu_worker_busy_ratio(1.5);
  recorder->set_max_inflight_tasks(9);
  recorder->set_server_health(true);
  recorder->set_starpu_prepared_queue_depth(5);
  recorder->set_batch_pending_jobs(6);
  recorder->set_congestion_flag(true);
  recorder->set_congestion_score(-0.25);
  recorder->set_congestion_arrival_rate(-1.0);
  recorder->set_congestion_completion_rate(7.25);
  recorder->set_congestion_rejection_rate(-2.0);
  recorder->set_congestion_rho(std::nan(""));
  recorder->set_congestion_fill_ewma(1.2);
  recorder->set_congestion_queue_growth_rate(2.5);
  recorder->set_congestion_queue_latency_p95(-3.0);
  recorder->set_congestion_queue_latency_p99(8.0);
  recorder->set_congestion_e2e_latency_p95(-4.0);
  recorder->set_congestion_e2e_latency_p99(9.0);
  recorder->increment_request_status(
      static_cast<int>(grpc::StatusCode::UNAVAILABLE), model_name);
  recorder->increment_requests_received(model_name);
  recorder->increment_inference_completed(model_name, 2);
  recorder->increment_inference_failure("submit", "oom", model_name, 3);
  recorder->observe_batch_size(4);
  recorder->observe_logical_batch_size(7);
  recorder->observe_batch_efficiency(1.25);
  recorder->observe_latency_breakdown(
      LatencyBreakdownMetrics{1, 2, 3, 4, 5, 6, 7, 8, 9});
  recorder->observe_starpu_task_runtime(14.0);
  recorder->observe_model_load_duration(150.0);
  recorder->set_model_loaded(model_name, "gpu:0", true);
  recorder->set_gpu_model_replication_policy(model_name, "per_worker");
  recorder->set_gpu_model_replicas_total(model_name, 4);
  recorder->set_starpu_cuda_worker_info(7, 0, true);
  recorder->increment_model_load_failure(model_name);
  recorder->increment_rejected_requests();
  recorder->observe_compute_latency_by_worker(7, 0, "cuda", 6.5);
  recorder->observe_task_runtime_by_worker(7, 0, "cuda", 3.25);
  recorder->set_worker_inflight_gauge(7, 0, "cuda", 2);
  recorder->observe_io_copy_latency("h2d", 7, 0, "cuda", 1.75);
  recorder->increment_transfer_bytes("h2d", 7, 0, "cuda", 512);

  const auto families = metrics->registry()->Collect();

  const auto requests_total = FindCounterValue(families, "requests_total", {});
  ASSERT_TRUE(requests_total.has_value());
  EXPECT_DOUBLE_EQ(*requests_total, 1.0);

  const auto rejected_total =
      FindCounterValue(families, "requests_rejected_total", {});
  ASSERT_TRUE(rejected_total.has_value());
  EXPECT_DOUBLE_EQ(*rejected_total, 1.0);

  const auto queue_size = FindGaugeValue(families, "inference_queue_size", {});
  ASSERT_TRUE(queue_size.has_value());
  EXPECT_DOUBLE_EQ(*queue_size, 2.0);

  const auto queue_capacity =
      FindGaugeValue(families, "inference_max_queue_size", {});
  ASSERT_TRUE(queue_capacity.has_value());
  EXPECT_DOUBLE_EQ(*queue_capacity, 4.0);

  const auto queue_ratio =
      FindGaugeValue(families, "inference_queue_fill_ratio", {});
  ASSERT_TRUE(queue_ratio.has_value());
  EXPECT_DOUBLE_EQ(*queue_ratio, 0.5);

  const auto inflight =
      FindGaugeValue(families, "inference_inflight_tasks", {});
  ASSERT_TRUE(inflight.has_value());
  EXPECT_DOUBLE_EQ(*inflight, 3.0);

  const auto max_inflight =
      FindGaugeValue(families, "inference_max_inflight_tasks", {});
  ASSERT_TRUE(max_inflight.has_value());
  EXPECT_DOUBLE_EQ(*max_inflight, 9.0);

  const auto busy_ratio =
      FindGaugeValue(families, "starpu_worker_busy_ratio", {});
  ASSERT_TRUE(busy_ratio.has_value());
  EXPECT_DOUBLE_EQ(*busy_ratio, 1.0);

  const auto server_health =
      FindGaugeValue(families, "server_health_state", {});
  ASSERT_TRUE(server_health.has_value());
  EXPECT_DOUBLE_EQ(*server_health, 1.0);

  const auto prepared_depth =
      FindGaugeValue(families, "starpu_prepared_queue_depth", {});
  ASSERT_TRUE(prepared_depth.has_value());
  EXPECT_DOUBLE_EQ(*prepared_depth, 5.0);

  const auto pending_jobs =
      FindGaugeValue(families, "inference_batch_collect_pending_jobs", {});
  ASSERT_TRUE(pending_jobs.has_value());
  EXPECT_DOUBLE_EQ(*pending_jobs, 6.0);

  const auto congestion_flag =
      FindGaugeValue(families, "inference_congestion_flag", {});
  ASSERT_TRUE(congestion_flag.has_value());
  EXPECT_DOUBLE_EQ(*congestion_flag, 1.0);

  const auto congestion_score =
      FindGaugeValue(families, "inference_congestion_score", {});
  ASSERT_TRUE(congestion_score.has_value());
  EXPECT_DOUBLE_EQ(*congestion_score, 0.0);

  const auto lambda_rps = FindGaugeValue(families, "inference_lambda_rps", {});
  ASSERT_TRUE(lambda_rps.has_value());
  EXPECT_DOUBLE_EQ(*lambda_rps, 0.0);

  const auto mu_rps = FindGaugeValue(families, "inference_mu_rps", {});
  ASSERT_TRUE(mu_rps.has_value());
  EXPECT_DOUBLE_EQ(*mu_rps, 7.25);

  const auto rejection_rps =
      FindGaugeValue(families, "inference_rejection_rate_rps", {});
  ASSERT_TRUE(rejection_rps.has_value());
  EXPECT_DOUBLE_EQ(*rejection_rps, 0.0);

  const auto rho_ewma = FindGaugeValue(families, "inference_rho_ewma", {});
  ASSERT_TRUE(rho_ewma.has_value());
  EXPECT_DOUBLE_EQ(*rho_ewma, 0.0);

  const auto fill_ewma =
      FindGaugeValue(families, "inference_queue_fill_ratio_ewma", {});
  ASSERT_TRUE(fill_ewma.has_value());
  EXPECT_DOUBLE_EQ(*fill_ewma, 1.0);

  const auto growth_rate =
      FindGaugeValue(families, "inference_queue_growth_rate", {});
  ASSERT_TRUE(growth_rate.has_value());
  EXPECT_DOUBLE_EQ(*growth_rate, 2.5);

  const auto queue_p95 =
      FindGaugeValue(families, "inference_queue_latency_p95_ms", {});
  ASSERT_TRUE(queue_p95.has_value());
  EXPECT_DOUBLE_EQ(*queue_p95, 0.0);

  const auto queue_p99 =
      FindGaugeValue(families, "inference_queue_latency_p99_ms", {});
  ASSERT_TRUE(queue_p99.has_value());
  EXPECT_DOUBLE_EQ(*queue_p99, 8.0);

  const auto e2e_p95 =
      FindGaugeValue(families, "inference_e2e_latency_p95_ms", {});
  ASSERT_TRUE(e2e_p95.has_value());
  EXPECT_DOUBLE_EQ(*e2e_p95, 0.0);

  const auto e2e_p99 =
      FindGaugeValue(families, "inference_e2e_latency_p99_ms", {});
  ASSERT_TRUE(e2e_p99.has_value());
  EXPECT_DOUBLE_EQ(*e2e_p99, 9.0);

  const auto requests_by_status = FindCounterValue(
      families, "requests_by_status_total",
      {{"code", "UNAVAILABLE"}, {"model", model_name}});
  ASSERT_TRUE(requests_by_status.has_value());
  EXPECT_DOUBLE_EQ(*requests_by_status, 1.0);

  const auto requests_received = FindCounterValue(
      families, "requests_received_total", {{"model", model_name}});
  ASSERT_TRUE(requests_received.has_value());
  EXPECT_DOUBLE_EQ(*requests_received, 1.0);

  const auto completed = FindCounterValue(
      families, "inference_completed_total", {{"model", model_name}});
  ASSERT_TRUE(completed.has_value());
  EXPECT_DOUBLE_EQ(*completed, 2.0);

  const auto failures = FindCounterValue(
      families, "inference_failures_total",
      {{"stage", "submit"}, {"reason", "oom"}, {"model", model_name}});
  ASSERT_TRUE(failures.has_value());
  EXPECT_DOUBLE_EQ(*failures, 3.0);

  const auto model_load_failures = FindCounterValue(
      families, "model_load_failures_total", {{"model", model_name}});
  ASSERT_TRUE(model_load_failures.has_value());
  EXPECT_DOUBLE_EQ(*model_load_failures, 1.0);

  const auto model_loaded = FindGaugeValue(
      families, "models_loaded", {{"model", model_name}, {"device", "gpu:0"}});
  ASSERT_TRUE(model_loaded.has_value());
  EXPECT_DOUBLE_EQ(*model_loaded, 1.0);

  const auto replication_policy = FindGaugeValue(
      families, "gpu_model_replication_policy_info",
      {{"model", model_name}, {"policy", "per_worker"}});
  ASSERT_TRUE(replication_policy.has_value());
  EXPECT_DOUBLE_EQ(*replication_policy, 1.0);

  const auto replicas_total = FindGaugeValue(
      families, "gpu_model_replicas_total", {{"model", model_name}});
  ASSERT_TRUE(replicas_total.has_value());
  EXPECT_DOUBLE_EQ(*replicas_total, 4.0);

  const auto cuda_worker = FindGaugeValue(
      families, "starpu_cuda_worker_info",
      {{"device", "0"}, {"worker_id", "7"}});
  ASSERT_TRUE(cuda_worker.has_value());
  EXPECT_DOUBLE_EQ(*cuda_worker, 1.0);

  const auto* inference_latency =
      FindHistogramMetric(families, "inference_latency_ms", {});
  ASSERT_NE(inference_latency, nullptr);
  EXPECT_EQ(inference_latency->histogram.sample_count, 1);
  EXPECT_DOUBLE_EQ(inference_latency->histogram.sample_sum, 11.5);

  const auto* batch_size_metric =
      FindHistogramMetric(families, "inference_batch_size", {});
  ASSERT_NE(batch_size_metric, nullptr);
  EXPECT_EQ(batch_size_metric->histogram.sample_count, 1);
  EXPECT_DOUBLE_EQ(batch_size_metric->histogram.sample_sum, 4.0);

  const auto* logical_batch_metric =
      FindHistogramMetric(families, "inference_logical_batch_size", {});
  ASSERT_NE(logical_batch_metric, nullptr);
  EXPECT_EQ(logical_batch_metric->histogram.sample_count, 1);
  EXPECT_DOUBLE_EQ(logical_batch_metric->histogram.sample_sum, 7.0);

  const auto* batch_efficiency_metric =
      FindHistogramMetric(families, "inference_batch_efficiency_ratio", {});
  ASSERT_NE(batch_efficiency_metric, nullptr);
  EXPECT_EQ(batch_efficiency_metric->histogram.sample_count, 1);
  EXPECT_DOUBLE_EQ(batch_efficiency_metric->histogram.sample_sum, 1.25);

  const auto* model_load_metric =
      FindHistogramMetric(families, "model_load_duration_ms", {});
  ASSERT_NE(model_load_metric, nullptr);
  EXPECT_EQ(model_load_metric->histogram.sample_count, 1);
  EXPECT_DOUBLE_EQ(model_load_metric->histogram.sample_sum, 150.0);

  const auto* task_runtime_metric =
      FindHistogramMetric(families, "starpu_task_runtime_ms", {});
  ASSERT_NE(task_runtime_metric, nullptr);
  EXPECT_EQ(task_runtime_metric->histogram.sample_count, 1);
  EXPECT_DOUBLE_EQ(task_runtime_metric->histogram.sample_sum, 14.0);

  const auto* queue_latency_metric =
      FindHistogramMetric(families, "inference_queue_latency_ms", {});
  ASSERT_NE(queue_latency_metric, nullptr);
  EXPECT_EQ(queue_latency_metric->histogram.sample_count, 1);
  EXPECT_DOUBLE_EQ(queue_latency_metric->histogram.sample_sum, 1.0);

  const auto* postprocess_metric =
      FindHistogramMetric(families, "inference_postprocess_latency_ms", {});
  ASSERT_NE(postprocess_metric, nullptr);
  EXPECT_EQ(postprocess_metric->histogram.sample_count, 1);
  EXPECT_DOUBLE_EQ(postprocess_metric->histogram.sample_sum, 9.0);

  const auto* compute_metric = FindHistogramMetric(
      families, "inference_compute_latency_ms_by_worker",
      {{"worker_id", "7"}, {"device", "0"}, {"worker_type", "cuda"}});
  ASSERT_NE(compute_metric, nullptr);
  EXPECT_EQ(compute_metric->histogram.sample_count, 1);
  EXPECT_DOUBLE_EQ(compute_metric->histogram.sample_sum, 6.5);

  const auto* worker_runtime_metric = FindHistogramMetric(
      families, "starpu_task_runtime_ms_by_worker",
      {{"worker_id", "7"}, {"device", "0"}, {"worker_type", "cuda"}});
  ASSERT_NE(worker_runtime_metric, nullptr);
  EXPECT_EQ(worker_runtime_metric->histogram.sample_count, 1);
  EXPECT_DOUBLE_EQ(worker_runtime_metric->histogram.sample_sum, 3.25);

  const auto worker_inflight = FindGaugeValue(
      families, "starpu_worker_inflight_tasks",
      {{"worker_id", "7"}, {"device", "0"}, {"worker_type", "cuda"}});
  ASSERT_TRUE(worker_inflight.has_value());
  EXPECT_DOUBLE_EQ(*worker_inflight, 2.0);

  const auto* io_metric = FindHistogramMetric(
      families, "inference_io_copy_ms",
      {{"direction", "h2d"},
       {"worker_id", "7"},
       {"device", "0"},
       {"worker_type", "cuda"}});
  ASSERT_NE(io_metric, nullptr);
  EXPECT_EQ(io_metric->histogram.sample_count, 1);
  EXPECT_DOUBLE_EQ(io_metric->histogram.sample_sum, 1.75);

  const auto transfer_bytes = FindCounterValue(
      families, "inference_transfer_bytes_total",
      {{"direction", "h2d"},
       {"worker_id", "7"},
       {"device", "0"},
       {"worker_type", "cuda"}});
  ASSERT_TRUE(transfer_bytes.has_value());
  EXPECT_DOUBLE_EQ(*transfer_bytes, 512.0);
}

TEST(MetricsRecorder, SetQueueCapacityReturnsWhenQueueFillRatioGaugeMissing)
{
  auto recorder = create_metrics_recorder(0);
  ASSERT_NE(recorder, nullptr);

  auto metrics = recorder->registry();
  ASSERT_NE(metrics, nullptr);
  starpu_server::testing::MetricsRegistryTestAccessor::ClearQueueFillRatioGauge(
      *metrics);

  recorder->set_queue_capacity(8);

  EXPECT_EQ(metrics->queue_capacity_value(), 8U);
}

TEST(MetricsRecorder, SetQueueCapacityResetsRatioWhenQueueSizeGaugeMissing)
{
  auto recorder = create_metrics_recorder(0);
  ASSERT_NE(recorder, nullptr);

  auto metrics = recorder->registry();
  ASSERT_NE(metrics, nullptr);

  recorder->set_queue_capacity(4);
  recorder->set_queue_size(2);

  const auto ratio_before = FindGaugeValue(
      metrics->registry()->Collect(), "inference_queue_fill_ratio", {});
  ASSERT_TRUE(ratio_before.has_value());
  EXPECT_DOUBLE_EQ(*ratio_before, 0.5);

  starpu_server::testing::MetricsRegistryTestAccessor::ClearQueueSizeGauge(
      *metrics);
  recorder->set_queue_capacity(8);

  const auto ratio_after = FindGaugeValue(
      metrics->registry()->Collect(), "inference_queue_fill_ratio", {});
  ASSERT_TRUE(ratio_after.has_value());
  EXPECT_DOUBLE_EQ(*ratio_after, 0.0);
}

TEST(MetricsRecorder, IncrementInferenceFailureRecordsCounter)
{
  auto recorder = create_metrics_recorder(0);
  ASSERT_NE(recorder, nullptr);

  recorder->increment_inference_failure(
      "dispatch", "retry", "recorder-model", 2);

  const auto metrics = recorder->registry();
  ASSERT_NE(metrics, nullptr);
  const auto value = FindCounterValue(
      metrics->registry()->Collect(), "inference_failures_total",
      {{"stage", "dispatch"},
       {"reason", "retry"},
       {"model", "recorder-model"}});
  ASSERT_TRUE(value.has_value());
  EXPECT_DOUBLE_EQ(*value, 2.0);
}

TEST(MetricsRecorder, SetModelLoadedRecordsGauge)
{
  auto recorder = create_metrics_recorder(0);
  ASSERT_NE(recorder, nullptr);

  recorder->set_model_loaded("recorder-model", "gpu:2", true);

  const auto metrics = recorder->registry();
  ASSERT_NE(metrics, nullptr);
  const auto value = FindGaugeValue(
      metrics->registry()->Collect(), "models_loaded",
      {{"model", "recorder-model"}, {"device", "gpu:2"}});
  ASSERT_TRUE(value.has_value());
  EXPECT_DOUBLE_EQ(*value, 1.0);
}

TEST(MetricsRecorder, CreateMetricsRecorderReturnsNullWhenMetricsRegistryThrows)
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

  EXPECT_EQ(create_metrics_recorder(0), nullptr);
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

  starpu_server::testing::MetricsRegistryTestAccessor::
      ClearStarpuWorkerInflightFamily(metrics);
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

  starpu_server::testing::MetricsRegistryTestAccessor::ClearModelsLoadedFamily(
      metrics);
  metrics.set_model_loaded_flag(
      MetricsRegistry::ModelLabel{"model-x"},
      MetricsRegistry::DeviceLabel{"dev-x"}, true);

  const auto value = FindGaugeValue(
      metrics.registry()->Collect(), "models_loaded",
      {{"model", "model-x"}, {"device", "dev-x"}});
  EXPECT_FALSE(value.has_value());
}

TEST(MetricsRegistry, RecordsGpuReplicationStartupMetrics)
{
  MetricsRegistry metrics(
      0, [] { return std::vector<MetricsRegistry::GpuSample>{}; },
      [] { return std::optional<double>{}; }, false);

  metrics.set_gpu_model_replication_policy_flag(
      MetricsRegistry::ModelLabel{"model-x"}, "per_worker");
  metrics.set_gpu_model_replicas_total_gauge(
      MetricsRegistry::ModelLabel{"model-x"}, 3);
  metrics.set_starpu_cuda_worker_info_gauge(7, 0, true);
  metrics.set_starpu_cuda_worker_info_gauge(9, 0, true);

  const auto families = metrics.registry()->Collect();

  const auto policy_value = FindGaugeValue(
      families, "gpu_model_replication_policy_info",
      {{"model", "model-x"}, {"policy", "per_worker"}});
  ASSERT_TRUE(policy_value.has_value());
  EXPECT_DOUBLE_EQ(*policy_value, 1.0);

  const auto replica_total = FindGaugeValue(
      families, "gpu_model_replicas_total", {{"model", "model-x"}});
  ASSERT_TRUE(replica_total.has_value());
  EXPECT_DOUBLE_EQ(*replica_total, 3.0);

  const auto worker_7 = FindGaugeValue(
      families, "starpu_cuda_worker_info",
      {{"device", "0"}, {"worker_id", "7"}});
  ASSERT_TRUE(worker_7.has_value());
  EXPECT_DOUBLE_EQ(*worker_7, 1.0);

  const auto worker_9 = FindGaugeValue(
      families, "starpu_cuda_worker_info",
      {{"device", "0"}, {"worker_id", "9"}});
  ASSERT_TRUE(worker_9.has_value());
  EXPECT_DOUBLE_EQ(*worker_9, 1.0);
}

TEST(MetricsRegistry, GpuReplicationPolicySkipsWhenFamilyMissing)
{
  MetricsRegistry metrics(
      0, [] { return std::vector<MetricsRegistry::GpuSample>{}; },
      [] { return std::optional<double>{}; }, false);

  starpu_server::testing::MetricsRegistryTestAccessor::
      ClearGpuModelReplicationPolicyInfoFamily(metrics);
  metrics.set_gpu_model_replication_policy_flag(
      MetricsRegistry::ModelLabel{"model-x"}, "per_worker");

  const auto value = FindGaugeValue(
      metrics.registry()->Collect(), "gpu_model_replication_policy_info",
      {{"model", "model-x"}, {"policy", "per_worker"}});
  EXPECT_FALSE(value.has_value());
}

TEST(MetricsRegistry, GpuReplicasTotalSkipsWhenFamilyMissing)
{
  MetricsRegistry metrics(
      0, [] { return std::vector<MetricsRegistry::GpuSample>{}; },
      [] { return std::optional<double>{}; }, false);

  starpu_server::testing::MetricsRegistryTestAccessor::
      ClearGpuModelReplicasTotalFamily(metrics);
  metrics.set_gpu_model_replicas_total_gauge(
      MetricsRegistry::ModelLabel{"model-x"}, 3);

  const auto value = FindGaugeValue(
      metrics.registry()->Collect(), "gpu_model_replicas_total",
      {{"model", "model-x"}});
  EXPECT_FALSE(value.has_value());
}

TEST(MetricsRegistry, StarpuCudaWorkerInfoSkipsWhenFamilyMissing)
{
  MetricsRegistry metrics(
      0, [] { return std::vector<MetricsRegistry::GpuSample>{}; },
      [] { return std::optional<double>{}; }, false);

  starpu_server::testing::MetricsRegistryTestAccessor::
      ClearStarpuCudaWorkerInfoFamily(metrics);
  metrics.set_starpu_cuda_worker_info_gauge(7, 0, true);

  const auto value = FindGaugeValue(
      metrics.registry()->Collect(), "starpu_cuda_worker_info",
      {{"device", "0"}, {"worker_id", "7"}});
  EXPECT_FALSE(value.has_value());
}

TEST(MetricsRegistry, WorkerMetricOverflowUsesReservedBucket)
{
  MetricsRegistry metrics(
      0, [] { return std::vector<MetricsRegistry::GpuSample>{}; },
      [] { return std::optional<double>{}; }, false);

  constexpr std::size_t kMaxLabelSeriesForTest = 10000;
  constexpr std::size_t kOverflowSamples = 2;
  for (std::size_t i = 0; i < kMaxLabelSeriesForTest + kOverflowSamples; ++i) {
    metrics.set_worker_inflight_gauge(static_cast<int>(i), 0, "cpu", i + 1);
  }

  const auto overflow_value = FindGaugeValue(
      metrics.registry()->Collect(), "starpu_worker_inflight_tasks",
      {{"worker_id", "__overflow__"},
       {"device", "__overflow__"},
       {"worker_type", "__overflow__"}});
  ASSERT_TRUE(overflow_value.has_value());
  EXPECT_DOUBLE_EQ(
      *overflow_value,
      static_cast<double>(kMaxLabelSeriesForTest + kOverflowSamples));
}

TEST(MetricsRegistry, IoMetricOverflowUsesReservedBucket)
{
  MetricsRegistry metrics(
      0, [] { return std::vector<MetricsRegistry::GpuSample>{}; },
      [] { return std::optional<double>{}; }, false);

  constexpr std::size_t kMaxLabelSeriesForTest = 10000;
  constexpr std::size_t kOverflowSamples = 2;
  for (std::size_t i = 0; i < kMaxLabelSeriesForTest + kOverflowSamples; ++i) {
    metrics.increment_transfer_bytes("h2d", static_cast<int>(i), 0, "cpu", 1);
  }

  const auto overflow_value = FindCounterValue(
      metrics.registry()->Collect(), "inference_transfer_bytes_total",
      {{"direction", "__overflow__"},
       {"worker_id", "__overflow__"},
       {"device", "__overflow__"},
       {"worker_type", "__overflow__"}});
  ASSERT_TRUE(overflow_value.has_value());
  EXPECT_DOUBLE_EQ(*overflow_value, static_cast<double>(kOverflowSamples));
}

TEST(MetricsRegistry, IncrementModelLoadFailureCounterSkipsWhenFamilyMissing)
{
  MetricsRegistry metrics(
      0, [] { return std::vector<MetricsRegistry::GpuSample>{}; },
      [] { return std::optional<double>{}; }, false);

  starpu_server::testing::MetricsRegistryTestAccessor::
      ClearModelLoadFailuresFamily(metrics);
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

  starpu_server::testing::MetricsRegistryTestAccessor::
      ClearInferenceFailuresFamily(metrics);
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

  starpu_server::testing::MetricsRegistryTestAccessor::
      ClearInferenceCompletedFamily(metrics);
  metrics.increment_completed_counter("model-w", 4);

  const auto value = FindCounterValue(
      metrics.registry()->Collect(), "inference_completed_total",
      {{"model", "model-w"}});
  EXPECT_FALSE(value.has_value());
}

TEST(MetricsRegistry, IncrementReceivedCounterSkipsWhenFamilyMissing)
{
  MetricsRegistry metrics(
      0, [] { return std::vector<MetricsRegistry::GpuSample>{}; },
      [] { return std::optional<double>{}; }, false);

  starpu_server::testing::MetricsRegistryTestAccessor::
      ClearRequestsReceivedFamily(metrics);
  metrics.increment_received_counter("model-recv");

  const auto value = FindCounterValue(
      metrics.registry()->Collect(), "requests_received_total",
      {{"model", "model-recv"}});
  EXPECT_FALSE(value.has_value());
}

TEST(MetricsRegistry, IncrementStatusCounterSkipsWhenFamilyMissing)
{
  MetricsRegistry metrics(
      0, [] { return std::vector<MetricsRegistry::GpuSample>{}; },
      [] { return std::optional<double>{}; }, false);

  starpu_server::testing::MetricsRegistryTestAccessor::
      ClearRequestsByStatusFamily(metrics);
  metrics.increment_status_counter(
      MetricsRegistry::StatusCodeLabel{"5"},
      MetricsRegistry::ModelLabel{"model-status"});

  const auto value = FindCounterValue(
      metrics.registry()->Collect(), "requests_by_status_total",
      {{"code", "5"}, {"model", "model-status"}});
  EXPECT_FALSE(value.has_value());
}
