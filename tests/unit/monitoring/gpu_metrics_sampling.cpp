#include <gtest/gtest.h>
#include <prometheus/client_metric.h>
#include <prometheus/metric_family.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "monitoring/metrics.hpp"
#include "monitoring/metrics_test_api.hpp"
#include "utils/monotonic_clock.hpp"
#include "utils/perf_observer.hpp"

using namespace starpu_server;

namespace {

struct CerrCapture {
  CerrCapture() : old_buf(std::cerr.rdbuf(stream.rdbuf())) {}

  ~CerrCapture() { std::cerr.rdbuf(old_buf); }

  auto str() const -> std::string { return stream.str(); }

  std::stringstream stream;
  std::streambuf* old_buf;
};

struct ProcessMetricOverrideGuard {
  ProcessMetricOverrideGuard(
      std::function<std::optional<double>()> open_fds_override,
      std::function<std::optional<double>()> rss_override)
  {
    monitoring::detail::set_process_open_fds_reader_override(
        std::move(open_fds_override));
    monitoring::detail::set_process_rss_bytes_reader_override(
        std::move(rss_override));
  }

  ~ProcessMetricOverrideGuard()
  {
    monitoring::detail::set_process_open_fds_reader_override({});
    monitoring::detail::set_process_rss_bytes_reader_override({});
  }
};

struct PerfObserverGuard {
  PerfObserverGuard() { perf_observer::reset(); }

  ~PerfObserverGuard() { perf_observer::reset(); }
};

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
FindGaugeValue(
    const prometheus::MetricFamily& family,
    std::string_view label_value) -> std::optional<double>
{
  for (const auto& metric : family.metric) {
    const bool matches = std::ranges::any_of(
        metric.label,
        [label_value](const prometheus::ClientMetric::Label& label) {
          return label.name == "gpu" && label.value == label_value;
        });
    if (matches) {
      return metric.gauge.value;
    }
  }
  return std::nullopt;
}

}  // namespace

TEST(MetricsSampling, UpdatesGpuAndCpuGaugesOnSuccess)
{
  const std::vector<MetricsRegistry::GpuSample> samples{
      MetricsRegistry::GpuSample{
          .index = 0,
          .util_percent = 42.0,
          .mem_used_bytes = 1024.0,
          .mem_total_bytes = 2048.0},
      MetricsRegistry::GpuSample{
          .index = 1,
          .util_percent = 13.0,
          .mem_used_bytes = 512.0,
          .mem_total_bytes = 1024.0}};

  auto gpu_provider = [samples]() { return samples; };
  bool cpu_called = false;
  constexpr double kCpuUsage = 33.5;
  auto cpu_provider = [kCpuUsage, &cpu_called]() mutable {
    cpu_called = true;
    return std::optional<double>{kCpuUsage};
  };

  MetricsRegistry metrics(
      0, std::move(gpu_provider), std::move(cpu_provider),
      /*start_sampler_thread=*/false);

  metrics.run_sampling_request_nb();

  const auto families = metrics.registry()->Collect();

  const auto* cpu_family = FindFamily(families, "system_cpu_usage_percent");
  ASSERT_NE(cpu_family, nullptr);
  ASSERT_EQ(cpu_family->metric.size(), 1);
  EXPECT_NEAR(cpu_family->metric.front().gauge.value, kCpuUsage, 1e-6);
  EXPECT_TRUE(cpu_called);

  const auto* util_family = FindFamily(families, "gpu_utilization_percent");
  ASSERT_NE(util_family, nullptr);
  EXPECT_EQ(util_family->metric.size(), samples.size());
  const auto util0 = FindGaugeValue(*util_family, "0");
  ASSERT_TRUE(util0.has_value());
  EXPECT_DOUBLE_EQ(*util0, samples[0].util_percent);
  const auto util1 = FindGaugeValue(*util_family, "1");
  ASSERT_TRUE(util1.has_value());
  EXPECT_DOUBLE_EQ(*util1, samples[1].util_percent);

  const auto* used_family = FindFamily(families, "gpu_memory_used_bytes");
  ASSERT_NE(used_family, nullptr);
  const auto used0 = FindGaugeValue(*used_family, "0");
  ASSERT_TRUE(used0.has_value());
  EXPECT_DOUBLE_EQ(*used0, samples[0].mem_used_bytes);
  const auto used1 = FindGaugeValue(*used_family, "1");
  ASSERT_TRUE(used1.has_value());
  EXPECT_DOUBLE_EQ(*used1, samples[1].mem_used_bytes);

  const auto* total_family = FindFamily(families, "gpu_memory_total_bytes");
  ASSERT_NE(total_family, nullptr);
  const auto total0 = FindGaugeValue(*total_family, "0");
  ASSERT_TRUE(total0.has_value());
  EXPECT_DOUBLE_EQ(*total0, samples[0].mem_total_bytes);
  const auto total1 = FindGaugeValue(*total_family, "1");
  ASSERT_TRUE(total1.has_value());
  EXPECT_DOUBLE_EQ(*total1, samples[1].mem_total_bytes);

  metrics.request_stop();
}

TEST(MetricsSampling, RecordsGpuTemperatureAndPower)
{
  const std::vector<MetricsRegistry::GpuSample> samples{
      MetricsRegistry::GpuSample{
          .index = 0,
          .util_percent = 50.0,
          .mem_used_bytes = 123.0,
          .mem_total_bytes = 456.0,
          .temperature_celsius = 70.5,
          .power_watts = 120.0}};

  auto gpu_provider = [samples]() { return samples; };
  auto cpu_provider = []() { return std::optional<double>{}; };

  MetricsRegistry metrics(
      0, std::move(gpu_provider), std::move(cpu_provider),
      /*start_sampler_thread=*/false);

  metrics.run_sampling_request_nb();

  const auto families = metrics.registry()->Collect();

  const auto* temp_family = FindFamily(families, "gpu_temperature_celsius");
  ASSERT_NE(temp_family, nullptr);
  const auto temp0 = FindGaugeValue(*temp_family, "0");
  ASSERT_TRUE(temp0.has_value());
  EXPECT_DOUBLE_EQ(*temp0, samples[0].temperature_celsius);

  const auto* power_family = FindFamily(families, "gpu_power_watts");
  ASSERT_NE(power_family, nullptr);
  const auto power0 = FindGaugeValue(*power_family, "0");
  ASSERT_TRUE(power0.has_value());
  EXPECT_DOUBLE_EQ(*power0, samples[0].power_watts);

  metrics.request_stop();
}

TEST(MetricsSampling, RemovesGpuTemperatureGaugeWhenValueIsNaN)
{
  const std::vector<MetricsRegistry::GpuSample> initial_samples{
      MetricsRegistry::GpuSample{
          .index = 0,
          .util_percent = 50.0,
          .mem_used_bytes = 123.0,
          .mem_total_bytes = 456.0,
          .temperature_celsius = 70.5,
          .power_watts = 120.0}};
  const std::vector<MetricsRegistry::GpuSample> nan_samples{
      MetricsRegistry::GpuSample{
          .index = 0,
          .util_percent = 51.0,
          .mem_used_bytes = 124.0,
          .mem_total_bytes = 456.0,
          .temperature_celsius = std::numeric_limits<double>::quiet_NaN(),
          .power_watts = 120.0}};

  int calls = 0;
  auto gpu_provider = [initial_samples, nan_samples, &calls]() mutable {
    ++calls;
    return calls == 1 ? initial_samples : nan_samples;
  };
  auto cpu_provider = []() { return std::optional<double>{}; };

  MetricsRegistry metrics(
      0, std::move(gpu_provider), std::move(cpu_provider),
      /*start_sampler_thread=*/false);

  metrics.run_sampling_request_nb();

  auto families = metrics.registry()->Collect();
  const auto* temp_family = FindFamily(families, "gpu_temperature_celsius");
  ASSERT_NE(temp_family, nullptr);
  const auto temp0 = FindGaugeValue(*temp_family, "0");
  ASSERT_TRUE(temp0.has_value());

  metrics.run_sampling_request_nb();

  families = metrics.registry()->Collect();
  temp_family = FindFamily(families, "gpu_temperature_celsius");
  // Collect omits empty families, so null means the gauge was removed.
  if (temp_family != nullptr) {
    const auto temp0_after = FindGaugeValue(*temp_family, "0");
    EXPECT_FALSE(temp0_after.has_value());
  }

  metrics.request_stop();
}

TEST(MetricsSampling, RemovesMissingGpuGauges)
{
  const std::vector<MetricsRegistry::GpuSample> initial_samples{
      MetricsRegistry::GpuSample{
          .index = 0,
          .util_percent = 42.0,
          .mem_used_bytes = 1024.0,
          .mem_total_bytes = 2048.0},
      MetricsRegistry::GpuSample{
          .index = 1,
          .util_percent = 13.0,
          .mem_used_bytes = 512.0,
          .mem_total_bytes = 1024.0}};
  const std::vector<MetricsRegistry::GpuSample> reduced_samples{
      MetricsRegistry::GpuSample{
          .index = 0,
          .util_percent = 12.0,
          .mem_used_bytes = 900.0,
          .mem_total_bytes = 2048.0}};

  int calls = 0;
  auto gpu_provider = [initial_samples, reduced_samples, &calls]() mutable {
    ++calls;
    return calls == 1 ? initial_samples : reduced_samples;
  };
  auto cpu_provider = []() { return std::optional<double>{}; };

  MetricsRegistry metrics(
      0, std::move(gpu_provider), std::move(cpu_provider),
      /*start_sampler_thread=*/false);

  metrics.run_sampling_request_nb();

  auto families = metrics.registry()->Collect();
  const auto* util_family = FindFamily(families, "gpu_utilization_percent");
  ASSERT_NE(util_family, nullptr);
  EXPECT_TRUE(FindGaugeValue(*util_family, "1").has_value());

  metrics.run_sampling_request_nb();

  families = metrics.registry()->Collect();
  util_family = FindFamily(families, "gpu_utilization_percent");
  ASSERT_NE(util_family, nullptr);
  EXPECT_TRUE(FindGaugeValue(*util_family, "0").has_value());
  EXPECT_FALSE(FindGaugeValue(*util_family, "1").has_value());

  metrics.request_stop();
}

TEST(MetricsSampling, SkipsProcessSamplingWhenGaugesNull)
{
  auto gpu_provider = []() {
    return std::vector<MetricsRegistry::GpuSample>{};
  };
  auto cpu_provider = []() { return std::optional<double>{}; };

  MetricsRegistry metrics(
      0, std::move(gpu_provider), std::move(cpu_provider),
      /*start_sampler_thread=*/false);

  starpu_server::testing::MetricsRegistryTestAccessor::ClearProcessOpenFdsGauge(
      metrics);
  starpu_server::testing::MetricsRegistryTestAccessor::
      ClearProcessResidentMemoryGauge(metrics);
  starpu_server::testing::MetricsRegistryTestAccessor::
      ClearInferenceThroughputGauge(metrics);

  EXPECT_NO_THROW(
      starpu_server::testing::MetricsRegistryTestAccessor::SampleProcessOpenFds(
          metrics));
  EXPECT_NO_THROW(starpu_server::testing::MetricsRegistryTestAccessor::
                      SampleProcessResidentMemory(metrics));
  EXPECT_NO_THROW(starpu_server::testing::MetricsRegistryTestAccessor::
                      SampleInferenceThroughput(metrics));

  metrics.request_stop();
}

TEST(MetricsSampling, MarksProcessGaugesUnknownWhenSamplingFails)
{
  auto gpu_provider = []() {
    return std::vector<MetricsRegistry::GpuSample>{};
  };
  auto cpu_provider = []() { return std::optional<double>{}; };

  MetricsRegistry metrics(
      0, std::move(gpu_provider), std::move(cpu_provider),
      /*start_sampler_thread=*/false);

  ProcessMetricOverrideGuard guard(
      []() { return std::optional<double>{}; },
      []() { return std::optional<double>{}; });

  auto* open_fds_gauge =
      starpu_server::testing::MetricsRegistryTestAccessor::ProcessOpenFdsGauge(
          metrics);
  auto* rss_gauge = starpu_server::testing::MetricsRegistryTestAccessor::
      ProcessResidentMemoryGauge(metrics);
  ASSERT_NE(open_fds_gauge, nullptr);
  ASSERT_NE(rss_gauge, nullptr);

  open_fds_gauge->Set(42.0);
  rss_gauge->Set(84.0);

  starpu_server::testing::MetricsRegistryTestAccessor::SampleProcessOpenFds(
      metrics);
  starpu_server::testing::MetricsRegistryTestAccessor::
      SampleProcessResidentMemory(metrics);

  EXPECT_TRUE(std::isnan(open_fds_gauge->Value()));
  EXPECT_TRUE(std::isnan(rss_gauge->Value()));

  metrics.request_stop();
}

TEST(MetricsSampling, UpdatesInferenceThroughputGaugeFromPerfObserver)
{
  using namespace std::chrono_literals;
  PerfObserverGuard guard;

  auto gpu_provider = []() {
    return std::vector<MetricsRegistry::GpuSample>{};
  };
  auto cpu_provider = []() { return std::optional<double>{}; };

  MetricsRegistry metrics(
      0, std::move(gpu_provider), std::move(cpu_provider),
      /*start_sampler_thread=*/false);

  const auto base = MonotonicClock::time_point{};
  const auto enqueue_time = base + 10ms;
  const auto completion_time = base + 210ms;
  constexpr std::size_t kBatchSize = 8;

  perf_observer::record_job(
      enqueue_time, completion_time, kBatchSize, /*is_warmup_job=*/false);

  starpu_server::testing::MetricsRegistryTestAccessor::
      SampleInferenceThroughput(metrics);

  auto* gauge = starpu_server::testing::MetricsRegistryTestAccessor::
      InferenceThroughputGauge(metrics);
  ASSERT_NE(gauge, nullptr);

  const double expected_duration =
      std::chrono::duration<double>(completion_time - enqueue_time).count();
  const double expected_throughput =
      static_cast<double>(kBatchSize) / expected_duration;
  EXPECT_DOUBLE_EQ(gauge->Value(), expected_throughput);

  metrics.request_stop();
}

TEST(MetricsSampling, UsesDefaultProvidersWhenMissing)
{
  MetricsRegistry metrics(
      0, MetricsRegistry::GpuStatsProvider{},
      MetricsRegistry::CpuUsageProvider{},
      /*start_sampler_thread=*/false);

  EXPECT_TRUE(metrics.has_gpu_stats_provider());
  EXPECT_TRUE(metrics.has_cpu_usage_provider());

  EXPECT_NO_THROW(metrics.run_sampling_request_nb());

  metrics.request_stop();
}

TEST(MetricsSampling, SkipsGpuMetricsWhenProviderMissing)
{
  MetricsRegistry metrics(
      0, MetricsRegistry::GpuStatsProvider{},
      MetricsRegistry::CpuUsageProvider{},
      /*start_sampler_thread=*/false);

  starpu_server::testing::MetricsRegistryTestAccessor::ClearGpuStatsProvider(
      metrics);

  ASSERT_FALSE(metrics.has_gpu_stats_provider());

  EXPECT_NO_THROW(metrics.run_sampling_request_nb());

  EXPECT_EQ(
      starpu_server::testing::MetricsRegistryTestAccessor::
          GpuUtilizationGaugeCount(metrics),
      0U);
  EXPECT_EQ(
      starpu_server::testing::MetricsRegistryTestAccessor::
          GpuMemoryUsedGaugeCount(metrics),
      0U);
  EXPECT_EQ(
      starpu_server::testing::MetricsRegistryTestAccessor::
          GpuMemoryTotalGaugeCount(metrics),
      0U);

  metrics.request_stop();
}

TEST(MetricsSampling, LogsStdExceptionsFromGpuProvider)
{
  auto gpu_provider = []() -> std::vector<MetricsRegistry::GpuSample> {
    throw std::runtime_error("boom");
  };
  auto cpu_provider = []() { return std::optional<double>{}; };

  MetricsRegistry metrics(
      0, std::move(gpu_provider), std::move(cpu_provider),
      /*start_sampler_thread=*/false);

  CerrCapture capture;
  EXPECT_NO_THROW(metrics.run_sampling_request_nb());
  metrics.request_stop();

  const std::string output = capture.str();
  EXPECT_NE(
      output.find("GPU metrics sampling failed: boom"), std::string::npos);
}

TEST(MetricsSampling, LogsUnknownExceptionsFromGpuProvider)
{
  auto gpu_provider = []() -> std::vector<MetricsRegistry::GpuSample> {
    throw 42;
  };
  auto cpu_provider = []() { return std::optional<double>{}; };

  MetricsRegistry metrics(
      0, std::move(gpu_provider), std::move(cpu_provider),
      /*start_sampler_thread=*/false);

  CerrCapture capture;
  EXPECT_NO_THROW(metrics.run_sampling_request_nb());
  metrics.request_stop();

  const std::string output = capture.str();
  EXPECT_NE(
      output.find("GPU metrics sampling failed due to an unknown error"),
      std::string::npos);
}

TEST(MetricsSampling, LogsStdExceptionsFromCpuProvider)
{
  auto gpu_provider = []() -> std::vector<MetricsRegistry::GpuSample> {
    return {};
  };
  auto cpu_provider = []() -> std::optional<double> {
    throw std::runtime_error("boom");
  };

  MetricsRegistry metrics(
      0, std::move(gpu_provider), std::move(cpu_provider),
      /*start_sampler_thread=*/false);

  CerrCapture capture;
  EXPECT_NO_THROW(metrics.run_sampling_request_nb());
  metrics.request_stop();

  const std::string output = capture.str();
  EXPECT_NE(
      output.find("CPU metrics sampling failed: boom"), std::string::npos);
}

TEST(MetricsSampling, LogsUnknownExceptionsFromCpuProvider)
{
  auto gpu_provider = []() -> std::vector<MetricsRegistry::GpuSample> {
    return {};
  };
  auto cpu_provider = []() -> std::optional<double> { throw 42; };

  MetricsRegistry metrics(
      0, std::move(gpu_provider), std::move(cpu_provider),
      /*start_sampler_thread=*/false);

  CerrCapture capture;
  EXPECT_NO_THROW(metrics.run_sampling_request_nb());
  metrics.request_stop();

  const std::string output = capture.str();
  EXPECT_NE(
      output.find("CPU metrics sampling failed due to an unknown error"),
      std::string::npos);
}
