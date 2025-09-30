#include <gtest/gtest.h>
#include <prometheus/client_metric.h>
#include <prometheus/metric_family.h>

#include <algorithm>
#include <iostream>
#include <optional>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "monitoring/metrics.hpp"

using namespace starpu_server;

namespace {

struct CerrCapture {
  CerrCapture() : old_buf(std::cerr.rdbuf(stream.rdbuf())) {}

  ~CerrCapture() { std::cerr.rdbuf(old_buf); }

  auto str() const -> std::string { return stream.str(); }

  std::stringstream stream;
  std::streambuf* old_buf;
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

  metrics.run_sampling_iteration();

  const auto families = metrics.registry->Collect();

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

TEST(MetricsSampling, UsesDefaultProvidersWhenMissing)
{
  MetricsRegistry metrics(
      0, MetricsRegistry::GpuStatsProvider{},
      MetricsRegistry::CpuUsageProvider{},
      /*start_sampler_thread=*/false);

  EXPECT_TRUE(metrics.has_gpu_stats_provider());
  EXPECT_TRUE(metrics.has_cpu_usage_provider());

  EXPECT_NO_THROW(metrics.run_sampling_iteration());

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
  EXPECT_NO_THROW(metrics.run_sampling_iteration());
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
  EXPECT_NO_THROW(metrics.run_sampling_iteration());
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
  EXPECT_NO_THROW(metrics.run_sampling_iteration());
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
  auto cpu_provider = []() -> std::optional<double> {
    throw 42;
  };

  MetricsRegistry metrics(
      0, std::move(gpu_provider), std::move(cpu_provider),
      /*start_sampler_thread=*/false);

  CerrCapture capture;
  EXPECT_NO_THROW(metrics.run_sampling_iteration());
  metrics.request_stop();

  const std::string output = capture.str();
  EXPECT_NE(
      output.find("CPU metrics sampling failed due to an unknown error"),
      std::string::npos);
}
