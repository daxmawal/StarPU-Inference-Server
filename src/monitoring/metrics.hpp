#pragma once

#include <prometheus/counter.h>
#include <prometheus/gauge.h>
#include <prometheus/histogram.h>
#include <prometheus/registry.h>

#include <cstddef>
#include <filesystem>
#include <functional>
#include <iosfwd>
#include <memory>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>

namespace prometheus {
class Exposer;
class Collectable;
template <typename T>
class Family;
}  // namespace prometheus

namespace starpu_server {

class MetricsRegistry {
 public:
  struct ExposerHandle {
    ExposerHandle() = default;
    ExposerHandle(const ExposerHandle&) = delete;
    auto operator=(const ExposerHandle&) -> ExposerHandle& = delete;
    ExposerHandle(ExposerHandle&&) = delete;
    auto operator=(ExposerHandle&&) -> ExposerHandle& = delete;
    virtual ~ExposerHandle() = default;
    virtual void RegisterCollectable(
        const std::shared_ptr<prometheus::Collectable>& collectable) = 0;
    virtual void RemoveCollectable(
        const std::shared_ptr<prometheus::Collectable>& collectable) = 0;
  };

  struct GpuSample {
    int index{0};
    double util_percent{0.0};
    double mem_used_bytes{0.0};
    double mem_total_bytes{0.0};
  };

  using GpuStatsProvider = std::function<std::vector<GpuSample>()>;
  using CpuUsageProvider = std::function<std::optional<double>()>;

  explicit MetricsRegistry(int port);
  MetricsRegistry(
      int port, GpuStatsProvider gpu_provider, CpuUsageProvider cpu_provider,
      bool start_sampler_thread = true,
      std::unique_ptr<ExposerHandle> exposer_handle = nullptr);
  ~MetricsRegistry() noexcept;
  MetricsRegistry(const MetricsRegistry&) = delete;
  auto operator=(const MetricsRegistry&) -> MetricsRegistry& = delete;
  MetricsRegistry(MetricsRegistry&&) = delete;
  auto operator=(MetricsRegistry&&) -> MetricsRegistry& = delete;

  std::shared_ptr<prometheus::Registry>
      registry;  // NOLINT(cppcoreguidelines-non-private-member-variables-in-classes)
  prometheus::Counter*
      requests_total;  // NOLINT(cppcoreguidelines-non-private-member-variables-in-classes)
  prometheus::Histogram*
      inference_latency;  // NOLINT(cppcoreguidelines-non-private-member-variables-in-classes)
  prometheus::Gauge*
      queue_size_gauge;  // NOLINT(cppcoreguidelines-non-private-member-variables-in-classes)

  prometheus::Gauge* system_cpu_usage_percent{
      nullptr};  // NOLINT(cppcoreguidelines-non-private-member-variables-in-classes)
  prometheus::Family<prometheus::Gauge>* gpu_utilization_family{
      nullptr};  // NOLINT(cppcoreguidelines-non-private-member-variables-in-classes)
  prometheus::Family<prometheus::Gauge>* gpu_memory_used_bytes_family{
      nullptr};  // NOLINT(cppcoreguidelines-non-private-member-variables-in-classes)
  prometheus::Family<prometheus::Gauge>* gpu_memory_total_bytes_family{
      nullptr};  // NOLINT(cppcoreguidelines-non-private-member-variables-in-classes)

  void run_sampling_request_nb();
  void request_stop();

  auto has_gpu_stats_provider() const -> bool;
  auto has_cpu_usage_provider() const -> bool;

 private:
  void initialize(
      int port, bool start_sampler_thread,
      std::unique_ptr<ExposerHandle> exposer_handle);
  void perform_sampling_request_nb();
  void sampling_loop(const std::stop_token& stop);

  std::unique_ptr<ExposerHandle> exposer_;
  std::jthread sampler_thread_;
  GpuStatsProvider gpu_stats_provider_;
  CpuUsageProvider cpu_usage_provider_;
  std::unordered_map<int, prometheus::Gauge*> gpu_utilization_gauges_;
  std::unordered_map<int, prometheus::Gauge*> gpu_memory_used_gauges_;
  std::unordered_map<int, prometheus::Gauge*> gpu_memory_total_gauges_;
};

auto init_metrics(int port) -> bool;
void shutdown_metrics();
void set_queue_size(std::size_t size);
auto get_metrics() -> std::shared_ptr<MetricsRegistry>;

}  // namespace starpu_server

namespace starpu_server::monitoring::detail {

struct CpuTotals {
  unsigned long long user{0};
  unsigned long long nice{0};
  unsigned long long system{0};
  unsigned long long idle{0};
  unsigned long long iowait{0};
  unsigned long long irq{0};
  unsigned long long softirq{0};
  unsigned long long steal{0};
};

auto read_total_cpu_times(std::istream& input, CpuTotals& out) -> bool;
auto read_total_cpu_times(const std::filesystem::path& path, CpuTotals& out)
    -> bool;
auto make_cpu_usage_provider(std::function<bool(CpuTotals&)> reader)
    -> starpu_server::MetricsRegistry::CpuUsageProvider;

}  // namespace starpu_server::monitoring::detail
