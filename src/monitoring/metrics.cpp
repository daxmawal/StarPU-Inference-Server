#include "monitoring/metrics.hpp"
#if defined(STARPU_TESTING)
#include "monitoring/metrics_test_api.hpp"
#endif

#include <prometheus/exposer.h>
#include <prometheus/histogram.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef STARPU_HAVE_NVML
#include <nvml.h>
#endif

#include "utils/logger.hpp"
#include "utils/perf_observer.hpp"

namespace starpu_server {

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
class TestingExposerHandle : public MetricsRegistry::ExposerHandle {
 public:
  void RegisterCollectable(
      const std::shared_ptr<prometheus::Collectable>& collectable) override
  {
  }

  void RemoveCollectable(
      const std::shared_ptr<prometheus::Collectable>& collectable) override
  {
  }
};
#endif  // SONAR_IGNORE_END

namespace monitoring::detail {

auto
read_total_cpu_times(std::istream& input, CpuTotals& out) -> bool
{
  std::string cpu{};
  if (!(input >> cpu)) {
    return false;
  }
  if (cpu != "cpu") {
    return false;
  }
  if (!(input >> out.user >> out.nice >> out.system >> out.idle >> out.iowait >>
        out.irq >> out.softirq >> out.steal)) {
    return false;
  }
  return true;
}

auto
read_total_cpu_times(const std::filesystem::path& path, CpuTotals& out) -> bool
{
  std::ifstream input{path};
  if (!input.is_open()) {
    return false;
  }
  return read_total_cpu_times(input, out);
}

}  // namespace monitoring::detail

class PrometheusExposerHandle : public MetricsRegistry::ExposerHandle {
 public:
  explicit PrometheusExposerHandle(std::unique_ptr<prometheus::Exposer> exposer)
      : exposer_(std::move(exposer))
  {
  }

  void RegisterCollectable(
      const std::shared_ptr<prometheus::Collectable>& collectable) override
  {
    exposer_->RegisterCollectable(collectable);
  }

  void RemoveCollectable(
      const std::shared_ptr<prometheus::Collectable>& collectable) override
  {
    exposer_->RemoveCollectable(collectable);
  }

 private:
  std::unique_ptr<prometheus::Exposer> exposer_;
};

auto make_default_cpu_usage_provider() -> MetricsRegistry::CpuUsageProvider;
auto make_default_cpu_usage_provider(
    std::function<bool(monitoring::detail::CpuTotals&)> reader)
    -> MetricsRegistry::CpuUsageProvider;

namespace {
using monitoring::detail::CpuTotals;

const std::filesystem::path kProcStatm{"/proc/self/statm"};

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

constexpr std::size_t kMaxLabelSeries = 10000;
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
constexpr auto kSamplingErrorLogThrottle = std::chrono::seconds(0);
#else
constexpr auto kSamplingErrorLogThrottle = std::chrono::seconds(60);
#endif  // SONAR_IGNORE_END

#include "metrics_labels.inl"

auto
ensure_gpu_gauge(
    std::unordered_map<int, prometheus::Gauge*>& gauges,
    prometheus::Family<prometheus::Gauge>* family, int gpu_index,
    const std::string& label) -> prometheus::Gauge*
{
  auto [entry, inserted] = gauges.try_emplace(gpu_index, nullptr);
  if (inserted) {
    entry->second = &family->Add({{"gpu", label}});
  }
  return entry->second;
}

void
set_or_clear_nan(
    std::unordered_map<int, prometheus::Gauge*>& gauges,
    prometheus::Family<prometheus::Gauge>* family, int gpu_index,
    const std::string& label, double value)
{
  if (std::isnan(value)) {
    if (auto entry = gauges.find(gpu_index); entry != gauges.end()) {
      if (family != nullptr) {
        family->Remove(entry->second);
      }
      gauges.erase(entry);
    }
    return;
  }
  ensure_gpu_gauge(gauges, family, gpu_index, label)->Set(value);
}

void
clear_missing_gauges(
    std::unordered_map<int, prometheus::Gauge*>& gauges,
    prometheus::Family<prometheus::Gauge>* family,
    const std::unordered_set<int>& seen_indices)
{
  std::erase_if(gauges, [family, &seen_indices](const auto& entry) {
    if (seen_indices.contains(entry.first)) {
      return false;
    }
    if (family != nullptr) {
      family->Remove(entry.second);
    }
    return true;
  });
}

auto
cpu_sampling_error_log_ts() -> std::atomic<std::int64_t>&
{
  static std::atomic<std::int64_t> timestamp{0};
  return timestamp;
}

auto
gpu_sampling_error_log_ts() -> std::atomic<std::int64_t>&
{
  static std::atomic<std::int64_t> timestamp{0};
  return timestamp;
}

auto
should_log_sampling_error(std::atomic<std::int64_t>& last_log) -> bool
{
  const auto now = std::chrono::duration_cast<std::chrono::seconds>(
                       std::chrono::steady_clock::now().time_since_epoch())
                       .count();
  auto prev = last_log.load(std::memory_order_relaxed);
  if (now - prev < kSamplingErrorLogThrottle.count()) {
    return false;
  }
  return last_log.compare_exchange_strong(prev, now, std::memory_order_relaxed);
}

#ifndef STARPU_HAVE_NVML
auto
nvml_warning_flag() -> std::once_flag&
{
  static std::once_flag flag;
  return flag;
}
#endif

auto
read_total_cpu_times(CpuTotals& out) -> bool
{
  static const std::filesystem::path kProcStat{"/proc/stat"};
  return monitoring::detail::read_total_cpu_times(kProcStat, out);
}

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
using ProcessSampleReader = std::function<std::optional<double>()>;
auto
process_open_fds_reader_override_storage() -> ProcessSampleReader&
{
  static ProcessSampleReader reader;
  return reader;
}

auto
process_rss_bytes_reader_override_storage() -> ProcessSampleReader&
{
  static ProcessSampleReader reader;
  return reader;
}
#endif  // SONAR_IGNORE_END

auto
read_process_rss_bytes_impl() -> std::optional<double>
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  auto& override_reader = process_rss_bytes_reader_override_storage();
  if (override_reader) {
    return override_reader();
  }
  const std::filesystem::path& path =
      monitoring::detail::process_rss_bytes_path_for_test();
#else
  const std::filesystem::path& path = kProcStatm;
#endif  // SONAR_IGNORE_END
  std::ifstream statm{path};
  if (!statm.is_open()) {
    return std::nullopt;
  }
  unsigned long size = 0;
  unsigned long resident = 0;
  if (!(statm >> size >> resident)) {
    return std::nullopt;
  }
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  const auto& page_size_provider =
      monitoring::detail::process_page_size_provider_for_test();
  const long page_size =
      page_size_provider ? page_size_provider() : sysconf(_SC_PAGESIZE);
#else
  const long page_size = sysconf(_SC_PAGESIZE);
#endif  // SONAR_IGNORE_END
  if (page_size <= 0) {
    return std::nullopt;
  }
  return static_cast<double>(resident) * static_cast<double>(page_size);
}

auto
read_process_open_fds_impl() -> std::optional<double>
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  auto& override_reader = process_open_fds_reader_override_storage();
  if (override_reader) {
    return override_reader();
  }
#endif  // SONAR_IGNORE_END
  static const std::filesystem::path kProcFd{"/proc/self/fd"};
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  const std::filesystem::path& path =
      monitoring::detail::process_fd_path_for_test();
#else
  const std::filesystem::path& path = kProcFd;
#endif  // SONAR_IGNORE_END
  try {
    if (!std::filesystem::exists(path)) {
      return std::nullopt;
    }
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
    const auto factory =
        monitoring::detail::process_fd_directory_iterator_for_test();
    std::filesystem::directory_iterator iter =
        factory ? factory(path) : std::filesystem::directory_iterator(path);
#else
    std::filesystem::directory_iterator iter(path);
#endif  // SONAR_IGNORE_END
    const std::filesystem::directory_iterator end{};
    std::size_t count = 0;
    for (; iter != end; ++iter) {
      ++count;
    }
    if (count > 0 && path == kProcFd) {
      --count;
    }
    return static_cast<double>(count);
  }
  catch (const std::exception&) {
    return std::nullopt;
  }
}

using GpuSample = MetricsRegistry::GpuSample;
using GpuStatsProvider = MetricsRegistry::GpuStatsProvider;
using CpuUsageProvider = MetricsRegistry::CpuUsageProvider;

#ifdef STARPU_HAVE_NVML

class NvmlWrapper {
 public:
  static auto instance() -> NvmlWrapper&;

  auto query_stats() -> std::vector<GpuSample>;

  NvmlWrapper(const NvmlWrapper&) = delete;
  NvmlWrapper(NvmlWrapper&&) = delete;
  auto operator=(const NvmlWrapper&) -> NvmlWrapper& = delete;
  auto operator=(NvmlWrapper&&) -> NvmlWrapper& = delete;

 private:
  NvmlWrapper();
  ~NvmlWrapper();

  static auto error_string(nvmlReturn_t status) -> const char*;

  bool initialized_{false};
  std::mutex mutex_;
};

auto
NvmlWrapper::instance() -> NvmlWrapper&
{
  static NvmlWrapper wrapper;
  return wrapper;
}

NvmlWrapper::NvmlWrapper()
{
  if (const nvmlReturn_t status = nvmlInit(); status != NVML_SUCCESS) {
    log_warning(std::format(
        "Failed to initialize NVML: {} (code {})",
        std::string(error_string(status)), static_cast<int>(status)));
    return;
  }
  initialized_ = true;
}

NvmlWrapper::~NvmlWrapper()
{
  if (initialized_) {
    nvmlShutdown();
  }
}

auto
NvmlWrapper::error_string(nvmlReturn_t status) -> const char*
{
  const char* err = nvmlErrorString(status);
  return err != nullptr ? err : "unknown error";
}

auto
NvmlWrapper::query_stats() -> std::vector<GpuSample>
{
  std::scoped_lock<std::mutex> guard(mutex_);
  if (!initialized_) {
    return {};
  }

  unsigned int device_count = 0;
  nvmlReturn_t status = nvmlDeviceGetCount(&device_count);
  if (status != NVML_SUCCESS) {
    log_warning(
        std::string("nvmlDeviceGetCount failed: ") + error_string(status));
    return {};
  }

  std::vector<GpuSample> stats;
  stats.reserve(device_count);

  for (unsigned int idx = 0; idx < device_count; ++idx) {
    nvmlDevice_t device{};
    status = nvmlDeviceGetHandleByIndex(idx, &device);
    if (status != NVML_SUCCESS) {
      log_warning(std::format(
          "nvmlDeviceGetHandleByIndex failed for GPU {}: {} (code {})", idx,
          std::string(error_string(status)), static_cast<int>(status)));
      continue;
    }

    nvmlUtilization_t utilization{};
    status = nvmlDeviceGetUtilizationRates(device, &utilization);
    if (status != NVML_SUCCESS) {
      log_warning(std::format(
          "nvmlDeviceGetUtilizationRates failed for GPU {}: {} (code {})", idx,
          std::string(error_string(status)), static_cast<int>(status)));
      continue;
    }

    nvmlMemory_t memory_info{};
    status = nvmlDeviceGetMemoryInfo(device, &memory_info);
    if (status != NVML_SUCCESS) {
      log_warning(std::format(
          "nvmlDeviceGetMemoryInfo failed for GPU {}: {} (code {})", idx,
          std::string(error_string(status)), static_cast<int>(status)));
      continue;
    }

    GpuSample stat;
    stat.index = static_cast<int>(idx);
    stat.util_percent = static_cast<double>(utilization.gpu);
    stat.mem_used_bytes = static_cast<double>(memory_info.used);
    stat.mem_total_bytes = static_cast<double>(memory_info.total);
    unsigned int temperature = 0;
    status =
        nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature);
    if (status == NVML_SUCCESS) {
      stat.temperature_celsius = static_cast<double>(temperature);
    } else {
      log_warning(std::format(
          "nvmlDeviceGetTemperature failed for GPU {}: {} (code {})", idx,
          std::string(error_string(status)), static_cast<int>(status)));
    }

    unsigned int power_mw = 0;
    status = nvmlDeviceGetPowerUsage(device, &power_mw);
    if (status == NVML_SUCCESS) {
      constexpr double milliwatts_per_watt = 1000.0;
      stat.power_watts = static_cast<double>(power_mw) / milliwatts_per_watt;
    } else {
      log_warning(std::format(
          "nvmlDeviceGetPowerUsage failed for GPU {}: {} (code {})", idx,
          std::string(error_string(status)), static_cast<int>(status)));
    }
    stats.push_back(stat);
  }

  return stats;
}

auto
query_gpu_stats_nvml() -> std::vector<GpuSample>
{
  return NvmlWrapper::instance().query_stats();
}

#else

auto
query_gpu_stats_nvml() -> std::vector<GpuSample>
{
  return {};
}

#endif  // STARPU_HAVE_NVML
}  // namespace

namespace monitoring::detail {

auto
cpu_usage_percent(const CpuTotals& prev, const CpuTotals& curr) -> double
{
  const unsigned long long prev_idle = prev.idle + prev.iowait;
  const unsigned long long curr_idle = curr.idle + curr.iowait;
  const unsigned long long prev_non_idle = prev.user + prev.nice + prev.system +
                                           prev.irq + prev.softirq + prev.steal;
  const unsigned long long curr_non_idle = curr.user + curr.nice + curr.system +
                                           curr.irq + curr.softirq + curr.steal;

  const unsigned long long prev_total = prev_idle + prev_non_idle;
  const unsigned long long curr_total = curr_idle + curr_non_idle;

  const long long totald_signed =
      curr_total >= prev_total
          ? static_cast<long long>(curr_total - prev_total)
          : -static_cast<long long>(prev_total - curr_total);
  const long long idled_signed =
      curr_idle >= prev_idle ? static_cast<long long>(curr_idle - prev_idle)
                             : -static_cast<long long>(prev_idle - curr_idle);

  const auto totald = static_cast<double>(totald_signed);
  const auto idled = static_cast<double>(idled_signed);
  if (totald <= 0.0) {
    return 0.0;
  }
  const double usage = (totald - idled) / totald * 100.0;
  if (usage < 0.0) {
    return 0.0;
  }
  if (usage > 100.0) {
    return 100.0;
  }
  return usage;
}

auto
make_cpu_usage_provider(std::function<bool(CpuTotals&)> reader)
    -> MetricsRegistry::CpuUsageProvider
{
  CpuTotals prev_cpu{};
  bool have_prev_cpu = reader(prev_cpu);

  return [reader_fn = std::move(reader), prev_cpu,
          have_prev_cpu]() mutable -> std::optional<double> {
    CpuTotals cur_cpu{};
    if (!reader_fn(cur_cpu)) {
      return std::nullopt;
    }
    std::optional<double> usage{};
    if (have_prev_cpu) {
      usage = cpu_usage_percent(prev_cpu, cur_cpu);
    }
    prev_cpu = cur_cpu;
    have_prev_cpu = true;
    return usage;
  };
}

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
void
set_process_open_fds_reader_override(
    std::function<std::optional<double>()> reader)
{
  process_open_fds_reader_override_storage() = std::move(reader);
}

void
set_process_rss_bytes_reader_override(
    std::function<std::optional<double>()> reader)
{
  process_rss_bytes_reader_override_storage() = std::move(reader);
}

auto
process_fd_path_override_storage() -> std::optional<std::filesystem::path>&
{
  static std::optional<std::filesystem::path> path;
  return path;
}

auto
process_fd_directory_iterator_factory_storage()
    -> ProcessFdDirectoryIteratorFactory&
{
  static ProcessFdDirectoryIteratorFactory factory;
  return factory;
}

void
set_process_fd_path_for_test(std::filesystem::path path)
{
  process_fd_path_override_storage() = std::move(path);
}

void
reset_process_fd_path_for_test()
{
  process_fd_path_override_storage().reset();
}

auto
process_fd_path_for_test() -> const std::filesystem::path&
{
  static const std::filesystem::path kDefaultProcFd{"/proc/self/fd"};
  auto& override_path = process_fd_path_override_storage();
  if (override_path.has_value()) {
    return *override_path;
  }
  return kDefaultProcFd;
}

void
set_process_fd_directory_iterator_for_test(
    ProcessFdDirectoryIteratorFactory factory)
{
  process_fd_directory_iterator_factory_storage() = std::move(factory);
}

void
reset_process_fd_directory_iterator_for_test()
{
  process_fd_directory_iterator_factory_storage() = nullptr;
}

auto
process_fd_directory_iterator_for_test() -> ProcessFdDirectoryIteratorFactory
{
  return process_fd_directory_iterator_factory_storage();
}

auto
process_rss_bytes_path_override_storage()
    -> std::optional<std::filesystem::path>&
{
  static std::optional<std::filesystem::path> path;
  return path;
}

auto
process_page_size_provider_storage() -> ProcessPageSizeProvider&
{
  static ProcessPageSizeProvider provider;
  return provider;
}

void
set_process_rss_bytes_path_for_test(std::filesystem::path path)
{
  process_rss_bytes_path_override_storage() = std::move(path);
}

void
reset_process_rss_bytes_path_for_test()
{
  process_rss_bytes_path_override_storage().reset();
}

auto
process_rss_bytes_path_for_test() -> const std::filesystem::path&
{
  auto& override_path = process_rss_bytes_path_override_storage();
  if (override_path.has_value()) {
    return *override_path;
  }
  return kProcStatm;
}

void
set_process_page_size_provider_for_test(ProcessPageSizeProvider provider)
{
  process_page_size_provider_storage() = std::move(provider);
}

void
reset_process_page_size_provider_for_test()
{
  process_page_size_provider_storage() = nullptr;
}

auto
process_page_size_provider_for_test() -> const ProcessPageSizeProvider&
{
  return process_page_size_provider_storage();
}

auto
should_log_sampling_error_for_test(std::atomic<std::int64_t>& last_log) -> bool
{
  return should_log_sampling_error(last_log);
}

auto
status_code_label_for_test(int code) -> std::string
{
  return status_code_label(code);
}
#endif  // SONAR_IGNORE_END

auto
read_process_open_fds() -> std::optional<double>
{
  return read_process_open_fds_impl();
}

auto
read_process_rss_bytes() -> std::optional<double>
{
  return read_process_rss_bytes_impl();
}

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
auto
metrics_init_failure_flag() -> std::atomic<bool>&
{
  static std::atomic<bool> flag{false};
  return flag;
}

auto
metrics_request_stop_skip_join_flag() -> std::atomic<bool>&
{
  static std::atomic<bool> flag{false};
  return flag;
}

void
set_metrics_init_failure_for_test(bool fail)
{
  metrics_init_failure_flag().store(fail, std::memory_order_release);
}

auto
metrics_init_failure_for_test() -> bool
{
  return metrics_init_failure_flag().load(std::memory_order_acquire);
}

void
set_metrics_request_stop_skip_join_for_test(bool skip_join)
{
  metrics_request_stop_skip_join_flag().store(
      skip_join, std::memory_order_release);
}

auto
metrics_request_stop_skip_join_for_test() -> bool
{
  return metrics_request_stop_skip_join_flag().load(std::memory_order_acquire);
}
#endif  // SONAR_IGNORE_END

}  // namespace monitoring::detail

auto
make_default_cpu_usage_provider(
    std::function<bool(monitoring::detail::CpuTotals&)> reader)
    -> CpuUsageProvider
{
  return monitoring::detail::make_cpu_usage_provider(std::move(reader));
}

auto
make_default_cpu_usage_provider() -> CpuUsageProvider
{
  return make_default_cpu_usage_provider(
      [](monitoring::detail::CpuTotals& totals) {
        return read_total_cpu_times(totals);
      });
}

namespace {

auto
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

auto
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

auto
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

void
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

void
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

void
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

void
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

void
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
}  // namespace

class MetricsRegistry::Sampler {
 public:
  explicit Sampler(MetricsRegistry& registry) : registry_(&registry) {}

  void run_sampling_request_nb();
  void sampling_loop(const std::stop_token& stop);
  void sample_process_open_fds();
  void sample_process_resident_memory();
  void sample_inference_throughput();

 private:
  void sample_cpu_usage();
  void sample_gpu_stats();
  void perform_sampling_request_nb();

  MetricsRegistry* registry_;
};

MetricsRegistry::MetricsRegistry(int port)
    : MetricsRegistry(
          port, query_gpu_stats_nvml, make_default_cpu_usage_provider())
{
}

MetricsRegistry::MetricsRegistry(
    int port, GpuStatsProvider gpu_provider, CpuUsageProvider cpu_provider,
    bool start_sampler_thread, std::unique_ptr<ExposerHandle> exposer_handle)
    : registry_state_{std::make_shared<prometheus::Registry>()},
      providers_{std::move(gpu_provider), std::move(cpu_provider)}
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (monitoring::detail::metrics_init_failure_for_test()) {
    throw std::runtime_error("forced metrics initialization failure");
  }
#endif  // SONAR_IGNORE_END
  if (!providers_.gpu_stats_provider) {
    providers_.gpu_stats_provider = query_gpu_stats_nvml;
  }
  if (!providers_.cpu_usage_provider) {
    providers_.cpu_usage_provider = make_default_cpu_usage_provider();
  }
  sampler_ = std::make_unique<Sampler>(*this);
  initialize(port, start_sampler_thread, std::move(exposer_handle));
}

void
MetricsRegistry::initialize(
    [[maybe_unused]] int port, bool start_sampler_thread,
    std::unique_ptr<ExposerHandle> exposer_handle)
{
  try {
    if (!exposer_handle) {
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
      exposer_handle = std::make_unique<TestingExposerHandle>();
#else
      auto exposer = std::make_unique<prometheus::Exposer>(
          std::format("0.0.0.0:{}", port));
      exposer_handle =
          std::make_unique<PrometheusExposerHandle>(std::move(exposer));
#endif  // SONAR_IGNORE_END
    }
    exposer_handle->RegisterCollectable(registry_state_.registry);
    registry_state_.exposer = std::move(exposer_handle);
  }
  catch (const std::exception& e) {
    log_error(std::string("Failed to initialize metrics exposer: ") + e.what());
    throw;
  }

  auto& registry = *registry_state_.registry;
  register_request_counters_and_families(registry, counters_, families_);
  register_queue_runtime_and_batching_metrics(registry, gauges_, histograms_);
  register_congestion_gauges(registry, gauges_.congestion);
  register_latency_histograms(registry, histograms_);
  register_model_gpu_and_worker_metrics(registry, histograms_, families_);

  if (start_sampler_thread) {
    registry_state_.sampler_thread = std::jthread(
        [this](const std::stop_token& stop) { sampler_->sampling_loop(stop); });
  }
}

MetricsRegistry::~MetricsRegistry() noexcept
{
  request_stop();
  if (registry_state_.sampler_thread.joinable()) {
    registry_state_.sampler_thread.join();
  }
  if (registry_state_.exposer && registry_state_.registry) {
    try {
      registry_state_.exposer->RemoveCollectable(registry_state_.registry);
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

auto
metrics_init_mutex() -> std::mutex&
{
  static std::mutex mutex;
  return mutex;
}

auto
metrics_shutdown_once_flag() -> std::once_flag&
{
  static std::once_flag flag;
  return flag;
}

template <typename Fn>
void
with_metrics_registry(Fn&& callback)
{
  auto metrics_ptr = metrics_atomic().load(std::memory_order_acquire);
  if (metrics_ptr != nullptr) {
    std::forward<Fn>(callback)(*metrics_ptr);
  }
}

void
set_gauge_if_present(prometheus::Gauge* gauge, double value)
{
  if (gauge != nullptr) {
    gauge->Set(value);
  }
}

void
increment_counter_if_present(prometheus::Counter* counter, double value = 1.0)
{
  if (counter != nullptr) {
    counter->Increment(value);
  }
}

void
observe_histogram_if_non_negative(
    prometheus::Histogram* histogram, double value)
{
  if (histogram != nullptr && value >= 0.0) {
    histogram->Observe(value);
  }
}

template <typename Key, typename Map, typename Factory>
auto
get_or_create_cached_series(Map& cache, Key lookup_key, Factory&& create_metric)
    -> typename Map::mapped_type
{
  auto entry = cache.find(lookup_key);
  if (entry == cache.end()) {
    const bool overflow = cache.size() >= kMaxLabelSeries;
    Key map_key = overflow ? Key::Overflow() : std::move(lookup_key);
    auto [inserted_it, inserted] =
        cache.try_emplace(std::move(map_key), nullptr);
    entry = inserted_it;
    if (inserted) {
      entry->second = create_metric(overflow);
    }
  }
  return entry->second;
}

struct WorkerMetricLabels {
  std::string worker_id;
  std::string device;
  std::string worker_type;
};

auto
make_worker_metric_labels(
    bool overflow, int worker_id, int device_id,
    std::string_view worker_type) -> WorkerMetricLabels
{
  if (overflow) {
    return {kOverflowLabel, kOverflowLabel, kOverflowLabel};
  }
  return {
      std::to_string(worker_id), std::to_string(device_id),
      escape_label_value(worker_type)};
}

struct IoMetricLabels {
  std::string direction;
  WorkerMetricLabels worker;
};

auto
make_io_metric_labels(
    bool overflow, std::string_view direction, int worker_id, int device_id,
    std::string_view worker_type) -> IoMetricLabels
{
  if (overflow) {
    return {kOverflowLabel, {kOverflowLabel, kOverflowLabel, kOverflowLabel}};
  }
  return {
      escape_label_value(direction),
      make_worker_metric_labels(false, worker_id, device_id, worker_type)};
}
}  // namespace

auto
init_metrics(int port) -> bool
{
  std::scoped_lock<std::mutex> lock(metrics_init_mutex());
  if (metrics_atomic().load(std::memory_order_acquire) != nullptr) {
    log_warning("Metrics were previously initialized");
    return false;
  }

  try {
    auto new_metrics = std::make_shared<MetricsRegistry>(port);
    std::call_once(
        metrics_shutdown_once_flag(), [] { std::atexit(shutdown_metrics); });

#ifndef STARPU_HAVE_NVML
    std::call_once(nvml_warning_flag(), [] {
      log_warning_critical(
          "NVML support is not available; GPU metrics collection is "
          "disabled.");
    });
#endif

    metrics_atomic().store(new_metrics, std::memory_order_release);

    set_queue_size(0);
    set_inflight_tasks(0);
    set_max_inflight_tasks(0);
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
  std::scoped_lock<std::mutex> lock(metrics_init_mutex());
  auto metrics_ptr = metrics_atomic().load(std::memory_order_acquire);
  if (metrics_ptr) {
    metrics_ptr->request_stop();
  }
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
  with_metrics_registry([size](MetricsRegistry& metrics) {
    auto& gauges = metrics.gauges();
    set_gauge_if_present(gauges.queue_size, static_cast<double>(size));
    const auto capacity = metrics.queue_capacity_value();
    if (capacity > 0 && gauges.queue_fill_ratio != nullptr) {
      const double ratio =
          static_cast<double>(size) / static_cast<double>(capacity);
      gauges.queue_fill_ratio->Set(std::clamp(ratio, 0.0, 1.0));
    }
  });
}

void
set_inflight_tasks(std::size_t size)
{
  with_metrics_registry([size](MetricsRegistry& metrics) {
    set_gauge_if_present(
        metrics.gauges().inflight_tasks, static_cast<double>(size));
  });
}

void
set_starpu_worker_busy_ratio(double ratio)
{
  with_metrics_registry([ratio](MetricsRegistry& metrics) {
    set_gauge_if_present(
        metrics.gauges().starpu_worker_busy_ratio, std::clamp(ratio, 0.0, 1.0));
  });
}

void
set_max_inflight_tasks(std::size_t max_tasks)
{
  with_metrics_registry([max_tasks](MetricsRegistry& metrics) {
    set_gauge_if_present(
        metrics.gauges().max_inflight_tasks, static_cast<double>(max_tasks));
  });
}

void
set_queue_capacity(std::size_t capacity)
{
  with_metrics_registry([capacity](MetricsRegistry& metrics) {
    auto& gauges = metrics.gauges();
    metrics.set_queue_capacity(capacity);
    set_gauge_if_present(gauges.queue_capacity, static_cast<double>(capacity));
    if (gauges.queue_fill_ratio == nullptr) {
      return;
    }
    if (capacity > 0 && gauges.queue_size != nullptr) {
      const double size = gauges.queue_size->Value();
      gauges.queue_fill_ratio->Set(
          std::clamp(size / static_cast<double>(capacity), 0.0, 1.0));
      return;
    }
    gauges.queue_fill_ratio->Set(0.0);
  });
}

// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
void
set_queue_fill_ratio(std::size_t size, std::size_t capacity)
{
  with_metrics_registry([size, capacity](MetricsRegistry& metrics) {
    if (capacity == 0) {
      return;
    }
    if (auto* gauge = metrics.gauges().queue_fill_ratio; gauge != nullptr) {
      gauge->Set(static_cast<double>(size) / static_cast<double>(capacity));
    }
  });
}
#endif  // SONAR_IGNORE_END
// GCOVR_EXCL_STOP

void
set_starpu_prepared_queue_depth(std::size_t depth)
{
  with_metrics_registry([depth](MetricsRegistry& metrics) {
    set_gauge_if_present(
        metrics.gauges().starpu_prepared_queue_depth,
        static_cast<double>(depth));
  });
}

void
set_batch_pending_jobs(std::size_t pending)
{
  with_metrics_registry([pending](MetricsRegistry& metrics) {
    set_gauge_if_present(
        metrics.gauges().batch_pending_jobs, static_cast<double>(pending));
  });
}

void
increment_rejected_requests()
{
  with_metrics_registry([](MetricsRegistry& metrics) {
    increment_counter_if_present(metrics.counters().requests_rejected_total);
  });
}

void
set_congestion_flag(bool congested)
{
  with_metrics_registry([congested](MetricsRegistry& metrics) {
    set_gauge_if_present(
        metrics.gauges().congestion.flag, congested ? 1.0 : 0.0);
  });
}

void
set_congestion_score(double score)
{
  with_metrics_registry([score](MetricsRegistry& metrics) {
    set_gauge_if_present(
        metrics.gauges().congestion.score, std::clamp(score, 0.0, 1.0));
  });
}

void
set_congestion_arrival_rate(double rps)
{
  with_metrics_registry([rps](MetricsRegistry& metrics) {
    set_gauge_if_present(
        metrics.gauges().congestion.lambda_rps, std::max(0.0, rps));
  });
}

void
set_congestion_completion_rate(double rps)
{
  with_metrics_registry([rps](MetricsRegistry& metrics) {
    set_gauge_if_present(
        metrics.gauges().congestion.mu_rps, std::max(0.0, rps));
  });
}

void
set_congestion_rejection_rate(double rps)
{
  with_metrics_registry([rps](MetricsRegistry& metrics) {
    set_gauge_if_present(
        metrics.gauges().congestion.rejection_rps, std::max(0.0, rps));
  });
}

void
set_congestion_rho(double rho)
{
  with_metrics_registry([rho](MetricsRegistry& metrics) {
    const double value = std::isfinite(rho) ? rho : 0.0;
    set_gauge_if_present(
        metrics.gauges().congestion.rho_ewma, std::max(0.0, value));
  });
}

void
set_congestion_fill_ewma(double fill)
{
  with_metrics_registry([fill](MetricsRegistry& metrics) {
    set_gauge_if_present(
        metrics.gauges().congestion.queue_fill_ewma,
        std::clamp(fill, 0.0, 1.0));
  });
}

void
set_congestion_queue_growth_rate(double rate)
{
  with_metrics_registry([rate](MetricsRegistry& metrics) {
    set_gauge_if_present(metrics.gauges().congestion.queue_growth_rate, rate);
  });
}

void
set_congestion_queue_latency_p95(double latency_ms)
{
  with_metrics_registry([latency_ms](MetricsRegistry& metrics) {
    set_gauge_if_present(
        metrics.gauges().congestion.queue_p95_ms, std::max(0.0, latency_ms));
  });
}

void
set_congestion_queue_latency_p99(double latency_ms)
{
  with_metrics_registry([latency_ms](MetricsRegistry& metrics) {
    set_gauge_if_present(
        metrics.gauges().congestion.queue_p99_ms, std::max(0.0, latency_ms));
  });
}

void
set_congestion_e2e_latency_p95(double latency_ms)
{
  with_metrics_registry([latency_ms](MetricsRegistry& metrics) {
    set_gauge_if_present(
        metrics.gauges().congestion.e2e_p95_ms, std::max(0.0, latency_ms));
  });
}

void
set_congestion_e2e_latency_p99(double latency_ms)
{
  with_metrics_registry([latency_ms](MetricsRegistry& metrics) {
    set_gauge_if_present(
        metrics.gauges().congestion.e2e_p99_ms, std::max(0.0, latency_ms));
  });
}

void
set_server_health(bool ready)
{
  with_metrics_registry([ready](MetricsRegistry& metrics) {
    set_gauge_if_present(
        metrics.gauges().server_health_state, ready ? 1.0 : 0.0);
  });
}

void
increment_request_status(int status_code, std::string_view model_name)
{
  with_metrics_registry([status_code, model_name](MetricsRegistry& metrics) {
    metrics.increment_status_counter(
        MetricsRegistry::StatusCodeLabel{status_code_label(status_code)},
        MetricsRegistry::ModelLabel{model_name});
  });
}

void
increment_requests_received(std::string_view model_name)
{
  with_metrics_registry([model_name](MetricsRegistry& metrics) {
    metrics.increment_received_counter(model_name);
  });
}

void
observe_batch_size(std::size_t batch_size)
{
  with_metrics_registry([batch_size](MetricsRegistry& metrics) {
    observe_histogram_if_non_negative(
        metrics.histograms().batch_size, static_cast<double>(batch_size));
  });
}

void
observe_logical_batch_size(std::size_t logical_jobs)
{
  with_metrics_registry([logical_jobs](MetricsRegistry& metrics) {
    observe_histogram_if_non_negative(
        metrics.histograms().logical_batch_size,
        static_cast<double>(logical_jobs));
  });
}

void
observe_batch_efficiency(double ratio)
{
  with_metrics_registry([ratio](MetricsRegistry& metrics) {
    observe_histogram_if_non_negative(
        metrics.histograms().batch_efficiency, ratio);
  });
}

void
observe_latency_breakdown(const LatencyBreakdownMetrics& breakdown)
{
  with_metrics_registry([&breakdown](MetricsRegistry& metrics) {
    auto& histograms = metrics.histograms();
    observe_histogram_if_non_negative(
        histograms.queue_latency, breakdown.queue_ms);
    observe_histogram_if_non_negative(
        histograms.batch_collect_latency, breakdown.batch_ms);
    observe_histogram_if_non_negative(
        histograms.submit_latency, breakdown.submit_ms);
    observe_histogram_if_non_negative(
        histograms.scheduling_latency, breakdown.scheduling_ms);
    observe_histogram_if_non_negative(
        histograms.codelet_latency, breakdown.codelet_ms);
    observe_histogram_if_non_negative(
        histograms.inference_compute_latency, breakdown.inference_ms);
    observe_histogram_if_non_negative(
        histograms.callback_latency, breakdown.callback_ms);
    observe_histogram_if_non_negative(
        histograms.preprocess_latency, breakdown.preprocess_ms);
    observe_histogram_if_non_negative(
        histograms.postprocess_latency, breakdown.postprocess_ms);
  });
}

void
observe_starpu_task_runtime(double runtime_ms)
{
  with_metrics_registry([runtime_ms](MetricsRegistry& metrics) {
    observe_histogram_if_non_negative(
        metrics.histograms().starpu_task_runtime, runtime_ms);
  });
}

void
observe_model_load_duration(double duration_ms)
{
  with_metrics_registry([duration_ms](MetricsRegistry& metrics) {
    observe_histogram_if_non_negative(
        metrics.histograms().model_load_duration, duration_ms);
  });
}

void
set_model_loaded(
    std::string_view model_name, std::string_view device_label, bool loaded)
{
  with_metrics_registry(
      [model_name, device_label, loaded](MetricsRegistry& metrics) {
        metrics.set_model_loaded_flag(
            MetricsRegistry::ModelLabel{model_name},
            MetricsRegistry::DeviceLabel{device_label}, loaded);
      });
}

void
increment_model_load_failure(std::string_view model_name)
{
  with_metrics_registry([model_name](MetricsRegistry& metrics) {
    metrics.increment_model_load_failure_counter(model_name);
  });
}

void
observe_compute_latency_by_worker(
    int worker_id, int device_id, std::string_view worker_type,
    double latency_ms)
{
  with_metrics_registry([worker_id, device_id, worker_type,
                         latency_ms](MetricsRegistry& metrics) {
    metrics.observe_compute_latency_by_worker(
        worker_id, device_id, worker_type, latency_ms);
  });
}

void
observe_task_runtime_by_worker(
    int worker_id, int device_id, std::string_view worker_type,
    double latency_ms)
{
  with_metrics_registry([worker_id, device_id, worker_type,
                         latency_ms](MetricsRegistry& metrics) {
    metrics.observe_task_runtime_by_worker(
        worker_id, device_id, worker_type, latency_ms);
  });
}

void
set_worker_inflight_gauge(
    int worker_id, int device_id, std::string_view worker_type,
    std::size_t value)
{
  with_metrics_registry([worker_id, device_id, worker_type,
                         value](MetricsRegistry& metrics) {
    metrics.set_worker_inflight_gauge(worker_id, device_id, worker_type, value);
  });
}

void
observe_io_copy_latency(
    std::string_view direction, int worker_id, int device_id,
    std::string_view worker_type, double duration_ms)
{
  with_metrics_registry([direction, worker_id, device_id, worker_type,
                         duration_ms](MetricsRegistry& metrics) {
    metrics.observe_io_copy_latency(
        direction, worker_id, device_id, worker_type, duration_ms);
  });
}

void
increment_transfer_bytes(
    std::string_view direction, int worker_id, int device_id,
    std::string_view worker_type, std::size_t bytes)
{
  with_metrics_registry([direction, worker_id, device_id, worker_type,
                         bytes](MetricsRegistry& metrics) {
    metrics.increment_transfer_bytes(
        direction, worker_id, device_id, worker_type, bytes);
  });
}

void
MetricsRegistry::request_stop()
{
  if (registry_state_.sampler_thread.joinable()) {
    registry_state_.sampler_thread.request_stop();
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
    if (monitoring::detail::metrics_request_stop_skip_join_for_test()) {
      return;
    }
#endif  // SONAR_IGNORE_END
    registry_state_.sampler_thread.join();
  }
}

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
auto
MetricsRegistry::has_gpu_stats_provider() const -> bool
{
  return static_cast<bool>(providers_.gpu_stats_provider);
}

auto
MetricsRegistry::has_cpu_usage_provider() const -> bool
{
  return static_cast<bool>(providers_.cpu_usage_provider);
}
#endif  // SONAR_IGNORE_END

auto
MetricsRegistry::registry() const -> std::shared_ptr<prometheus::Registry>
{
  return registry_state_.registry;
}

void
MetricsRegistry::increment_status_counter(
    MetricsRegistry::StatusCodeLabel code_label,
    MetricsRegistry::ModelLabel model_label)
{
  if (families_.requests_by_status == nullptr) {
    return;
  }

  StatusKey key{std::string(code_label.value), std::string(model_label.value)};

  std::scoped_lock lock(mutexes_.status);
  auto* counter = get_or_create_cached_series(
      caches_.status.counters, std::move(key),
      [this, code_label, model_label](bool overflow) -> prometheus::Counter* {
        const std::string code_label_value =
            overflow ? kOverflowLabel : escape_label_value(code_label.value);
        const std::string model_label_value =
            overflow ? kOverflowLabel : escape_label_value(model_label.value);
        return &families_.requests_by_status->Add(
            {{"code", code_label_value}, {"model", model_label_value}});
      });
  increment_counter_if_present(counter);
}

void
MetricsRegistry::increment_completed_counter(
    std::string_view model_label, std::size_t logical_jobs)
{
  if (families_.inference_completed == nullptr) {
    return;
  }
  std::scoped_lock lock(mutexes_.model_metrics);
  auto* counter = get_or_create_cached_series(
      caches_.model.inference_completed, ModelKey{std::string(model_label)},
      [this, model_label](bool overflow) -> prometheus::Counter* {
        const std::string model_label_value =
            overflow ? kOverflowLabel : escape_label_value(model_label);
        return &families_.inference_completed->Add(
            {{"model", model_label_value}});
      });
  increment_counter_if_present(counter, static_cast<double>(logical_jobs));
}

void
MetricsRegistry::increment_received_counter(std::string_view model_label)
{
  if (families_.requests_received == nullptr) {
    return;
  }
  std::scoped_lock lock(mutexes_.model_metrics);
  auto* counter = get_or_create_cached_series(
      caches_.model.requests_received, ModelKey{std::string(model_label)},
      [this, model_label](bool overflow) -> prometheus::Counter* {
        const std::string model_label_value =
            overflow ? kOverflowLabel : escape_label_value(model_label);
        return &families_.requests_received->Add(
            {{"model", model_label_value}});
      });
  increment_counter_if_present(counter);
}

void
MetricsRegistry::increment_failure_counter(
    MetricsRegistry::FailureStageLabel stage_label,
    MetricsRegistry::FailureReasonLabel reason_label,
    MetricsRegistry::ModelLabel model_label, std::size_t count)
{
  if (families_.inference_failures == nullptr) {
    return;
  }
  FailureKey key{
      std::string(stage_label.value), std::string(reason_label.value),
      std::string(model_label.value)};

  std::scoped_lock lock(mutexes_.model_metrics);
  auto* counter = get_or_create_cached_series(
      caches_.model.inference_failures, std::move(key),
      [this, stage_label, reason_label,
       model_label](bool overflow) -> prometheus::Counter* {
        const std::string stage_label_value =
            overflow ? kOverflowLabel : escape_label_value(stage_label.value);
        const std::string reason_label_value =
            overflow ? kOverflowLabel : escape_label_value(reason_label.value);
        const std::string model_label_value =
            overflow ? kOverflowLabel : escape_label_value(model_label.value);
        return &families_.inference_failures->Add(
            {{"stage", stage_label_value},
             {"reason", reason_label_value},
             {"model", model_label_value}});
      });
  increment_counter_if_present(counter, static_cast<double>(count));
}

void
MetricsRegistry::increment_model_load_failure_counter(
    std::string_view model_label)
{
  if (families_.model_load_failures == nullptr) {
    return;
  }
  std::scoped_lock lock(mutexes_.model_metrics);
  auto* counter = get_or_create_cached_series(
      caches_.model.model_load_failures, ModelKey{std::string(model_label)},
      [this, model_label](bool overflow) -> prometheus::Counter* {
        const std::string model_label_value =
            overflow ? kOverflowLabel : escape_label_value(model_label);
        return &families_.model_load_failures->Add(
            {{"model", model_label_value}});
      });
  increment_counter_if_present(counter);
}

void
MetricsRegistry::set_model_loaded_flag(
    MetricsRegistry::ModelLabel model_label,
    MetricsRegistry::DeviceLabel device_label, bool loaded)
{
  if (families_.models_loaded == nullptr) {
    return;
  }
  ModelDeviceKey key{
      std::string(model_label.value), std::string(device_label.value)};

  std::scoped_lock lock(mutexes_.model_metrics);
  auto* gauge = get_or_create_cached_series(
      caches_.model.models_loaded, std::move(key),
      [this, model_label, device_label](bool overflow) -> prometheus::Gauge* {
        const std::string model_label_value =
            overflow ? kOverflowLabel : escape_label_value(model_label.value);
        const std::string device_label_value =
            overflow ? kOverflowLabel : escape_label_value(device_label.value);
        return &families_.models_loaded->Add(
            {{"model", model_label_value}, {"device", device_label_value}});
      });
  set_gauge_if_present(gauge, loaded ? 1.0 : 0.0);
}

void
MetricsRegistry::observe_compute_latency_by_worker(
    int worker_id, int device_id, std::string_view worker_type,
    double latency_ms)
{
  if (latency_ms < 0.0 ||
      families_.inference_compute_latency_by_worker == nullptr) {
    return;
  }
  WorkerKey key{worker_id, device_id, std::string(worker_type)};
  std::scoped_lock lock(mutexes_.worker_metrics);
  auto* histogram = get_or_create_cached_series(
      caches_.worker.compute_latency, std::move(key),
      [this, worker_id, device_id,
       worker_type](bool overflow) -> prometheus::Histogram* {
        const auto labels = make_worker_metric_labels(
            overflow, worker_id, device_id, worker_type);
        return &families_.inference_compute_latency_by_worker->Add(
            {{"worker_id", labels.worker_id},
             {"device", labels.device},
             {"worker_type", labels.worker_type}},
            kInferenceLatencyMsBuckets);
      });
  observe_histogram_if_non_negative(histogram, latency_ms);
}

void
MetricsRegistry::observe_task_runtime_by_worker(
    int worker_id, int device_id, std::string_view worker_type,
    double latency_ms)
{
  if (latency_ms < 0.0 || families_.starpu_task_runtime_by_worker == nullptr) {
    return;
  }
  WorkerKey key{worker_id, device_id, std::string(worker_type)};
  std::scoped_lock lock(mutexes_.worker_metrics);
  auto* histogram = get_or_create_cached_series(
      caches_.worker.task_runtime, std::move(key),
      [this, worker_id, device_id,
       worker_type](bool overflow) -> prometheus::Histogram* {
        const auto labels = make_worker_metric_labels(
            overflow, worker_id, device_id, worker_type);
        return &families_.starpu_task_runtime_by_worker->Add(
            {{"worker_id", labels.worker_id},
             {"device", labels.device},
             {"worker_type", labels.worker_type}},
            kTaskRuntimeMsBuckets);
      });
  observe_histogram_if_non_negative(histogram, latency_ms);
}

void
MetricsRegistry::set_worker_inflight_gauge(
    int worker_id, int device_id, std::string_view worker_type,
    std::size_t value)
{
  if (families_.starpu_worker_inflight == nullptr) {
    return;
  }
  WorkerKey key{worker_id, device_id, std::string(worker_type)};
  std::scoped_lock lock(mutexes_.worker_metrics);
  auto* gauge = get_or_create_cached_series(
      caches_.worker.inflight, std::move(key),
      [this, worker_id, device_id,
       worker_type](bool overflow) -> prometheus::Gauge* {
        const auto labels = make_worker_metric_labels(
            overflow, worker_id, device_id, worker_type);
        return &families_.starpu_worker_inflight->Add(
            {{"worker_id", labels.worker_id},
             {"device", labels.device},
             {"worker_type", labels.worker_type}});
      });
  set_gauge_if_present(gauge, static_cast<double>(value));
}

void
MetricsRegistry::observe_io_copy_latency(
    std::string_view direction, int worker_id, int device_id,
    std::string_view worker_type, double duration_ms)
{
  if (duration_ms < 0.0 || families_.io_copy_latency == nullptr) {
    return;
  }
  IoKey key{
      std::string(direction), worker_id, device_id, std::string(worker_type)};
  std::scoped_lock lock(mutexes_.io_metrics);
  auto* histogram = get_or_create_cached_series(
      caches_.io.copy_latency, std::move(key),
      [this, direction, worker_id, device_id,
       worker_type](bool overflow) -> prometheus::Histogram* {
        const auto labels = make_io_metric_labels(
            overflow, direction, worker_id, device_id, worker_type);
        return &families_.io_copy_latency->Add(
            {{"direction", labels.direction},
             {"worker_id", labels.worker.worker_id},
             {"device", labels.worker.device},
             {"worker_type", labels.worker.worker_type}},
            kInferenceLatencyMsBuckets);
      });
  observe_histogram_if_non_negative(histogram, duration_ms);
}

void
MetricsRegistry::increment_transfer_bytes(
    std::string_view direction, int worker_id, int device_id,
    std::string_view worker_type, std::size_t bytes)
{
  if (bytes == 0 || families_.transfer_bytes == nullptr) {
    return;
  }
  IoKey key{
      std::string(direction), worker_id, device_id, std::string(worker_type)};
  std::scoped_lock lock(mutexes_.io_metrics);
  auto* counter = get_or_create_cached_series(
      caches_.io.transfer_bytes, std::move(key),
      [this, direction, worker_id, device_id,
       worker_type](bool overflow) -> prometheus::Counter* {
        const auto labels = make_io_metric_labels(
            overflow, direction, worker_id, device_id, worker_type);
        return &families_.transfer_bytes->Add(
            {{"direction", labels.direction},
             {"worker_id", labels.worker.worker_id},
             {"device", labels.worker.device},
             {"worker_type", labels.worker.worker_type}});
      });
  increment_counter_if_present(counter, static_cast<double>(bytes));
}

void
increment_inference_completed(
    std::string_view model_name, std::size_t logical_jobs)
{
  with_metrics_registry([model_name, logical_jobs](MetricsRegistry& metrics) {
    metrics.increment_completed_counter(model_name, logical_jobs);
  });
}

void
increment_inference_failure(
    std::string_view stage, std::string_view reason,
    std::string_view model_name, std::size_t count)
{
  with_metrics_registry(
      [stage, reason, model_name, count](MetricsRegistry& metrics) {
        metrics.increment_failure_counter(
            MetricsRegistry::FailureStageLabel{stage},
            MetricsRegistry::FailureReasonLabel{reason},
            MetricsRegistry::ModelLabel{model_name}, count);
      });
}

void
MetricsRegistry::set_queue_capacity(std::size_t capacity)
{
  registry_state_.queue_capacity.store(capacity, std::memory_order_release);
}

auto
MetricsRegistry::queue_capacity_value() const -> std::size_t
{
  return registry_state_.queue_capacity.load(std::memory_order_acquire);
}

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
void
MetricsRegistry::run_sampling_request_nb()
{
  if (sampler_ != nullptr) {
    sampler_->run_sampling_request_nb();
  }
}
#endif  // SONAR_IGNORE_END

void
MetricsRegistry::Sampler::run_sampling_request_nb()
{
  std::scoped_lock<std::mutex> lock(registry_->mutexes_.sampling);
  perform_sampling_request_nb();
}

void
MetricsRegistry::Sampler::sample_cpu_usage()
{
  auto& gauges = registry_->gauges_;
  const auto& providers = registry_->providers_;
  if (gauges.system_cpu_usage_percent == nullptr) {
    return;
  }
  if (!providers.cpu_usage_provider) {
    gauges.system_cpu_usage_percent->Set(
        std::numeric_limits<double>::quiet_NaN());
    return;
  }

  try {
    auto usage = providers.cpu_usage_provider();
    if (usage.has_value()) {
      gauges.system_cpu_usage_percent->Set(*usage);
    } else {
      gauges.system_cpu_usage_percent->Set(
          std::numeric_limits<double>::quiet_NaN());
    }
  }
  catch (const std::exception& e) {
    if (should_log_sampling_error(cpu_sampling_error_log_ts())) {
      log_error(std::format("CPU metrics sampling failed: {}", e.what()));
    }
    gauges.system_cpu_usage_percent->Set(
        std::numeric_limits<double>::quiet_NaN());
  }
  catch (...) {
    if (should_log_sampling_error(cpu_sampling_error_log_ts())) {
      log_error("CPU metrics sampling failed due to an unknown error");
    }
    gauges.system_cpu_usage_percent->Set(
        std::numeric_limits<double>::quiet_NaN());
  }
}

void
MetricsRegistry::Sampler::sample_inference_throughput()
{
  auto& gauges = registry_->gauges_;
  if (gauges.inference_throughput == nullptr) {
    return;
  }

  if (auto snap = perf_observer::snapshot()) {
    gauges.inference_throughput->Set(snap->throughput);
  }
}

void
MetricsRegistry::Sampler::sample_process_resident_memory()
{
  auto& gauges = registry_->gauges_;
  if (gauges.process_resident_memory_bytes == nullptr) {
    return;
  }

  if (auto rss_bytes = monitoring::detail::read_process_rss_bytes();
      rss_bytes.has_value()) {
    gauges.process_resident_memory_bytes->Set(*rss_bytes);
  } else {
    gauges.process_resident_memory_bytes->Set(
        std::numeric_limits<double>::quiet_NaN());
  }
}

void
MetricsRegistry::Sampler::sample_process_open_fds()
{
  auto& gauges = registry_->gauges_;
  if (gauges.process_open_fds == nullptr) {
    return;
  }

  if (auto fds = monitoring::detail::read_process_open_fds(); fds.has_value()) {
    gauges.process_open_fds->Set(*fds);
  } else {
    gauges.process_open_fds->Set(std::numeric_limits<double>::quiet_NaN());
  }
}

void
MetricsRegistry::Sampler::sample_gpu_stats()
{
  const auto& providers = registry_->providers_;
  auto& caches = registry_->caches_;
  auto& families = registry_->families_;
  if (!providers.gpu_stats_provider) {
    return;
  }

  try {
    auto gstats = providers.gpu_stats_provider();
    std::unordered_set<int> seen_indices;
    seen_indices.reserve(gstats.size());
    for (const auto& stats : gstats) {
      seen_indices.insert(stats.index);
      const std::string label = std::to_string(stats.index);

      ensure_gpu_gauge(
          caches.gpu.utilization, families.gpu_utilization, stats.index, label)
          ->Set(stats.util_percent);
      ensure_gpu_gauge(
          caches.gpu.memory_used, families.gpu_memory_used_bytes, stats.index,
          label)
          ->Set(stats.mem_used_bytes);
      ensure_gpu_gauge(
          caches.gpu.memory_total, families.gpu_memory_total_bytes, stats.index,
          label)
          ->Set(stats.mem_total_bytes);

      set_or_clear_nan(
          caches.gpu.temperature, families.gpu_temperature, stats.index, label,
          stats.temperature_celsius);
      set_or_clear_nan(
          caches.gpu.power, families.gpu_power, stats.index, label,
          stats.power_watts);
    }

    clear_missing_gauges(
        caches.gpu.utilization, families.gpu_utilization, seen_indices);
    clear_missing_gauges(
        caches.gpu.memory_used, families.gpu_memory_used_bytes, seen_indices);
    clear_missing_gauges(
        caches.gpu.memory_total, families.gpu_memory_total_bytes, seen_indices);
    clear_missing_gauges(
        caches.gpu.temperature, families.gpu_temperature, seen_indices);
    clear_missing_gauges(caches.gpu.power, families.gpu_power, seen_indices);
  }
  catch (const std::exception& e) {
    if (should_log_sampling_error(gpu_sampling_error_log_ts())) {
      log_error(std::format("GPU metrics sampling failed: {}", e.what()));
    }
  }
  catch (...) {
    if (should_log_sampling_error(gpu_sampling_error_log_ts())) {
      log_error("GPU metrics sampling failed due to an unknown error");
    }
  }
}

void
MetricsRegistry::Sampler::perform_sampling_request_nb()
{
  sample_cpu_usage();
  sample_inference_throughput();
  sample_process_resident_memory();
  sample_process_open_fds();
  sample_gpu_stats();
}

void
MetricsRegistry::Sampler::sampling_loop(const std::stop_token& stop)
{
  using namespace std::chrono_literals;
  auto next_sleep = 1000ms;
  while (!stop.stop_requested()) {
    {
      std::scoped_lock<std::mutex> lock(registry_->mutexes_.sampling);
      perform_sampling_request_nb();
    }
    for (auto slept = 0ms; slept < next_sleep && !stop.stop_requested();
         slept += 50ms) {
      std::this_thread::sleep_for(50ms);
    }
  }
}

}  // namespace starpu_server

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
void
starpu_server::testing::MetricsRegistryTestAccessor::ClearCpuUsageProvider(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.providers_.cpu_usage_provider = {};
}

void
starpu_server::testing::MetricsRegistryTestAccessor::ClearSystemCpuUsageGauge(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.gauges_.system_cpu_usage_percent = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::ClearProcessOpenFdsGauge(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.gauges_.process_open_fds = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    ClearProcessResidentMemoryGauge(starpu_server::MetricsRegistry& metrics)
{
  metrics.gauges_.process_resident_memory_bytes = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    ClearInferenceThroughputGauge(starpu_server::MetricsRegistry& metrics)
{
  metrics.gauges_.inference_throughput = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::ClearGpuStatsProvider(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.providers_.gpu_stats_provider = {};
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::ProcessOpenFdsGauge(
    starpu_server::MetricsRegistry& metrics) -> prometheus::Gauge*
{
  return metrics.gauges_.process_open_fds;
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::ProcessResidentMemoryGauge(
    starpu_server::MetricsRegistry& metrics) -> prometheus::Gauge*
{
  return metrics.gauges_.process_resident_memory_bytes;
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::InferenceThroughputGauge(
    starpu_server::MetricsRegistry& metrics) -> prometheus::Gauge*
{
  return metrics.gauges_.inference_throughput;
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::GpuUtilizationGaugeCount(
    const starpu_server::MetricsRegistry& metrics) -> std::size_t
{
  std::scoped_lock<std::mutex> lock(metrics.mutexes_.sampling);
  return metrics.caches_.gpu.utilization.size();
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::GpuMemoryUsedGaugeCount(
    const starpu_server::MetricsRegistry& metrics) -> std::size_t
{
  std::scoped_lock<std::mutex> lock(metrics.mutexes_.sampling);
  return metrics.caches_.gpu.memory_used.size();
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::GpuMemoryTotalGaugeCount(
    const starpu_server::MetricsRegistry& metrics) -> std::size_t
{
  std::scoped_lock<std::mutex> lock(metrics.mutexes_.sampling);
  return metrics.caches_.gpu.memory_total.size();
}

void
starpu_server::testing::MetricsRegistryTestAccessor::SampleProcessOpenFds(
    starpu_server::MetricsRegistry& metrics)
{
  if (metrics.sampler_ != nullptr) {
    metrics.sampler_->sample_process_open_fds();
  }
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    SampleProcessResidentMemory(starpu_server::MetricsRegistry& metrics)
{
  if (metrics.sampler_ != nullptr) {
    metrics.sampler_->sample_process_resident_memory();
  }
}

void
starpu_server::testing::MetricsRegistryTestAccessor::SampleInferenceThroughput(
    starpu_server::MetricsRegistry& metrics)
{
  if (metrics.sampler_ != nullptr) {
    metrics.sampler_->sample_inference_throughput();
  }
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    ClearStarpuWorkerInflightFamily(starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.starpu_worker_inflight = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    ClearStarpuTaskRuntimeByWorkerFamily(
        starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.starpu_task_runtime_by_worker = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    ClearInferenceComputeLatencyByWorkerFamily(
        starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.inference_compute_latency_by_worker = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::ClearIoCopyLatencyFamily(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.io_copy_latency = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::ClearTransferBytesFamily(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.transfer_bytes = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::ClearModelsLoadedFamily(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.models_loaded = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    ClearModelLoadFailuresFamily(starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.model_load_failures = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    ClearInferenceFailuresFamily(starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.inference_failures = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    ClearInferenceCompletedFamily(starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.inference_completed = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    ClearRequestsReceivedFamily(starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.requests_received = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    ClearRequestsByStatusFamily(starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.requests_by_status = nullptr;
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::FailureKeyOverflowIsEmpty()
    -> bool
{
  const auto key = MetricsRegistry::FailureKey::Overflow();
  return key.overflow && key.stage.empty() && key.reason.empty() &&
         key.model.empty();
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::FailureKeyEquals(
    std::string_view stage_lhs, std::string_view reason_lhs,
    std::string_view model_lhs, bool overflow_lhs, std::string_view stage_rhs,
    std::string_view reason_rhs, std::string_view model_rhs,
    bool overflow_rhs) -> bool
{
  MetricsRegistry::FailureKey lhs{
      std::string(stage_lhs), std::string(reason_lhs), std::string(model_lhs),
      overflow_lhs};
  MetricsRegistry::FailureKey rhs{
      std::string(stage_rhs), std::string(reason_rhs), std::string(model_rhs),
      overflow_rhs};
  return lhs == rhs;
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::ModelKeyOverflowIsEmpty()
    -> bool
{
  const auto key = MetricsRegistry::ModelKey::Overflow();
  return key.overflow && key.model.empty();
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::ModelKeyEquals(
    std::string_view model_lhs, bool overflow_lhs, std::string_view model_rhs,
    bool overflow_rhs) -> bool
{
  MetricsRegistry::ModelKey lhs{std::string(model_lhs), overflow_lhs};
  MetricsRegistry::ModelKey rhs{std::string(model_rhs), overflow_rhs};
  return lhs == rhs;
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::
    ModelDeviceKeyOverflowIsEmpty() -> bool
{
  const auto key = MetricsRegistry::ModelDeviceKey::Overflow();
  return key.overflow && key.model.empty() && key.device.empty();
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::ModelDeviceKeyEquals(
    std::string_view model_lhs, std::string_view device_lhs, bool overflow_lhs,
    std::string_view model_rhs, std::string_view device_rhs,
    bool overflow_rhs) -> bool
{
  MetricsRegistry::ModelDeviceKey lhs{
      std::string(model_lhs), std::string(device_lhs), overflow_lhs};
  MetricsRegistry::ModelDeviceKey rhs{
      std::string(model_rhs), std::string(device_rhs), overflow_rhs};
  return lhs == rhs;
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::IoKeyOverflowIsEmpty()
    -> bool
{
  const auto key = MetricsRegistry::IoKey::Overflow();
  return key.overflow && key.direction.empty() && key.worker_id == 0 &&
         key.device_id == 0 && key.worker_type.empty();
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::IoKeyEquals(
    std::string_view direction_lhs, int worker_id_lhs, int device_id_lhs,
    std::string_view worker_type_lhs, bool overflow_lhs,
    std::string_view direction_rhs, int worker_id_rhs, int device_id_rhs,
    std::string_view worker_type_rhs, bool overflow_rhs) -> bool
{
  MetricsRegistry::IoKey lhs{
      std::string(direction_lhs), worker_id_lhs, device_id_lhs,
      std::string(worker_type_lhs), overflow_lhs};
  MetricsRegistry::IoKey rhs{
      std::string(direction_rhs), worker_id_rhs, device_id_rhs,
      std::string(worker_type_rhs), overflow_rhs};
  return lhs == rhs;
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::WorkerKeyOverflowIsEmpty()
    -> bool
{
  const auto key = MetricsRegistry::WorkerKey::Overflow();
  return key.overflow && key.worker_id == 0 && key.device_id == 0 &&
         key.worker_type.empty();
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::WorkerKeyEquals(
    int worker_id_lhs, int device_id_lhs, std::string_view worker_type_lhs,
    bool overflow_lhs, int worker_id_rhs, int device_id_rhs,
    std::string_view worker_type_rhs, bool overflow_rhs) -> bool
{
  MetricsRegistry::WorkerKey lhs{
      worker_id_lhs, device_id_lhs, std::string(worker_type_lhs), overflow_lhs};
  MetricsRegistry::WorkerKey rhs{
      worker_id_rhs, device_id_rhs, std::string(worker_type_rhs), overflow_rhs};
  return lhs == rhs;
}
#endif  // SONAR_IGNORE_END
