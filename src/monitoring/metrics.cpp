#include "monitoring/metrics.hpp"

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
const std::string kOverflowLabel{"__overflow__"};
constexpr std::string_view kLabelEscapePrefix{"__label__"};
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
constexpr auto kSamplingErrorLogThrottle = std::chrono::seconds(0);
#else
constexpr auto kSamplingErrorLogThrottle = std::chrono::seconds(60);
#endif  // SONAR_IGNORE_END

constexpr int kStatusOk = 0;
constexpr int kStatusCancelled = 1;
constexpr int kStatusUnknown = 2;
constexpr int kStatusInvalidArgument = 3;
constexpr int kStatusDeadlineExceeded = 4;
constexpr int kStatusNotFound = 5;
constexpr int kStatusAlreadyExists = 6;
constexpr int kStatusPermissionDenied = 7;
constexpr int kStatusResourceExhausted = 8;
constexpr int kStatusFailedPrecondition = 9;
constexpr int kStatusAborted = 10;
constexpr int kStatusOutOfRange = 11;
constexpr int kStatusUnimplemented = 12;
constexpr int kStatusInternal = 13;
constexpr int kStatusUnavailable = 14;
constexpr int kStatusDataLoss = 15;
constexpr int kStatusUnauthenticated = 16;

auto
status_code_label(int code) -> std::string
{
  switch (code) {
    case kStatusOk:
      return "OK";
    case kStatusCancelled:
      return "CANCELLED";
    case kStatusUnknown:
      return "UNKNOWN";
    case kStatusInvalidArgument:
      return "INVALID_ARGUMENT";
    case kStatusDeadlineExceeded:
      return "DEADLINE_EXCEEDED";
    case kStatusNotFound:
      return "NOT_FOUND";
    case kStatusAlreadyExists:
      return "ALREADY_EXISTS";
    case kStatusPermissionDenied:
      return "PERMISSION_DENIED";
    case kStatusResourceExhausted:
      return "RESOURCE_EXHAUSTED";
    case kStatusFailedPrecondition:
      return "FAILED_PRECONDITION";
    case kStatusAborted:
      return "ABORTED";
    case kStatusOutOfRange:
      return "OUT_OF_RANGE";
    case kStatusUnimplemented:
      return "UNIMPLEMENTED";
    case kStatusInternal:
      return "INTERNAL";
    case kStatusUnavailable:
      return "UNAVAILABLE";
    case kStatusDataLoss:
      return "DATA_LOSS";
    case kStatusUnauthenticated:
      return "UNAUTHENTICATED";
    default:
      return std::to_string(code);
  }
}

auto
escape_label_value(std::string_view value) -> std::string
{
  if (value == kOverflowLabel || value.starts_with(kLabelEscapePrefix)) {
    std::string escaped;
    escaped.reserve(kLabelEscapePrefix.size() + value.size());
    escaped.append(kLabelEscapePrefix);
    escaped.append(value);
    return escaped;
  }
  return std::string(value);
}

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
  for (auto iter = gauges.begin(); iter != gauges.end();) {
    if (!seen_indices.contains(iter->first)) {
      if (family != nullptr) {
        family->Remove(iter->second);
      }
      iter = gauges.erase(iter);
    } else {
      ++iter;
    }
  }
}

auto
cpu_sampling_error_log_ts() -> std::atomic<std::int64_t>&
{
  static std::atomic<std::int64_t> last_log_ts{0};
  return last_log_ts;
}

auto
gpu_sampling_error_log_ts() -> std::atomic<std::int64_t>&
{
  static std::atomic<std::int64_t> last_log_ts{0};
  return last_log_ts;
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
  initialize(port, start_sampler_thread, std::move(exposer_handle));
}

void
MetricsRegistry::initialize(
    int port, bool start_sampler_thread,
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

  auto& counter_family = prometheus::BuildCounter()
                             .Name("requests_total")
                             .Help("Total requests received")
                             .Register(*registry_state_.registry);
  counters_.requests_total = &counter_family.Add({});

  auto& status_family = prometheus::BuildCounter()
                            .Name("requests_by_status_total")
                            .Help(
                                "Total requests grouped by gRPC status code "
                                "and model name")
                            .Register(*registry_state_.registry);
  families_.requests_by_status = &status_family;
  (void)families_.requests_by_status->Add(
      {{"code", "unlabeled"}, {"model", "unlabeled"}});

  auto& completed_family = prometheus::BuildCounter()
                               .Name("inference_completed_total")
                               .Help("Total logical inferences completed")
                               .Register(*registry_state_.registry);
  families_.inference_completed = &completed_family;
  (void)families_.inference_completed->Add({{"model", "unlabeled"}});

  auto& failures_family = prometheus::BuildCounter()
                              .Name("inference_failures_total")
                              .Help(
                                  "Inference failures grouped by stage and "
                                  "reason")
                              .Register(*registry_state_.registry);
  families_.inference_failures = &failures_family;
  (void)families_.inference_failures->Add(
      {{"stage", "unlabeled"},
       {"reason", "unlabeled"},
       {"model", "unlabeled"}});

  auto& rejected_family =
      prometheus::BuildCounter()
          .Name("requests_rejected_total")
          .Help("Total requests rejected (e.g., queue full)")
          .Register(*registry_state_.registry);
  counters_.requests_rejected_total = &rejected_family.Add({});

  auto& histogram_family = prometheus::BuildHistogram()
                               .Name("inference_latency_ms")
                               .Help("Inference latency in milliseconds")
                               .Register(*registry_state_.registry);
  histograms_.inference_latency =
      &histogram_family.Add({}, kInferenceLatencyMsBuckets);

  auto& gauge_family = prometheus::BuildGauge()
                           .Name("inference_queue_size")
                           .Help("Number of jobs in the inference queue")
                           .Register(*registry_state_.registry);
  gauges_.queue_size = &gauge_family.Add({});

  auto& queue_capacity_family = prometheus::BuildGauge()
                                    .Name("inference_max_queue_size")
                                    .Help(
                                        "Configured maximum inference queue "
                                        "capacity")
                                    .Register(*registry_state_.registry);
  gauges_.queue_capacity = &queue_capacity_family.Add({});

  auto& queue_fill_family = prometheus::BuildGauge()
                                .Name("inference_queue_fill_ratio")
                                .Help(
                                    "Queue occupancy ratio "
                                    "(queue_size / max_queue_size)")
                                .Register(*registry_state_.registry);
  gauges_.queue_fill_ratio = &queue_fill_family.Add({});

  auto& inflight_family = prometheus::BuildGauge()
                              .Name("inference_inflight_tasks")
                              .Help(
                                  "Number of StarPU tasks currently submitted "
                                  "and not yet completed")
                              .Register(*registry_state_.registry);
  gauges_.inflight_tasks = &inflight_family.Add({});

  auto& inflight_cap_family = prometheus::BuildGauge()
                                  .Name("inference_max_inflight_tasks")
                                  .Help(
                                      "Configured cap on inflight StarPU tasks "
                                      "(0 means unbounded)")
                                  .Register(*registry_state_.registry);
  gauges_.max_inflight_tasks = &inflight_cap_family.Add({});

  auto& worker_busy_family = prometheus::BuildGauge()
                                 .Name("starpu_worker_busy_ratio")
                                 .Help(
                                     "Approximate ratio of inflight tasks to "
                                     "max inflight limit (0-1, 0 when "
                                     "unbounded)")
                                 .Register(*registry_state_.registry);
  gauges_.starpu_worker_busy_ratio = &worker_busy_family.Add({});

  auto& prepared_queue_family = prometheus::BuildGauge()
                                    .Name("starpu_prepared_queue_depth")
                                    .Help(
                                        "Number of batched jobs waiting for "
                                        "StarPU submission")
                                    .Register(*registry_state_.registry);
  gauges_.starpu_prepared_queue_depth = &prepared_queue_family.Add({});

  auto& cpu_family = prometheus::BuildGauge()
                         .Name("system_cpu_usage_percent")
                         .Help("System-wide CPU utilization percentage (0-100)")
                         .Register(*registry_state_.registry);
  gauges_.system_cpu_usage_percent = &cpu_family.Add({});

  auto& throughput_family = prometheus::BuildGauge()
                                .Name("inference_throughput_rps")
                                .Help(
                                    "Rolling throughput of logical "
                                    "inferences/s based on completed jobs")
                                .Register(*registry_state_.registry);
  gauges_.inference_throughput = &throughput_family.Add({});

  auto& rss_family = prometheus::BuildGauge()
                         .Name("process_resident_memory_bytes")
                         .Help("Resident Set Size of the server process")
                         .Register(*registry_state_.registry);
  gauges_.process_resident_memory_bytes = &rss_family.Add({});

  auto& fds_family = prometheus::BuildGauge()
                         .Name("process_open_fds")
                         .Help("Number of open file descriptors")
                         .Register(*registry_state_.registry);
  gauges_.process_open_fds = &fds_family.Add({});

  auto& health_family = prometheus::BuildGauge()
                            .Name("server_health_state")
                            .Help(
                                "Server health state: 1=ready, 0=not ready or "
                                "shutting down")
                            .Register(*registry_state_.registry);
  gauges_.server_health_state = &health_family.Add({});

  auto& queue_latency_family = prometheus::BuildHistogram()
                                   .Name("inference_queue_latency_ms")
                                   .Help("Time spent waiting in the queue")
                                   .Register(*registry_state_.registry);
  histograms_.queue_latency =
      &queue_latency_family.Add({}, kInferenceLatencyMsBuckets);

  auto& batch_efficiency_family = prometheus::BuildHistogram()
                                      .Name("inference_batch_efficiency_ratio")
                                      .Help(
                                          "Ratio of effective batch size to "
                                          "logical request count")
                                      .Register(*registry_state_.registry);
  histograms_.batch_efficiency =
      &batch_efficiency_family.Add({}, kBatchEfficiencyBuckets);

  auto& batch_pending_family = prometheus::BuildGauge()
                                   .Name("inference_batch_collect_pending_jobs")
                                   .Help(
                                       "Number of requests aggregated in the "
                                       "current batch collection")
                                   .Register(*registry_state_.registry);
  gauges_.batch_pending_jobs = &batch_pending_family.Add({});

  auto& batch_collect_family = prometheus::BuildHistogram()
                                   .Name("inference_batch_collect_ms")
                                   .Help("Time spent collecting a batch")
                                   .Register(*registry_state_.registry);
  histograms_.batch_collect_latency =
      &batch_collect_family.Add({}, kInferenceLatencyMsBuckets);

  auto& submit_family = prometheus::BuildHistogram()
                            .Name("inference_submit_latency_ms")
                            .Help(
                                "Time spent between dequeue and submission "
                                "into StarPU")
                            .Register(*registry_state_.registry);
  histograms_.submit_latency =
      &submit_family.Add({}, kInferenceLatencyMsBuckets);

  auto& scheduling_family = prometheus::BuildHistogram()
                                .Name("inference_scheduling_latency_ms")
                                .Help(
                                    "Time spent waiting for scheduling on a "
                                    "StarPU worker")
                                .Register(*registry_state_.registry);
  histograms_.scheduling_latency =
      &scheduling_family.Add({}, kInferenceLatencyMsBuckets);

  auto& codelet_family = prometheus::BuildHistogram()
                             .Name("inference_codelet_latency_ms")
                             .Help("Duration of the StarPU codelet execution")
                             .Register(*registry_state_.registry);
  histograms_.codelet_latency =
      &codelet_family.Add({}, kInferenceLatencyMsBuckets);

  auto& compute_family = prometheus::BuildHistogram()
                             .Name("inference_compute_latency_ms")
                             .Help("Model compute time (inference)")
                             .Register(*registry_state_.registry);
  histograms_.inference_compute_latency =
      &compute_family.Add({}, kInferenceLatencyMsBuckets);

  auto& callback_family = prometheus::BuildHistogram()
                              .Name("inference_callback_latency_ms")
                              .Help("Callback/response handling latency")
                              .Register(*registry_state_.registry);
  histograms_.callback_latency =
      &callback_family.Add({}, kInferenceLatencyMsBuckets);

  auto& preprocess_family = prometheus::BuildHistogram()
                                .Name("inference_preprocess_latency_ms")
                                .Help("Server-side preprocessing latency")
                                .Register(*registry_state_.registry);
  histograms_.preprocess_latency =
      &preprocess_family.Add({}, kInferenceLatencyMsBuckets);

  auto& postprocess_family = prometheus::BuildHistogram()
                                 .Name("inference_postprocess_latency_ms")
                                 .Help("Server-side postprocessing latency")
                                 .Register(*registry_state_.registry);
  histograms_.postprocess_latency =
      &postprocess_family.Add({}, kInferenceLatencyMsBuckets);

  auto& batch_size_family = prometheus::BuildHistogram()
                                .Name("inference_batch_size")
                                .Help("Effective batch size executed")
                                .Register(*registry_state_.registry);
  histograms_.batch_size = &batch_size_family.Add({}, kBatchSizeBuckets);

  auto& logical_batch_size_family = prometheus::BuildHistogram()
                                        .Name("inference_logical_batch_size")
                                        .Help(
                                            "Number of logical requests "
                                            "aggregated into a batch")
                                        .Register(*registry_state_.registry);
  histograms_.logical_batch_size =
      &logical_batch_size_family.Add({}, kBatchSizeBuckets);

  auto& model_load_hist_family = prometheus::BuildHistogram()
                                     .Name("model_load_duration_ms")
                                     .Help("Duration of model load and wiring")
                                     .Register(*registry_state_.registry);
  histograms_.model_load_duration =
      &model_load_hist_family.Add({}, kModelLoadDurationMsBuckets);

  auto& model_load_fail_family = prometheus::BuildCounter()
                                     .Name("model_load_failures_total")
                                     .Help("Total failed model load attempts")
                                     .Register(*registry_state_.registry);
  families_.model_load_failures = &model_load_fail_family;
  (void)families_.model_load_failures->Add({{"model", "unlabeled"}});

  auto& models_loaded_family =
      prometheus::BuildGauge()
          .Name("models_loaded")
          .Help("Flag indicating a model is loaded on a device")
          .Register(*registry_state_.registry);
  families_.models_loaded = &models_loaded_family;
  (void)families_.models_loaded->Add(
      {{"model", "unlabeled"}, {"device", "unknown"}});

  families_.gpu_utilization =
      &prometheus::BuildGauge()
           .Name("gpu_utilization_percent")
           .Help("GPU utilization percentage per GPU (0-100)")
           .Register(*registry_state_.registry);
  families_.gpu_memory_used_bytes =
      &prometheus::BuildGauge()
           .Name("gpu_memory_used_bytes")
           .Help("Used GPU memory in bytes per GPU")
           .Register(*registry_state_.registry);
  families_.gpu_memory_total_bytes =
      &prometheus::BuildGauge()
           .Name("gpu_memory_total_bytes")
           .Help("Total GPU memory in bytes per GPU")
           .Register(*registry_state_.registry);

  families_.gpu_temperature = &prometheus::BuildGauge()
                                   .Name("gpu_temperature_celsius")
                                   .Help("Reported GPU temperature in Celsius")
                                   .Register(*registry_state_.registry);

  families_.gpu_power = &prometheus::BuildGauge()
                             .Name("gpu_power_watts")
                             .Help("Reported GPU power draw in Watts")
                             .Register(*registry_state_.registry);

  auto& starpu_runtime_family = prometheus::BuildHistogram()
                                    .Name("starpu_task_runtime_ms")
                                    .Help("Wall-clock runtime of a StarPU task")
                                    .Register(*registry_state_.registry);
  histograms_.starpu_task_runtime =
      &starpu_runtime_family.Add({}, kTaskRuntimeMsBuckets);

  auto& compute_latency_by_worker_family =
      prometheus::BuildHistogram()
          .Name("inference_compute_latency_ms_by_worker")
          .Help(
              "Model compute latency by worker and device "
              "(callback start - inference start)")
          .Register(*registry_state_.registry);
  families_.inference_compute_latency_by_worker =
      &compute_latency_by_worker_family;

  auto& starpu_runtime_by_worker_family =
      prometheus::BuildHistogram()
          .Name("starpu_task_runtime_ms_by_worker")
          .Help("StarPU codelet runtime by worker/device")
          .Register(*registry_state_.registry);
  families_.starpu_task_runtime_by_worker = &starpu_runtime_by_worker_family;

  auto& worker_inflight_family =
      prometheus::BuildGauge()
          .Name("starpu_worker_inflight_tasks")
          .Help("Inflight StarPU tasks per worker/device")
          .Register(*registry_state_.registry);
  families_.starpu_worker_inflight = &worker_inflight_family;

  auto& io_copy_latency_family =
      prometheus::BuildHistogram()
          .Name("inference_io_copy_ms")
          .Help("Host/device copy latency by direction/device/worker")
          .Register(*registry_state_.registry);
  families_.io_copy_latency = &io_copy_latency_family;

  auto& transfer_bytes_family =
      prometheus::BuildCounter()
          .Name("inference_transfer_bytes_total")
          .Help("Total bytes transferred by direction/device/worker")
          .Register(*registry_state_.registry);
  families_.transfer_bytes = &transfer_bytes_family;

  if (start_sampler_thread) {
    registry_state_.sampler_thread = std::jthread(
        [this](const std::stop_token& stop) { this->sampling_loop(stop); });
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
  auto metrics_ptr = metrics_atomic().load(std::memory_order_acquire);
  if (metrics_ptr && metrics_ptr->queue_size_gauge() != nullptr) {
    metrics_ptr->queue_size_gauge()->Set(static_cast<double>(size));
    const auto capacity = metrics_ptr->queue_capacity_value();
    if (capacity > 0 && metrics_ptr->queue_fill_ratio_gauge() != nullptr) {
      const double ratio =
          static_cast<double>(size) / static_cast<double>(capacity);
      metrics_ptr->queue_fill_ratio_gauge()->Set(std::clamp(ratio, 0.0, 1.0));
    }
  }
}

void
set_inflight_tasks(std::size_t size)
{
  auto metrics_ptr = metrics_atomic().load(std::memory_order_acquire);
  if (metrics_ptr && metrics_ptr->inflight_tasks_gauge() != nullptr) {
    metrics_ptr->inflight_tasks_gauge()->Set(static_cast<double>(size));
  }
}

void
set_starpu_worker_busy_ratio(double ratio)
{
  auto metrics_ptr = metrics_atomic().load(std::memory_order_acquire);
  if (metrics_ptr && metrics_ptr->starpu_worker_busy_ratio_gauge() != nullptr) {
    metrics_ptr->starpu_worker_busy_ratio_gauge()->Set(
        std::clamp(ratio, 0.0, 1.0));
  }
}

void
set_max_inflight_tasks(std::size_t max_tasks)
{
  auto metrics_ptr = metrics_atomic().load(std::memory_order_acquire);
  if (metrics_ptr && metrics_ptr->max_inflight_tasks_gauge() != nullptr) {
    metrics_ptr->max_inflight_tasks_gauge()->Set(
        static_cast<double>(max_tasks));
  }
}

void
set_queue_capacity(std::size_t capacity)
{
  auto metrics_ptr = metrics_atomic().load(std::memory_order_acquire);
  if (metrics_ptr == nullptr) {
    return;
  }
  metrics_ptr->set_queue_capacity(capacity);
  if (metrics_ptr->queue_capacity_gauge() != nullptr) {
    metrics_ptr->queue_capacity_gauge()->Set(static_cast<double>(capacity));
  }
  if (capacity > 0 && metrics_ptr->queue_fill_ratio_gauge() != nullptr &&
      metrics_ptr->queue_size_gauge() != nullptr) {
    const double size = metrics_ptr->queue_size_gauge()->Value();
    metrics_ptr->queue_fill_ratio_gauge()->Set(
        std::clamp(size / static_cast<double>(capacity), 0.0, 1.0));
  } else if (metrics_ptr->queue_fill_ratio_gauge() != nullptr) {
    metrics_ptr->queue_fill_ratio_gauge()->Set(0.0);
  }
}

// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
void
set_queue_fill_ratio(std::size_t size, std::size_t capacity)
{
  auto metrics_ptr = metrics_atomic().load(std::memory_order_acquire);
  if (metrics_ptr == nullptr ||
      metrics_ptr->queue_fill_ratio_gauge() == nullptr || capacity == 0) {
    return;
  }
  metrics_ptr->queue_fill_ratio_gauge()->Set(
      static_cast<double>(size) / static_cast<double>(capacity));
}
#endif  // SONAR_IGNORE_END
// GCOVR_EXCL_STOP

void
set_starpu_prepared_queue_depth(std::size_t depth)
{
  auto metrics_ptr = metrics_atomic().load(std::memory_order_acquire);
  if (metrics_ptr &&
      metrics_ptr->starpu_prepared_queue_depth_gauge() != nullptr) {
    metrics_ptr->starpu_prepared_queue_depth_gauge()->Set(
        static_cast<double>(depth));
  }
}

void
set_batch_pending_jobs(std::size_t pending)
{
  auto metrics_ptr = metrics_atomic().load(std::memory_order_acquire);
  if (metrics_ptr && metrics_ptr->batch_pending_jobs_gauge() != nullptr) {
    metrics_ptr->batch_pending_jobs_gauge()->Set(static_cast<double>(pending));
  }
}

void
increment_rejected_requests()
{
  auto metrics_ptr = metrics_atomic().load(std::memory_order_acquire);
  if (metrics_ptr && metrics_ptr->requests_rejected_total() != nullptr) {
    metrics_ptr->requests_rejected_total()->Increment();
  }
}

void
set_server_health(bool ready)
{
  auto metrics_ptr = metrics_atomic().load(std::memory_order_acquire);
  if (metrics_ptr && metrics_ptr->server_health_state_gauge() != nullptr) {
    metrics_ptr->server_health_state_gauge()->Set(ready ? 1.0 : 0.0);
  }
}

void
increment_request_status(int status_code, std::string_view model_name)
{
  auto metrics_ptr = metrics_atomic().load(std::memory_order_acquire);
  if (!metrics_ptr) {
    return;
  }
  metrics_ptr->increment_status_counter(
      MetricsRegistry::StatusCodeLabel{status_code_label(status_code)},
      MetricsRegistry::ModelLabel{model_name});
}

void
observe_batch_size(std::size_t batch_size)
{
  auto metrics_ptr = metrics_atomic().load(std::memory_order_acquire);
  if (metrics_ptr && metrics_ptr->batch_size_histogram() != nullptr) {
    metrics_ptr->batch_size_histogram()->Observe(
        static_cast<double>(batch_size));
  }
}

void
observe_logical_batch_size(std::size_t logical_jobs)
{
  auto metrics_ptr = metrics_atomic().load(std::memory_order_acquire);
  if (metrics_ptr && metrics_ptr->logical_batch_size_histogram() != nullptr) {
    metrics_ptr->logical_batch_size_histogram()->Observe(
        static_cast<double>(logical_jobs));
  }
}

void
observe_batch_efficiency(double ratio)
{
  auto metrics_ptr = metrics_atomic().load(std::memory_order_acquire);
  if (metrics_ptr && metrics_ptr->batch_efficiency_histogram() != nullptr &&
      ratio >= 0.0) {
    metrics_ptr->batch_efficiency_histogram()->Observe(ratio);
  }
}

void
observe_latency_breakdown(
    double queue_ms, double batch_ms, double submit_ms, double scheduling_ms,
    double codelet_ms, double inference_ms, double callback_ms,
    double preprocess_ms, double postprocess_ms)
{
  auto metrics_ptr = metrics_atomic().load(std::memory_order_acquire);
  if (!metrics_ptr) {
    return;
  }

  const auto observe_if = [](prometheus::Histogram* hist, double value) {
    if (hist != nullptr && value >= 0.0) {
      hist->Observe(value);
    }
  };

  observe_if(metrics_ptr->queue_latency_histogram(), queue_ms);
  observe_if(metrics_ptr->batch_collect_latency_histogram(), batch_ms);
  observe_if(metrics_ptr->submit_latency_histogram(), submit_ms);
  observe_if(metrics_ptr->scheduling_latency_histogram(), scheduling_ms);
  observe_if(metrics_ptr->codelet_latency_histogram(), codelet_ms);
  observe_if(metrics_ptr->inference_compute_latency_histogram(), inference_ms);
  observe_if(metrics_ptr->callback_latency_histogram(), callback_ms);
  observe_if(metrics_ptr->preprocess_latency_histogram(), preprocess_ms);
  observe_if(metrics_ptr->postprocess_latency_histogram(), postprocess_ms);
}

void
observe_starpu_task_runtime(double runtime_ms)
{
  auto metrics_ptr = metrics_atomic().load(std::memory_order_acquire);
  if (metrics_ptr && metrics_ptr->starpu_task_runtime_histogram() != nullptr &&
      runtime_ms >= 0.0) {
    metrics_ptr->starpu_task_runtime_histogram()->Observe(runtime_ms);
  }
}

void
observe_model_load_duration(double duration_ms)
{
  auto metrics_ptr = metrics_atomic().load(std::memory_order_acquire);
  if (metrics_ptr && metrics_ptr->model_load_duration_histogram() != nullptr &&
      duration_ms >= 0.0) {
    metrics_ptr->model_load_duration_histogram()->Observe(duration_ms);
  }
}

void
set_model_loaded(
    std::string_view model_name, std::string_view device_label, bool loaded)
{
  auto metrics_ptr = metrics_atomic().load(std::memory_order_acquire);
  if (metrics_ptr == nullptr) {
    return;
  }
  metrics_ptr->set_model_loaded_flag(
      MetricsRegistry::ModelLabel{model_name},
      MetricsRegistry::DeviceLabel{device_label}, loaded);
}

void
increment_model_load_failure(std::string_view model_name)
{
  auto metrics_ptr = metrics_atomic().load(std::memory_order_acquire);
  if (metrics_ptr == nullptr) {
    return;
  }
  metrics_ptr->increment_model_load_failure_counter(model_name);
}

void
observe_compute_latency_by_worker(
    int worker_id, int device_id, std::string_view worker_type,
    double latency_ms)
{
  auto metrics_ptr = metrics_atomic().load(std::memory_order_acquire);
  if (!metrics_ptr) {
    return;
  }
  metrics_ptr->observe_compute_latency_by_worker(
      worker_id, device_id, worker_type, latency_ms);
}

void
observe_task_runtime_by_worker(
    int worker_id, int device_id, std::string_view worker_type,
    double latency_ms)
{
  auto metrics_ptr = metrics_atomic().load(std::memory_order_acquire);
  if (!metrics_ptr) {
    return;
  }
  metrics_ptr->observe_task_runtime_by_worker(
      worker_id, device_id, worker_type, latency_ms);
}

void
set_worker_inflight_gauge(
    int worker_id, int device_id, std::string_view worker_type,
    std::size_t value)
{
  auto metrics_ptr = metrics_atomic().load(std::memory_order_acquire);
  if (!metrics_ptr) {
    return;
  }
  metrics_ptr->set_worker_inflight_gauge(
      worker_id, device_id, worker_type, value);
}

void
observe_io_copy_latency(
    std::string_view direction, int worker_id, int device_id,
    std::string_view worker_type, double duration_ms)
{
  auto metrics_ptr = metrics_atomic().load(std::memory_order_acquire);
  if (!metrics_ptr) {
    return;
  }
  metrics_ptr->observe_io_copy_latency(
      direction, worker_id, device_id, worker_type, duration_ms);
}

void
increment_transfer_bytes(
    std::string_view direction, int worker_id, int device_id,
    std::string_view worker_type, std::size_t bytes)
{
  auto metrics_ptr = metrics_atomic().load(std::memory_order_acquire);
  if (!metrics_ptr) {
    return;
  }
  metrics_ptr->increment_transfer_bytes(
      direction, worker_id, device_id, worker_type, bytes);
}

void
MetricsRegistry::request_stop()
{
  if (registry_state_.sampler_thread.joinable()) {
    registry_state_.sampler_thread.request_stop();
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

auto
MetricsRegistry::requests_total() const -> prometheus::Counter*
{
  return counters_.requests_total;
}

auto
MetricsRegistry::requests_rejected_total() const -> prometheus::Counter*
{
  return counters_.requests_rejected_total;
}

auto
MetricsRegistry::inference_latency() const -> prometheus::Histogram*
{
  return histograms_.inference_latency;
}

auto
MetricsRegistry::queue_size_gauge() const -> prometheus::Gauge*
{
  return gauges_.queue_size;
}

auto
MetricsRegistry::inflight_tasks_gauge() const -> prometheus::Gauge*
{
  return gauges_.inflight_tasks;
}

auto
MetricsRegistry::max_inflight_tasks_gauge() const -> prometheus::Gauge*
{
  return gauges_.max_inflight_tasks;
}

auto
MetricsRegistry::starpu_worker_busy_ratio_gauge() const -> prometheus::Gauge*
{
  return gauges_.starpu_worker_busy_ratio;
}

auto
MetricsRegistry::starpu_prepared_queue_depth_gauge() const -> prometheus::Gauge*
{
  return gauges_.starpu_prepared_queue_depth;
}

auto
MetricsRegistry::system_cpu_usage_percent() const -> prometheus::Gauge*
{
  return gauges_.system_cpu_usage_percent;
}

auto
MetricsRegistry::inference_throughput_gauge() const -> prometheus::Gauge*
{
  return gauges_.inference_throughput;
}

auto
MetricsRegistry::process_resident_memory_gauge() const -> prometheus::Gauge*
{
  return gauges_.process_resident_memory_bytes;
}

auto
MetricsRegistry::process_open_fds_gauge() const -> prometheus::Gauge*
{
  return gauges_.process_open_fds;
}

auto
MetricsRegistry::server_health_state_gauge() const -> prometheus::Gauge*
{
  return gauges_.server_health_state;
}

auto
MetricsRegistry::queue_fill_ratio_gauge() const -> prometheus::Gauge*
{
  return gauges_.queue_fill_ratio;
}

auto
MetricsRegistry::queue_capacity_gauge() const -> prometheus::Gauge*
{
  return gauges_.queue_capacity;
}

auto
MetricsRegistry::queue_latency_histogram() const -> prometheus::Histogram*
{
  return histograms_.queue_latency;
}

auto
MetricsRegistry::batch_collect_latency_histogram() const
    -> prometheus::Histogram*
{
  return histograms_.batch_collect_latency;
}

auto
MetricsRegistry::batch_efficiency_histogram() const -> prometheus::Histogram*
{
  return histograms_.batch_efficiency;
}

auto
MetricsRegistry::batch_pending_jobs_gauge() const -> prometheus::Gauge*
{
  return gauges_.batch_pending_jobs;
}

auto
MetricsRegistry::submit_latency_histogram() const -> prometheus::Histogram*
{
  return histograms_.submit_latency;
}

auto
MetricsRegistry::scheduling_latency_histogram() const -> prometheus::Histogram*
{
  return histograms_.scheduling_latency;
}

auto
MetricsRegistry::codelet_latency_histogram() const -> prometheus::Histogram*
{
  return histograms_.codelet_latency;
}

auto
MetricsRegistry::inference_compute_latency_histogram() const
    -> prometheus::Histogram*
{
  return histograms_.inference_compute_latency;
}

auto
MetricsRegistry::callback_latency_histogram() const -> prometheus::Histogram*
{
  return histograms_.callback_latency;
}

auto
MetricsRegistry::preprocess_latency_histogram() const -> prometheus::Histogram*
{
  return histograms_.preprocess_latency;
}

auto
MetricsRegistry::postprocess_latency_histogram() const -> prometheus::Histogram*
{
  return histograms_.postprocess_latency;
}

auto
MetricsRegistry::batch_size_histogram() const -> prometheus::Histogram*
{
  return histograms_.batch_size;
}

auto
MetricsRegistry::logical_batch_size_histogram() const -> prometheus::Histogram*
{
  return histograms_.logical_batch_size;
}

auto
MetricsRegistry::model_load_duration_histogram() const -> prometheus::Histogram*
{
  return histograms_.model_load_duration;
}

auto
MetricsRegistry::starpu_task_runtime_histogram() const -> prometheus::Histogram*
{
  return histograms_.starpu_task_runtime;
}

auto
MetricsRegistry::inference_compute_latency_by_worker_family() const
    -> prometheus::Family<prometheus::Histogram>*
{
  return families_.inference_compute_latency_by_worker;
}

auto
MetricsRegistry::starpu_task_runtime_by_worker_family() const
    -> prometheus::Family<prometheus::Histogram>*
{
  return families_.starpu_task_runtime_by_worker;
}

auto
MetricsRegistry::starpu_worker_inflight_family() const
    -> prometheus::Family<prometheus::Gauge>*
{
  return families_.starpu_worker_inflight;
}

auto
MetricsRegistry::io_copy_latency_family() const
    -> prometheus::Family<prometheus::Histogram>*
{
  return families_.io_copy_latency;
}

auto
MetricsRegistry::transfer_bytes_family() const
    -> prometheus::Family<prometheus::Counter>*
{
  return families_.transfer_bytes;
}

auto
MetricsRegistry::gpu_utilization_family() const
    -> prometheus::Family<prometheus::Gauge>*
{
  return families_.gpu_utilization;
}

auto
MetricsRegistry::gpu_memory_used_bytes_family() const
    -> prometheus::Family<prometheus::Gauge>*
{
  return families_.gpu_memory_used_bytes;
}

auto
MetricsRegistry::gpu_memory_total_bytes_family() const
    -> prometheus::Family<prometheus::Gauge>*
{
  return families_.gpu_memory_total_bytes;
}

auto
MetricsRegistry::gpu_temperature_family() const
    -> prometheus::Family<prometheus::Gauge>*
{
  return families_.gpu_temperature;
}

auto
MetricsRegistry::gpu_power_family() const
    -> prometheus::Family<prometheus::Gauge>*
{
  return families_.gpu_power;
}

auto
MetricsRegistry::requests_by_status_family() const
    -> prometheus::Family<prometheus::Counter>*
{
  return families_.requests_by_status;
}

auto
MetricsRegistry::inference_completed_family() const
    -> prometheus::Family<prometheus::Counter>*
{
  return families_.inference_completed;
}

auto
MetricsRegistry::inference_failures_family() const
    -> prometheus::Family<prometheus::Counter>*
{
  return families_.inference_failures;
}

auto
MetricsRegistry::model_load_failures_family() const
    -> prometheus::Family<prometheus::Counter>*
{
  return families_.model_load_failures;
}

auto
MetricsRegistry::models_loaded_family() const
    -> prometheus::Family<prometheus::Gauge>*
{
  return families_.models_loaded;
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
  auto entry = caches_.status.counters.find(key);
  if (entry == caches_.status.counters.end()) {
    const bool overflow = caches_.status.counters.size() >= kMaxLabelSeries;
    StatusKey map_key = overflow ? StatusKey::Overflow() : std::move(key);
    auto [inserted_it, inserted] =
        caches_.status.counters.try_emplace(std::move(map_key), nullptr);
    entry = inserted_it;
    if (inserted) {
      const std::string code_label_value =
          overflow ? kOverflowLabel : escape_label_value(code_label.value);
      const std::string model_label_value =
          overflow ? kOverflowLabel : escape_label_value(model_label.value);
      entry->second = &families_.requests_by_status->Add(
          {{"code", code_label_value}, {"model", model_label_value}});
    }
  }
  if (entry->second != nullptr) {
    entry->second->Increment();
  }
}

void
MetricsRegistry::increment_completed_counter(
    std::string_view model_label, std::size_t logical_jobs)
{
  if (families_.inference_completed == nullptr) {
    return;
  }
  ModelKey key{std::string(model_label)};
  std::scoped_lock lock(mutexes_.model_metrics);
  auto entry = caches_.model.inference_completed.find(key);
  if (entry == caches_.model.inference_completed.end()) {
    const bool overflow =
        caches_.model.inference_completed.size() >= kMaxLabelSeries;
    ModelKey map_key = overflow ? ModelKey::Overflow() : std::move(key);
    auto [inserted_it, inserted] =
        caches_.model.inference_completed.try_emplace(
            std::move(map_key), nullptr);
    entry = inserted_it;
    if (inserted) {
      const std::string model_label_value =
          overflow ? kOverflowLabel : escape_label_value(model_label);
      entry->second =
          &families_.inference_completed->Add({{"model", model_label_value}});
    }
  }
  if (entry->second != nullptr) {
    entry->second->Increment(static_cast<double>(logical_jobs));
  }
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
  auto entry = caches_.model.inference_failures.find(key);
  if (entry == caches_.model.inference_failures.end()) {
    const bool overflow =
        caches_.model.inference_failures.size() >= kMaxLabelSeries;
    FailureKey map_key = overflow ? FailureKey::Overflow() : std::move(key);
    auto [inserted_it, inserted] = caches_.model.inference_failures.try_emplace(
        std::move(map_key), nullptr);
    entry = inserted_it;
    if (inserted) {
      const std::string stage_label_value =
          overflow ? kOverflowLabel : escape_label_value(stage_label.value);
      const std::string reason_label_value =
          overflow ? kOverflowLabel : escape_label_value(reason_label.value);
      const std::string model_label_value =
          overflow ? kOverflowLabel : escape_label_value(model_label.value);
      entry->second = &families_.inference_failures->Add(
          {{"stage", stage_label_value},
           {"reason", reason_label_value},
           {"model", model_label_value}});
    }
  }
  if (entry->second != nullptr) {
    entry->second->Increment(static_cast<double>(count));
  }
}

void
MetricsRegistry::increment_model_load_failure_counter(
    std::string_view model_label)
{
  if (families_.model_load_failures == nullptr) {
    return;
  }
  ModelKey key{std::string(model_label)};
  std::scoped_lock lock(mutexes_.model_metrics);
  auto entry = caches_.model.model_load_failures.find(key);
  if (entry == caches_.model.model_load_failures.end()) {
    const bool overflow =
        caches_.model.model_load_failures.size() >= kMaxLabelSeries;
    ModelKey map_key = overflow ? ModelKey::Overflow() : std::move(key);
    auto [inserted_it, inserted] =
        caches_.model.model_load_failures.try_emplace(
            std::move(map_key), nullptr);
    entry = inserted_it;
    if (inserted) {
      const std::string model_label_value =
          overflow ? kOverflowLabel : escape_label_value(model_label);
      entry->second =
          &families_.model_load_failures->Add({{"model", model_label_value}});
    }
  }
  if (entry->second != nullptr) {
    entry->second->Increment();
  }
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
  auto entry = caches_.model.models_loaded.find(key);
  if (entry == caches_.model.models_loaded.end()) {
    const bool overflow = caches_.model.models_loaded.size() >= kMaxLabelSeries;
    ModelDeviceKey map_key =
        overflow ? ModelDeviceKey::Overflow() : std::move(key);
    auto [inserted_it, inserted] =
        caches_.model.models_loaded.try_emplace(std::move(map_key), nullptr);
    entry = inserted_it;
    if (inserted) {
      const std::string model_label_value =
          overflow ? kOverflowLabel : escape_label_value(model_label.value);
      const std::string device_label_value =
          overflow ? kOverflowLabel : escape_label_value(device_label.value);
      entry->second = &families_.models_loaded->Add(
          {{"model", model_label_value}, {"device", device_label_value}});
    }
  }
  if (entry->second != nullptr) {
    entry->second->Set(loaded ? 1.0 : 0.0);
  }
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
  auto entry = caches_.worker.compute_latency.find(key);
  if (entry == caches_.worker.compute_latency.end()) {
    const bool overflow =
        caches_.worker.compute_latency.size() >= kMaxLabelSeries;
    WorkerKey map_key = overflow ? WorkerKey::Overflow() : std::move(key);
    auto [inserted_it, inserted] =
        caches_.worker.compute_latency.try_emplace(std::move(map_key), nullptr);
    entry = inserted_it;
    if (inserted) {
      const std::string worker_id_label =
          overflow ? kOverflowLabel : std::to_string(worker_id);
      const std::string device_label =
          overflow ? kOverflowLabel : std::to_string(device_id);
      const std::string worker_type_label =
          overflow ? kOverflowLabel : escape_label_value(worker_type);
      entry->second = &families_.inference_compute_latency_by_worker->Add(
          {{"worker_id", worker_id_label},
           {"device", device_label},
           {"worker_type", worker_type_label}},
          kInferenceLatencyMsBuckets);
    }
  }
  if (entry->second != nullptr) {
    entry->second->Observe(latency_ms);
  }
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
  auto entry = caches_.worker.task_runtime.find(key);
  if (entry == caches_.worker.task_runtime.end()) {
    const bool overflow = caches_.worker.task_runtime.size() >= kMaxLabelSeries;
    WorkerKey map_key = overflow ? WorkerKey::Overflow() : std::move(key);
    auto [inserted_it, inserted] =
        caches_.worker.task_runtime.try_emplace(std::move(map_key), nullptr);
    entry = inserted_it;
    if (inserted) {
      const std::string worker_id_label =
          overflow ? kOverflowLabel : std::to_string(worker_id);
      const std::string device_label =
          overflow ? kOverflowLabel : std::to_string(device_id);
      const std::string worker_type_label =
          overflow ? kOverflowLabel : escape_label_value(worker_type);
      entry->second = &families_.starpu_task_runtime_by_worker->Add(
          {{"worker_id", worker_id_label},
           {"device", device_label},
           {"worker_type", worker_type_label}},
          kTaskRuntimeMsBuckets);
    }
  }
  if (entry->second != nullptr) {
    entry->second->Observe(latency_ms);
  }
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
  auto entry = caches_.worker.inflight.find(key);
  if (entry == caches_.worker.inflight.end()) {
    const bool overflow = caches_.worker.inflight.size() >= kMaxLabelSeries;
    WorkerKey map_key = overflow ? WorkerKey::Overflow() : std::move(key);
    auto [inserted_it, inserted] =
        caches_.worker.inflight.try_emplace(std::move(map_key), nullptr);
    entry = inserted_it;
    if (inserted) {
      const std::string worker_id_label =
          overflow ? kOverflowLabel : std::to_string(worker_id);
      const std::string device_label =
          overflow ? kOverflowLabel : std::to_string(device_id);
      const std::string worker_type_label =
          overflow ? kOverflowLabel : escape_label_value(worker_type);
      entry->second = &families_.starpu_worker_inflight->Add(
          {{"worker_id", worker_id_label},
           {"device", device_label},
           {"worker_type", worker_type_label}});
    }
  }
  if (entry->second != nullptr) {
    entry->second->Set(static_cast<double>(value));
  }
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
  auto entry = caches_.io.copy_latency.find(key);
  if (entry == caches_.io.copy_latency.end()) {
    const bool overflow = caches_.io.copy_latency.size() >= kMaxLabelSeries;
    IoKey map_key = overflow ? IoKey::Overflow() : std::move(key);
    auto [inserted_it, inserted] =
        caches_.io.copy_latency.try_emplace(std::move(map_key), nullptr);
    entry = inserted_it;
    if (inserted) {
      const std::string direction_label =
          overflow ? kOverflowLabel : escape_label_value(direction);
      const std::string worker_id_label =
          overflow ? kOverflowLabel : std::to_string(worker_id);
      const std::string device_label =
          overflow ? kOverflowLabel : std::to_string(device_id);
      const std::string worker_type_label =
          overflow ? kOverflowLabel : escape_label_value(worker_type);
      entry->second = &families_.io_copy_latency->Add(
          {{"direction", direction_label},
           {"worker_id", worker_id_label},
           {"device", device_label},
           {"worker_type", worker_type_label}},
          kInferenceLatencyMsBuckets);
    }
  }
  if (entry->second != nullptr) {
    entry->second->Observe(duration_ms);
  }
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
  auto entry = caches_.io.transfer_bytes.find(key);
  if (entry == caches_.io.transfer_bytes.end()) {
    const bool overflow = caches_.io.transfer_bytes.size() >= kMaxLabelSeries;
    IoKey map_key = overflow ? IoKey::Overflow() : std::move(key);
    auto [inserted_it, inserted] =
        caches_.io.transfer_bytes.try_emplace(std::move(map_key), nullptr);
    entry = inserted_it;
    if (inserted) {
      const std::string direction_label =
          overflow ? kOverflowLabel : escape_label_value(direction);
      const std::string worker_id_label =
          overflow ? kOverflowLabel : std::to_string(worker_id);
      const std::string device_label =
          overflow ? kOverflowLabel : std::to_string(device_id);
      const std::string worker_type_label =
          overflow ? kOverflowLabel : escape_label_value(worker_type);
      entry->second = &families_.transfer_bytes->Add(
          {{"direction", direction_label},
           {"worker_id", worker_id_label},
           {"device", device_label},
           {"worker_type", worker_type_label}});
    }
  }
  if (entry->second != nullptr) {
    entry->second->Increment(static_cast<double>(bytes));
  }
}

void
increment_inference_completed(
    std::string_view model_name, std::size_t logical_jobs)
{
  auto metrics_ptr = metrics_atomic().load(std::memory_order_acquire);
  if (metrics_ptr == nullptr) {
    return;
  }
  metrics_ptr->increment_completed_counter(model_name, logical_jobs);
}

void
increment_inference_failure(
    std::string_view stage, std::string_view reason,
    std::string_view model_name, std::size_t count)
{
  auto metrics_ptr = metrics_atomic().load(std::memory_order_acquire);
  if (metrics_ptr == nullptr) {
    return;
  }
  metrics_ptr->increment_failure_counter(
      MetricsRegistry::FailureStageLabel{stage},
      MetricsRegistry::FailureReasonLabel{reason},
      MetricsRegistry::ModelLabel{model_name}, count);
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
  std::scoped_lock<std::mutex> lock(mutexes_.sampling);
  perform_sampling_request_nb();
}
#endif  // SONAR_IGNORE_END

void
MetricsRegistry::sample_cpu_usage()
{
  if (gauges_.system_cpu_usage_percent == nullptr) {
    return;
  }
  if (!providers_.cpu_usage_provider) {
    gauges_.system_cpu_usage_percent->Set(
        std::numeric_limits<double>::quiet_NaN());
    return;
  }

  try {
    auto usage = providers_.cpu_usage_provider();
    if (usage.has_value()) {
      gauges_.system_cpu_usage_percent->Set(*usage);
    } else {
      gauges_.system_cpu_usage_percent->Set(
          std::numeric_limits<double>::quiet_NaN());
    }
  }
  catch (const std::exception& e) {
    if (should_log_sampling_error(cpu_sampling_error_log_ts())) {
      log_error(std::format("CPU metrics sampling failed: {}", e.what()));
    }
    gauges_.system_cpu_usage_percent->Set(
        std::numeric_limits<double>::quiet_NaN());
  }
  catch (...) {
    if (should_log_sampling_error(cpu_sampling_error_log_ts())) {
      log_error("CPU metrics sampling failed due to an unknown error");
    }
    gauges_.system_cpu_usage_percent->Set(
        std::numeric_limits<double>::quiet_NaN());
  }
}

void
MetricsRegistry::sample_inference_throughput()
{
  if (gauges_.inference_throughput == nullptr) {
    return;
  }

  if (auto snap = perf_observer::snapshot()) {
    gauges_.inference_throughput->Set(snap->throughput);
  }
}

void
MetricsRegistry::sample_process_resident_memory()
{
  if (gauges_.process_resident_memory_bytes == nullptr) {
    return;
  }

  if (auto rss_bytes = monitoring::detail::read_process_rss_bytes();
      rss_bytes.has_value()) {
    gauges_.process_resident_memory_bytes->Set(*rss_bytes);
  } else {
    gauges_.process_resident_memory_bytes->Set(
        std::numeric_limits<double>::quiet_NaN());
  }
}

void
MetricsRegistry::sample_process_open_fds()
{
  if (gauges_.process_open_fds == nullptr) {
    return;
  }

  if (auto fds = monitoring::detail::read_process_open_fds(); fds.has_value()) {
    gauges_.process_open_fds->Set(*fds);
  } else {
    gauges_.process_open_fds->Set(std::numeric_limits<double>::quiet_NaN());
  }
}

void
MetricsRegistry::sample_gpu_stats()
{
  if (!providers_.gpu_stats_provider) {
    return;
  }

  try {
    auto gstats = providers_.gpu_stats_provider();
    std::unordered_set<int> seen_indices;
    seen_indices.reserve(gstats.size());
    for (const auto& stats : gstats) {
      seen_indices.insert(stats.index);
      const std::string label = std::to_string(stats.index);

      ensure_gpu_gauge(
          caches_.gpu.utilization, families_.gpu_utilization, stats.index,
          label)
          ->Set(stats.util_percent);
      ensure_gpu_gauge(
          caches_.gpu.memory_used, families_.gpu_memory_used_bytes, stats.index,
          label)
          ->Set(stats.mem_used_bytes);
      ensure_gpu_gauge(
          caches_.gpu.memory_total, families_.gpu_memory_total_bytes,
          stats.index, label)
          ->Set(stats.mem_total_bytes);

      set_or_clear_nan(
          caches_.gpu.temperature, families_.gpu_temperature, stats.index,
          label, stats.temperature_celsius);
      set_or_clear_nan(
          caches_.gpu.power, families_.gpu_power, stats.index, label,
          stats.power_watts);
    }

    clear_missing_gauges(
        caches_.gpu.utilization, families_.gpu_utilization, seen_indices);
    clear_missing_gauges(
        caches_.gpu.memory_used, families_.gpu_memory_used_bytes, seen_indices);
    clear_missing_gauges(
        caches_.gpu.memory_total, families_.gpu_memory_total_bytes,
        seen_indices);
    clear_missing_gauges(
        caches_.gpu.temperature, families_.gpu_temperature, seen_indices);
    clear_missing_gauges(caches_.gpu.power, families_.gpu_power, seen_indices);
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
MetricsRegistry::perform_sampling_request_nb()
{
  sample_cpu_usage();
  sample_inference_throughput();
  sample_process_resident_memory();
  sample_process_open_fds();
  sample_gpu_stats();
}

void
MetricsRegistry::sampling_loop(const std::stop_token& stop)
{
  using namespace std::chrono_literals;
  auto next_sleep = 1000ms;
  while (!stop.stop_requested()) {
    {
      std::scoped_lock<std::mutex> lock(mutexes_.sampling);
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
starpu_server::MetricsRegistry::TestAccessor::ClearCpuUsageProvider(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.providers_.cpu_usage_provider = {};
}

void
starpu_server::MetricsRegistry::TestAccessor::ClearSystemCpuUsageGauge(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.gauges_.system_cpu_usage_percent = nullptr;
}

void
starpu_server::MetricsRegistry::TestAccessor::ClearProcessOpenFdsGauge(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.gauges_.process_open_fds = nullptr;
}

void
starpu_server::MetricsRegistry::TestAccessor::ClearProcessResidentMemoryGauge(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.gauges_.process_resident_memory_bytes = nullptr;
}

void
starpu_server::MetricsRegistry::TestAccessor::ClearInferenceThroughputGauge(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.gauges_.inference_throughput = nullptr;
}

void
starpu_server::MetricsRegistry::TestAccessor::ClearGpuStatsProvider(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.providers_.gpu_stats_provider = {};
}

auto
starpu_server::MetricsRegistry::TestAccessor::ProcessOpenFdsGauge(
    starpu_server::MetricsRegistry& metrics) -> prometheus::Gauge*
{
  return metrics.gauges_.process_open_fds;
}

auto
starpu_server::MetricsRegistry::TestAccessor::ProcessResidentMemoryGauge(
    starpu_server::MetricsRegistry& metrics) -> prometheus::Gauge*
{
  return metrics.gauges_.process_resident_memory_bytes;
}

auto
starpu_server::MetricsRegistry::TestAccessor::InferenceThroughputGauge(
    starpu_server::MetricsRegistry& metrics) -> prometheus::Gauge*
{
  return metrics.gauges_.inference_throughput;
}

auto
starpu_server::MetricsRegistry::TestAccessor::GpuUtilizationGaugeCount(
    const starpu_server::MetricsRegistry& metrics) -> std::size_t
{
  std::scoped_lock<std::mutex> lock(metrics.mutexes_.sampling);
  return metrics.caches_.gpu.utilization.size();
}

auto
starpu_server::MetricsRegistry::TestAccessor::GpuMemoryUsedGaugeCount(
    const starpu_server::MetricsRegistry& metrics) -> std::size_t
{
  std::scoped_lock<std::mutex> lock(metrics.mutexes_.sampling);
  return metrics.caches_.gpu.memory_used.size();
}

auto
starpu_server::MetricsRegistry::TestAccessor::GpuMemoryTotalGaugeCount(
    const starpu_server::MetricsRegistry& metrics) -> std::size_t
{
  std::scoped_lock<std::mutex> lock(metrics.mutexes_.sampling);
  return metrics.caches_.gpu.memory_total.size();
}

void
starpu_server::MetricsRegistry::TestAccessor::SampleProcessOpenFds(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.sample_process_open_fds();
}

void
starpu_server::MetricsRegistry::TestAccessor::SampleProcessResidentMemory(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.sample_process_resident_memory();
}

void
starpu_server::MetricsRegistry::TestAccessor::SampleInferenceThroughput(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.sample_inference_throughput();
}

void
starpu_server::MetricsRegistry::TestAccessor::ClearStarpuWorkerInflightFamily(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.starpu_worker_inflight = nullptr;
}

void
starpu_server::MetricsRegistry::TestAccessor::
    ClearStarpuTaskRuntimeByWorkerFamily(
        starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.starpu_task_runtime_by_worker = nullptr;
}

void
starpu_server::MetricsRegistry::TestAccessor::
    ClearInferenceComputeLatencyByWorkerFamily(
        starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.inference_compute_latency_by_worker = nullptr;
}

void
starpu_server::MetricsRegistry::TestAccessor::ClearIoCopyLatencyFamily(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.io_copy_latency = nullptr;
}

void
starpu_server::MetricsRegistry::TestAccessor::ClearTransferBytesFamily(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.transfer_bytes = nullptr;
}

void
starpu_server::MetricsRegistry::TestAccessor::ClearModelsLoadedFamily(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.models_loaded = nullptr;
}

void
starpu_server::MetricsRegistry::TestAccessor::ClearModelLoadFailuresFamily(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.model_load_failures = nullptr;
}

void
starpu_server::MetricsRegistry::TestAccessor::ClearInferenceFailuresFamily(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.inference_failures = nullptr;
}

void
starpu_server::MetricsRegistry::TestAccessor::ClearInferenceCompletedFamily(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.inference_completed = nullptr;
}

void
starpu_server::MetricsRegistry::TestAccessor::ClearRequestsByStatusFamily(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.requests_by_status = nullptr;
}

auto
starpu_server::MetricsRegistry::TestAccessor::FailureKeyOverflowIsEmpty()
    -> bool
{
  const auto key = FailureKey::Overflow();
  return key.overflow && key.stage.empty() && key.reason.empty() &&
         key.model.empty();
}

auto
starpu_server::MetricsRegistry::TestAccessor::FailureKeyEquals(
    std::string_view stage_lhs, std::string_view reason_lhs,
    std::string_view model_lhs, bool overflow_lhs, std::string_view stage_rhs,
    std::string_view reason_rhs, std::string_view model_rhs,
    bool overflow_rhs) -> bool
{
  FailureKey lhs{
      std::string(stage_lhs), std::string(reason_lhs), std::string(model_lhs),
      overflow_lhs};
  FailureKey rhs{
      std::string(stage_rhs), std::string(reason_rhs), std::string(model_rhs),
      overflow_rhs};
  return lhs == rhs;
}

auto
starpu_server::MetricsRegistry::TestAccessor::ModelKeyOverflowIsEmpty() -> bool
{
  const auto key = ModelKey::Overflow();
  return key.overflow && key.model.empty();
}

auto
starpu_server::MetricsRegistry::TestAccessor::ModelKeyEquals(
    std::string_view model_lhs, bool overflow_lhs, std::string_view model_rhs,
    bool overflow_rhs) -> bool
{
  ModelKey lhs{std::string(model_lhs), overflow_lhs};
  ModelKey rhs{std::string(model_rhs), overflow_rhs};
  return lhs == rhs;
}

auto
starpu_server::MetricsRegistry::TestAccessor::ModelDeviceKeyOverflowIsEmpty()
    -> bool
{
  const auto key = ModelDeviceKey::Overflow();
  return key.overflow && key.model.empty() && key.device.empty();
}

auto
starpu_server::MetricsRegistry::TestAccessor::ModelDeviceKeyEquals(
    std::string_view model_lhs, std::string_view device_lhs, bool overflow_lhs,
    std::string_view model_rhs, std::string_view device_rhs,
    bool overflow_rhs) -> bool
{
  ModelDeviceKey lhs{
      std::string(model_lhs), std::string(device_lhs), overflow_lhs};
  ModelDeviceKey rhs{
      std::string(model_rhs), std::string(device_rhs), overflow_rhs};
  return lhs == rhs;
}

auto
starpu_server::MetricsRegistry::TestAccessor::IoKeyOverflowIsEmpty() -> bool
{
  const auto key = IoKey::Overflow();
  return key.overflow && key.direction.empty() && key.worker_id == 0 &&
         key.device_id == 0 && key.worker_type.empty();
}

auto
starpu_server::MetricsRegistry::TestAccessor::IoKeyEquals(
    std::string_view direction_lhs, int worker_id_lhs, int device_id_lhs,
    std::string_view worker_type_lhs, bool overflow_lhs,
    std::string_view direction_rhs, int worker_id_rhs, int device_id_rhs,
    std::string_view worker_type_rhs, bool overflow_rhs) -> bool
{
  IoKey lhs{
      std::string(direction_lhs), worker_id_lhs, device_id_lhs,
      std::string(worker_type_lhs), overflow_lhs};
  IoKey rhs{
      std::string(direction_rhs), worker_id_rhs, device_id_rhs,
      std::string(worker_type_rhs), overflow_rhs};
  return lhs == rhs;
}

auto
starpu_server::MetricsRegistry::TestAccessor::WorkerKeyOverflowIsEmpty() -> bool
{
  const auto key = WorkerKey::Overflow();
  return key.overflow && key.worker_id == 0 && key.device_id == 0 &&
         key.worker_type.empty();
}

auto
starpu_server::MetricsRegistry::TestAccessor::WorkerKeyEquals(
    int worker_id_lhs, int device_id_lhs, std::string_view worker_type_lhs,
    bool overflow_lhs, int worker_id_rhs, int device_id_rhs,
    std::string_view worker_type_rhs, bool overflow_rhs) -> bool
{
  WorkerKey lhs{
      worker_id_lhs, device_id_lhs, std::string(worker_type_lhs), overflow_lhs};
  WorkerKey rhs{
      worker_id_rhs, device_id_rhs, std::string(worker_type_rhs), overflow_rhs};
  return lhs == rhs;
}
#endif  // SONAR_IGNORE_END
