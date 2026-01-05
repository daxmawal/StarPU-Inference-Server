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

#if defined(STARPU_TESTING)
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
#endif

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
#if defined(STARPU_TESTING)
constexpr auto kSamplingErrorLogThrottle = std::chrono::seconds(0);
#else
constexpr auto kSamplingErrorLogThrottle = std::chrono::seconds(60);
#endif

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
cpu_sampling_error_log_ts() -> std::atomic<std::int64_t>&
{
  static std::atomic<std::int64_t> ts{0};
  return ts;
}

auto
gpu_sampling_error_log_ts() -> std::atomic<std::int64_t>&
{
  static std::atomic<std::int64_t> ts{0};
  return ts;
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

#if defined(STARPU_TESTING)
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
#endif

auto
read_process_rss_bytes_impl() -> std::optional<double>
{
#if defined(STARPU_TESTING)
  auto& override_reader = process_rss_bytes_reader_override_storage();
  if (override_reader) {
    return override_reader();
  }
  const std::filesystem::path& path =
      monitoring::detail::process_rss_bytes_path_for_test();
#else
  const std::filesystem::path& path = kProcStatm;
#endif
  std::ifstream statm{path};
  if (!statm.is_open()) {
    return std::nullopt;
  }
  unsigned long size = 0;
  unsigned long resident = 0;
  if (!(statm >> size >> resident)) {
    return std::nullopt;
  }
#if defined(STARPU_TESTING)
  const auto& page_size_provider =
      monitoring::detail::process_page_size_provider_for_test();
  const long page_size =
      page_size_provider ? page_size_provider() : sysconf(_SC_PAGESIZE);
#else
  const long page_size = sysconf(_SC_PAGESIZE);
#endif
  if (page_size <= 0) {
    return std::nullopt;
  }
  return static_cast<double>(resident) * static_cast<double>(page_size);
}

auto
read_process_open_fds_impl() -> std::optional<double>
{
#if defined(STARPU_TESTING)
  auto& override_reader = process_open_fds_reader_override_storage();
  if (override_reader) {
    return override_reader();
  }
#endif
  static const std::filesystem::path kProcFd{"/proc/self/fd"};
#if defined(STARPU_TESTING)
  const std::filesystem::path& path =
      monitoring::detail::process_fd_path_for_test();
#else
  const std::filesystem::path& path = kProcFd;
#endif
  try {
    if (!std::filesystem::exists(path)) {
      return std::nullopt;
    }
#if defined(STARPU_TESTING)
    const auto factory =
        monitoring::detail::process_fd_directory_iterator_for_test();
    std::filesystem::directory_iterator iter =
        factory ? factory(path) : std::filesystem::directory_iterator(path);
#else
    std::filesystem::directory_iterator iter(path);
#endif
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

#if defined(STARPU_TESTING)
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
#endif

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

#if defined(STARPU_TESTING)
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
#endif

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
    : registry_(std::make_shared<prometheus::Registry>()), exposer_(nullptr),
      gpu_stats_provider_(std::move(gpu_provider)),
      cpu_usage_provider_(std::move(cpu_provider))
{
#if defined(STARPU_TESTING)
  if (monitoring::detail::metrics_init_failure_for_test()) {
    throw std::runtime_error("forced metrics initialization failure");
  }
#endif
  if (!gpu_stats_provider_) {
    gpu_stats_provider_ = query_gpu_stats_nvml;
  }
  if (!cpu_usage_provider_) {
    cpu_usage_provider_ = make_default_cpu_usage_provider();
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
#if defined(STARPU_TESTING)
      exposer_handle = std::make_unique<TestingExposerHandle>();
#else
      auto exposer = std::make_unique<prometheus::Exposer>(
          std::format("0.0.0.0:{}", port));
      exposer_handle =
          std::make_unique<PrometheusExposerHandle>(std::move(exposer));
#endif
    }
    exposer_handle->RegisterCollectable(registry_);
    exposer_ = std::move(exposer_handle);
  }
  catch (const std::exception& e) {
    log_error(std::string("Failed to initialize metrics exposer: ") + e.what());
    throw;
  }

  auto& counter_family = prometheus::BuildCounter()
                             .Name("requests_total")
                             .Help("Total requests received")
                             .Register(*registry_);
  requests_total_ = &counter_family.Add({});

  auto& status_family = prometheus::BuildCounter()
                            .Name("requests_by_status_total")
                            .Help(
                                "Total requests grouped by gRPC status code "
                                "and model name")
                            .Register(*registry_);
  requests_by_status_family_ = &status_family;
  (void)requests_by_status_family_->Add(
      {{"code", "unlabeled"}, {"model", "unlabeled"}});

  auto& completed_family = prometheus::BuildCounter()
                               .Name("inference_completed_total")
                               .Help("Total logical inferences completed")
                               .Register(*registry_);
  inference_completed_family_ = &completed_family;
  (void)inference_completed_family_->Add({{"model", "unlabeled"}});

  auto& failures_family = prometheus::BuildCounter()
                              .Name("inference_failures_total")
                              .Help(
                                  "Inference failures grouped by stage and "
                                  "reason")
                              .Register(*registry_);
  inference_failures_family_ = &failures_family;
  (void)inference_failures_family_->Add(
      {{"stage", "unlabeled"},
       {"reason", "unlabeled"},
       {"model", "unlabeled"}});

  auto& rejected_family =
      prometheus::BuildCounter()
          .Name("requests_rejected_total")
          .Help("Total requests rejected (e.g., queue full)")
          .Register(*registry_);
  requests_rejected_total_ = &rejected_family.Add({});

  auto& histogram_family = prometheus::BuildHistogram()
                               .Name("inference_latency_ms")
                               .Help("Inference latency in milliseconds")
                               .Register(*registry_);
  inference_latency_ = &histogram_family.Add({}, kInferenceLatencyMsBuckets);

  auto& gauge_family = prometheus::BuildGauge()
                           .Name("inference_queue_size")
                           .Help("Number of jobs in the inference queue")
                           .Register(*registry_);
  queue_size_gauge_ = &gauge_family.Add({});

  auto& queue_capacity_family = prometheus::BuildGauge()
                                    .Name("inference_max_queue_size")
                                    .Help(
                                        "Configured maximum inference queue "
                                        "capacity")
                                    .Register(*registry_);
  queue_capacity_gauge_ = &queue_capacity_family.Add({});

  auto& queue_fill_family = prometheus::BuildGauge()
                                .Name("inference_queue_fill_ratio")
                                .Help(
                                    "Queue occupancy ratio "
                                    "(queue_size / max_queue_size)")
                                .Register(*registry_);
  queue_fill_ratio_gauge_ = &queue_fill_family.Add({});

  auto& inflight_family = prometheus::BuildGauge()
                              .Name("inference_inflight_tasks")
                              .Help(
                                  "Number of StarPU tasks currently submitted "
                                  "and not yet completed")
                              .Register(*registry_);
  inflight_tasks_gauge_ = &inflight_family.Add({});

  auto& inflight_cap_family = prometheus::BuildGauge()
                                  .Name("inference_max_inflight_tasks")
                                  .Help(
                                      "Configured cap on inflight StarPU tasks "
                                      "(0 means unbounded)")
                                  .Register(*registry_);
  max_inflight_tasks_gauge_ = &inflight_cap_family.Add({});

  auto& worker_busy_family = prometheus::BuildGauge()
                                 .Name("starpu_worker_busy_ratio")
                                 .Help(
                                     "Approximate ratio of inflight tasks to "
                                     "max inflight limit (0-1, 0 when "
                                     "unbounded)")
                                 .Register(*registry_);
  starpu_worker_busy_ratio_ = &worker_busy_family.Add({});

  auto& prepared_queue_family = prometheus::BuildGauge()
                                    .Name("starpu_prepared_queue_depth")
                                    .Help(
                                        "Number of batched jobs waiting for "
                                        "StarPU submission")
                                    .Register(*registry_);
  starpu_prepared_queue_depth_ = &prepared_queue_family.Add({});

  auto& cpu_family = prometheus::BuildGauge()
                         .Name("system_cpu_usage_percent")
                         .Help("System-wide CPU utilization percentage (0-100)")
                         .Register(*registry_);
  system_cpu_usage_percent_ = &cpu_family.Add({});

  auto& throughput_family = prometheus::BuildGauge()
                                .Name("inference_throughput_rps")
                                .Help(
                                    "Rolling throughput of logical "
                                    "inferences/s based on completed jobs")
                                .Register(*registry_);
  inference_throughput_gauge_ = &throughput_family.Add({});

  auto& rss_family = prometheus::BuildGauge()
                         .Name("process_resident_memory_bytes")
                         .Help("Resident Set Size of the server process")
                         .Register(*registry_);
  process_resident_memory_bytes_ = &rss_family.Add({});

  auto& fds_family = prometheus::BuildGauge()
                         .Name("process_open_fds")
                         .Help("Number of open file descriptors")
                         .Register(*registry_);
  process_open_fds_ = &fds_family.Add({});

  auto& health_family = prometheus::BuildGauge()
                            .Name("server_health_state")
                            .Help(
                                "Server health state: 1=ready, 0=not ready or "
                                "shutting down")
                            .Register(*registry_);
  server_health_state_ = &health_family.Add({});

  auto& queue_latency_family = prometheus::BuildHistogram()
                                   .Name("inference_queue_latency_ms")
                                   .Help("Time spent waiting in the queue")
                                   .Register(*registry_);
  queue_latency_histogram_ =
      &queue_latency_family.Add({}, kInferenceLatencyMsBuckets);

  auto& batch_efficiency_family = prometheus::BuildHistogram()
                                      .Name("inference_batch_efficiency_ratio")
                                      .Help(
                                          "Ratio of effective batch size to "
                                          "logical request count")
                                      .Register(*registry_);
  batch_efficiency_histogram_ =
      &batch_efficiency_family.Add({}, kBatchEfficiencyBuckets);

  auto& batch_pending_family = prometheus::BuildGauge()
                                   .Name("inference_batch_collect_pending_jobs")
                                   .Help(
                                       "Number of requests aggregated in the "
                                       "current batch collection")
                                   .Register(*registry_);
  batch_pending_jobs_gauge_ = &batch_pending_family.Add({});

  auto& batch_collect_family = prometheus::BuildHistogram()
                                   .Name("inference_batch_collect_ms")
                                   .Help("Time spent collecting a batch")
                                   .Register(*registry_);
  batch_collect_latency_histogram_ =
      &batch_collect_family.Add({}, kInferenceLatencyMsBuckets);

  auto& submit_family = prometheus::BuildHistogram()
                            .Name("inference_submit_latency_ms")
                            .Help(
                                "Time spent between dequeue and submission "
                                "into StarPU")
                            .Register(*registry_);
  submit_latency_histogram_ =
      &submit_family.Add({}, kInferenceLatencyMsBuckets);

  auto& scheduling_family = prometheus::BuildHistogram()
                                .Name("inference_scheduling_latency_ms")
                                .Help(
                                    "Time spent waiting for scheduling on a "
                                    "StarPU worker")
                                .Register(*registry_);
  scheduling_latency_histogram_ =
      &scheduling_family.Add({}, kInferenceLatencyMsBuckets);

  auto& codelet_family = prometheus::BuildHistogram()
                             .Name("inference_codelet_latency_ms")
                             .Help("Duration of the StarPU codelet execution")
                             .Register(*registry_);
  codelet_latency_histogram_ =
      &codelet_family.Add({}, kInferenceLatencyMsBuckets);

  auto& compute_family = prometheus::BuildHistogram()
                             .Name("inference_compute_latency_ms")
                             .Help("Model compute time (inference)")
                             .Register(*registry_);
  inference_compute_latency_histogram_ =
      &compute_family.Add({}, kInferenceLatencyMsBuckets);

  auto& callback_family = prometheus::BuildHistogram()
                              .Name("inference_callback_latency_ms")
                              .Help("Callback/response handling latency")
                              .Register(*registry_);
  callback_latency_histogram_ =
      &callback_family.Add({}, kInferenceLatencyMsBuckets);

  auto& preprocess_family = prometheus::BuildHistogram()
                                .Name("inference_preprocess_latency_ms")
                                .Help("Server-side preprocessing latency")
                                .Register(*registry_);
  preprocess_latency_histogram_ =
      &preprocess_family.Add({}, kInferenceLatencyMsBuckets);

  auto& postprocess_family = prometheus::BuildHistogram()
                                 .Name("inference_postprocess_latency_ms")
                                 .Help("Server-side postprocessing latency")
                                 .Register(*registry_);
  postprocess_latency_histogram_ =
      &postprocess_family.Add({}, kInferenceLatencyMsBuckets);

  auto& batch_size_family = prometheus::BuildHistogram()
                                .Name("inference_batch_size")
                                .Help("Effective batch size executed")
                                .Register(*registry_);
  batch_size_histogram_ = &batch_size_family.Add({}, kBatchSizeBuckets);

  auto& logical_batch_size_family = prometheus::BuildHistogram()
                                        .Name("inference_logical_batch_size")
                                        .Help(
                                            "Number of logical requests "
                                            "aggregated into a batch")
                                        .Register(*registry_);
  logical_batch_size_histogram_ =
      &logical_batch_size_family.Add({}, kBatchSizeBuckets);

  auto& model_load_hist_family = prometheus::BuildHistogram()
                                     .Name("model_load_duration_ms")
                                     .Help("Duration of model load and wiring")
                                     .Register(*registry_);
  model_load_duration_histogram_ =
      &model_load_hist_family.Add({}, kModelLoadDurationMsBuckets);

  auto& model_load_fail_family = prometheus::BuildCounter()
                                     .Name("model_load_failures_total")
                                     .Help("Total failed model load attempts")
                                     .Register(*registry_);
  model_load_failures_family_ = &model_load_fail_family;
  (void)model_load_failures_family_->Add({{"model", "unlabeled"}});

  auto& models_loaded_family =
      prometheus::BuildGauge()
          .Name("models_loaded")
          .Help("Flag indicating a model is loaded on a device")
          .Register(*registry_);
  models_loaded_family_ = &models_loaded_family;
  (void)models_loaded_family_->Add(
      {{"model", "unlabeled"}, {"device", "unknown"}});

  gpu_utilization_family_ =
      &prometheus::BuildGauge()
           .Name("gpu_utilization_percent")
           .Help("GPU utilization percentage per GPU (0-100)")
           .Register(*registry_);
  gpu_memory_used_bytes_family_ = &prometheus::BuildGauge()
                                       .Name("gpu_memory_used_bytes")
                                       .Help("Used GPU memory in bytes per GPU")
                                       .Register(*registry_);
  gpu_memory_total_bytes_family_ =
      &prometheus::BuildGauge()
           .Name("gpu_memory_total_bytes")
           .Help("Total GPU memory in bytes per GPU")
           .Register(*registry_);

  gpu_temperature_family_ = &prometheus::BuildGauge()
                                 .Name("gpu_temperature_celsius")
                                 .Help("Reported GPU temperature in Celsius")
                                 .Register(*registry_);

  gpu_power_family_ = &prometheus::BuildGauge()
                           .Name("gpu_power_watts")
                           .Help("Reported GPU power draw in Watts")
                           .Register(*registry_);

  auto& starpu_runtime_family = prometheus::BuildHistogram()
                                    .Name("starpu_task_runtime_ms")
                                    .Help("Wall-clock runtime of a StarPU task")
                                    .Register(*registry_);
  starpu_task_runtime_histogram_ =
      &starpu_runtime_family.Add({}, kTaskRuntimeMsBuckets);

  auto& compute_latency_by_worker_family =
      prometheus::BuildHistogram()
          .Name("inference_compute_latency_ms_by_worker")
          .Help(
              "Model compute latency by worker and device "
              "(callback start - inference start)")
          .Register(*registry_);
  inference_compute_latency_by_worker_family_ =
      &compute_latency_by_worker_family;

  auto& starpu_runtime_by_worker_family =
      prometheus::BuildHistogram()
          .Name("starpu_task_runtime_ms_by_worker")
          .Help("StarPU codelet runtime by worker/device")
          .Register(*registry_);
  starpu_task_runtime_by_worker_family_ = &starpu_runtime_by_worker_family;

  auto& worker_inflight_family =
      prometheus::BuildGauge()
          .Name("starpu_worker_inflight_tasks")
          .Help("Inflight StarPU tasks per worker/device")
          .Register(*registry_);
  starpu_worker_inflight_family_ = &worker_inflight_family;

  auto& io_copy_latency_family =
      prometheus::BuildHistogram()
          .Name("inference_io_copy_ms")
          .Help("Host/device copy latency by direction/device/worker")
          .Register(*registry_);
  io_copy_latency_family_ = &io_copy_latency_family;

  auto& transfer_bytes_family =
      prometheus::BuildCounter()
          .Name("inference_transfer_bytes_total")
          .Help("Total bytes transferred by direction/device/worker")
          .Register(*registry_);
  transfer_bytes_family_ = &transfer_bytes_family;

  if (start_sampler_thread) {
    sampler_thread_ = std::jthread(
        [this](const std::stop_token& stop) { this->sampling_loop(stop); });
  }
}

MetricsRegistry::~MetricsRegistry() noexcept
{
  request_stop();
  if (sampler_thread_.joinable()) {
    sampler_thread_.join();
  }
  if (exposer_ && registry_) {
    try {
      exposer_->RemoveCollectable(registry_);
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
#if defined(STARPU_TESTING)
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
#endif
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
  if (sampler_thread_.joinable()) {
    sampler_thread_.request_stop();
  }
}

#if defined(STARPU_TESTING)
auto
MetricsRegistry::has_gpu_stats_provider() const -> bool
{
  return static_cast<bool>(gpu_stats_provider_);
}

auto
MetricsRegistry::has_cpu_usage_provider() const -> bool
{
  return static_cast<bool>(cpu_usage_provider_);
}
#endif

auto
MetricsRegistry::registry() const -> std::shared_ptr<prometheus::Registry>
{
  return registry_;
}

auto
MetricsRegistry::requests_total() const -> prometheus::Counter*
{
  return requests_total_;
}

auto
MetricsRegistry::requests_rejected_total() const -> prometheus::Counter*
{
  return requests_rejected_total_;
}

auto
MetricsRegistry::inference_latency() const -> prometheus::Histogram*
{
  return inference_latency_;
}

auto
MetricsRegistry::queue_size_gauge() const -> prometheus::Gauge*
{
  return queue_size_gauge_;
}

auto
MetricsRegistry::inflight_tasks_gauge() const -> prometheus::Gauge*
{
  return inflight_tasks_gauge_;
}

auto
MetricsRegistry::max_inflight_tasks_gauge() const -> prometheus::Gauge*
{
  return max_inflight_tasks_gauge_;
}

auto
MetricsRegistry::starpu_worker_busy_ratio_gauge() const -> prometheus::Gauge*
{
  return starpu_worker_busy_ratio_;
}

auto
MetricsRegistry::starpu_prepared_queue_depth_gauge() const -> prometheus::Gauge*
{
  return starpu_prepared_queue_depth_;
}

auto
MetricsRegistry::system_cpu_usage_percent() const -> prometheus::Gauge*
{
  return system_cpu_usage_percent_;
}

auto
MetricsRegistry::inference_throughput_gauge() const -> prometheus::Gauge*
{
  return inference_throughput_gauge_;
}

auto
MetricsRegistry::process_resident_memory_gauge() const -> prometheus::Gauge*
{
  return process_resident_memory_bytes_;
}

auto
MetricsRegistry::process_open_fds_gauge() const -> prometheus::Gauge*
{
  return process_open_fds_;
}

auto
MetricsRegistry::server_health_state_gauge() const -> prometheus::Gauge*
{
  return server_health_state_;
}

auto
MetricsRegistry::queue_fill_ratio_gauge() const -> prometheus::Gauge*
{
  return queue_fill_ratio_gauge_;
}

auto
MetricsRegistry::queue_capacity_gauge() const -> prometheus::Gauge*
{
  return queue_capacity_gauge_;
}

auto
MetricsRegistry::queue_latency_histogram() const -> prometheus::Histogram*
{
  return queue_latency_histogram_;
}

auto
MetricsRegistry::batch_collect_latency_histogram() const
    -> prometheus::Histogram*
{
  return batch_collect_latency_histogram_;
}

auto
MetricsRegistry::batch_efficiency_histogram() const -> prometheus::Histogram*
{
  return batch_efficiency_histogram_;
}

auto
MetricsRegistry::batch_pending_jobs_gauge() const -> prometheus::Gauge*
{
  return batch_pending_jobs_gauge_;
}

auto
MetricsRegistry::submit_latency_histogram() const -> prometheus::Histogram*
{
  return submit_latency_histogram_;
}

auto
MetricsRegistry::scheduling_latency_histogram() const -> prometheus::Histogram*
{
  return scheduling_latency_histogram_;
}

auto
MetricsRegistry::codelet_latency_histogram() const -> prometheus::Histogram*
{
  return codelet_latency_histogram_;
}

auto
MetricsRegistry::inference_compute_latency_histogram() const
    -> prometheus::Histogram*
{
  return inference_compute_latency_histogram_;
}

auto
MetricsRegistry::callback_latency_histogram() const -> prometheus::Histogram*
{
  return callback_latency_histogram_;
}

auto
MetricsRegistry::preprocess_latency_histogram() const -> prometheus::Histogram*
{
  return preprocess_latency_histogram_;
}

auto
MetricsRegistry::postprocess_latency_histogram() const -> prometheus::Histogram*
{
  return postprocess_latency_histogram_;
}

auto
MetricsRegistry::batch_size_histogram() const -> prometheus::Histogram*
{
  return batch_size_histogram_;
}

auto
MetricsRegistry::logical_batch_size_histogram() const -> prometheus::Histogram*
{
  return logical_batch_size_histogram_;
}

auto
MetricsRegistry::model_load_duration_histogram() const -> prometheus::Histogram*
{
  return model_load_duration_histogram_;
}

auto
MetricsRegistry::starpu_task_runtime_histogram() const -> prometheus::Histogram*
{
  return starpu_task_runtime_histogram_;
}

auto
MetricsRegistry::inference_compute_latency_by_worker_family() const
    -> prometheus::Family<prometheus::Histogram>*
{
  return inference_compute_latency_by_worker_family_;
}

auto
MetricsRegistry::starpu_task_runtime_by_worker_family() const
    -> prometheus::Family<prometheus::Histogram>*
{
  return starpu_task_runtime_by_worker_family_;
}

auto
MetricsRegistry::starpu_worker_inflight_family() const
    -> prometheus::Family<prometheus::Gauge>*
{
  return starpu_worker_inflight_family_;
}

auto
MetricsRegistry::io_copy_latency_family() const
    -> prometheus::Family<prometheus::Histogram>*
{
  return io_copy_latency_family_;
}

auto
MetricsRegistry::transfer_bytes_family() const
    -> prometheus::Family<prometheus::Counter>*
{
  return transfer_bytes_family_;
}

auto
MetricsRegistry::gpu_utilization_family() const
    -> prometheus::Family<prometheus::Gauge>*
{
  return gpu_utilization_family_;
}

auto
MetricsRegistry::gpu_memory_used_bytes_family() const
    -> prometheus::Family<prometheus::Gauge>*
{
  return gpu_memory_used_bytes_family_;
}

auto
MetricsRegistry::gpu_memory_total_bytes_family() const
    -> prometheus::Family<prometheus::Gauge>*
{
  return gpu_memory_total_bytes_family_;
}

auto
MetricsRegistry::gpu_temperature_family() const
    -> prometheus::Family<prometheus::Gauge>*
{
  return gpu_temperature_family_;
}

auto
MetricsRegistry::gpu_power_family() const
    -> prometheus::Family<prometheus::Gauge>*
{
  return gpu_power_family_;
}

auto
MetricsRegistry::requests_by_status_family() const
    -> prometheus::Family<prometheus::Counter>*
{
  return requests_by_status_family_;
}

auto
MetricsRegistry::inference_completed_family() const
    -> prometheus::Family<prometheus::Counter>*
{
  return inference_completed_family_;
}

auto
MetricsRegistry::inference_failures_family() const
    -> prometheus::Family<prometheus::Counter>*
{
  return inference_failures_family_;
}

auto
MetricsRegistry::model_load_failures_family() const
    -> prometheus::Family<prometheus::Counter>*
{
  return model_load_failures_family_;
}

auto
MetricsRegistry::models_loaded_family() const
    -> prometheus::Family<prometheus::Gauge>*
{
  return models_loaded_family_;
}

void
MetricsRegistry::increment_status_counter(
    MetricsRegistry::StatusCodeLabel code_label,
    MetricsRegistry::ModelLabel model_label)
{
  if (requests_by_status_family_ == nullptr) {
    return;
  }

  StatusKey key{std::string(code_label.value), std::string(model_label.value)};

  std::lock_guard<std::mutex> lock(status_mutex_);
  auto it = status_counters_.find(key);
  if (it == status_counters_.end()) {
    const bool overflow = status_counters_.size() >= kMaxLabelSeries;
    StatusKey map_key = overflow ? StatusKey::Overflow() : std::move(key);
    auto [inserted_it, inserted] =
        status_counters_.try_emplace(std::move(map_key), nullptr);
    it = inserted_it;
    if (inserted) {
      const std::string code_label_value =
          overflow ? kOverflowLabel : escape_label_value(code_label.value);
      const std::string model_label_value =
          overflow ? kOverflowLabel : escape_label_value(model_label.value);
      it->second = &requests_by_status_family_->Add(
          {{"code", code_label_value}, {"model", model_label_value}});
    }
  }
  if (it->second != nullptr) {
    it->second->Increment();
  }
}

void
MetricsRegistry::increment_completed_counter(
    std::string_view model_label, std::size_t logical_jobs)
{
  if (inference_completed_family_ == nullptr) {
    return;
  }
  ModelKey key{std::string(model_label)};
  std::lock_guard<std::mutex> lock(model_metrics_mutex_);
  auto it = inference_completed_counters_.find(key);
  if (it == inference_completed_counters_.end()) {
    const bool overflow =
        inference_completed_counters_.size() >= kMaxLabelSeries;
    ModelKey map_key = overflow ? ModelKey::Overflow() : std::move(key);
    auto [inserted_it, inserted] =
        inference_completed_counters_.try_emplace(std::move(map_key), nullptr);
    it = inserted_it;
    if (inserted) {
      const std::string model_label_value =
          overflow ? kOverflowLabel : escape_label_value(model_label);
      it->second =
          &inference_completed_family_->Add({{"model", model_label_value}});
    }
  }
  if (it->second != nullptr) {
    it->second->Increment(static_cast<double>(logical_jobs));
  }
}

void
MetricsRegistry::increment_failure_counter(
    MetricsRegistry::FailureStageLabel stage_label,
    MetricsRegistry::FailureReasonLabel reason_label,
    MetricsRegistry::ModelLabel model_label, std::size_t count)
{
  if (inference_failures_family_ == nullptr) {
    return;
  }
  FailureKey key{
      std::string(stage_label.value), std::string(reason_label.value),
      std::string(model_label.value)};

  std::lock_guard<std::mutex> lock(model_metrics_mutex_);
  auto it = inference_failure_counters_.find(key);
  if (it == inference_failure_counters_.end()) {
    const bool overflow = inference_failure_counters_.size() >= kMaxLabelSeries;
    FailureKey map_key = overflow ? FailureKey::Overflow() : std::move(key);
    auto [inserted_it, inserted] =
        inference_failure_counters_.try_emplace(std::move(map_key), nullptr);
    it = inserted_it;
    if (inserted) {
      const std::string stage_label_value =
          overflow ? kOverflowLabel : escape_label_value(stage_label.value);
      const std::string reason_label_value =
          overflow ? kOverflowLabel : escape_label_value(reason_label.value);
      const std::string model_label_value =
          overflow ? kOverflowLabel : escape_label_value(model_label.value);
      it->second = &inference_failures_family_->Add(
          {{"stage", stage_label_value},
           {"reason", reason_label_value},
           {"model", model_label_value}});
    }
  }
  if (it->second != nullptr) {
    it->second->Increment(static_cast<double>(count));
  }
}

void
MetricsRegistry::increment_model_load_failure_counter(
    std::string_view model_label)
{
  if (model_load_failures_family_ == nullptr) {
    return;
  }
  ModelKey key{std::string(model_label)};
  std::lock_guard<std::mutex> lock(model_metrics_mutex_);
  auto it = model_load_failure_counters_.find(key);
  if (it == model_load_failure_counters_.end()) {
    const bool overflow =
        model_load_failure_counters_.size() >= kMaxLabelSeries;
    ModelKey map_key = overflow ? ModelKey::Overflow() : std::move(key);
    auto [inserted_it, inserted] =
        model_load_failure_counters_.try_emplace(std::move(map_key), nullptr);
    it = inserted_it;
    if (inserted) {
      const std::string model_label_value =
          overflow ? kOverflowLabel : escape_label_value(model_label);
      it->second =
          &model_load_failures_family_->Add({{"model", model_label_value}});
    }
  }
  if (it->second != nullptr) {
    it->second->Increment();
  }
}

void
MetricsRegistry::set_model_loaded_flag(
    MetricsRegistry::ModelLabel model_label,
    MetricsRegistry::DeviceLabel device_label, bool loaded)
{
  if (models_loaded_family_ == nullptr) {
    return;
  }
  ModelDeviceKey key{
      std::string(model_label.value), std::string(device_label.value)};

  std::lock_guard<std::mutex> lock(model_metrics_mutex_);
  auto it = models_loaded_gauges_.find(key);
  if (it == models_loaded_gauges_.end()) {
    const bool overflow = models_loaded_gauges_.size() >= kMaxLabelSeries;
    ModelDeviceKey map_key =
        overflow ? ModelDeviceKey::Overflow() : std::move(key);
    auto [inserted_it, inserted] =
        models_loaded_gauges_.try_emplace(std::move(map_key), nullptr);
    it = inserted_it;
    if (inserted) {
      const std::string model_label_value =
          overflow ? kOverflowLabel : escape_label_value(model_label.value);
      const std::string device_label_value =
          overflow ? kOverflowLabel : escape_label_value(device_label.value);
      it->second = &models_loaded_family_->Add(
          {{"model", model_label_value}, {"device", device_label_value}});
    }
  }
  if (it->second != nullptr) {
    it->second->Set(loaded ? 1.0 : 0.0);
  }
}

void
MetricsRegistry::observe_compute_latency_by_worker(
    int worker_id, int device_id, std::string_view worker_type,
    double latency_ms)
{
  if (latency_ms < 0.0 ||
      inference_compute_latency_by_worker_family_ == nullptr) {
    return;
  }
  WorkerKey key{worker_id, device_id, std::string(worker_type)};
  std::lock_guard<std::mutex> lock(worker_metrics_mutex_);
  auto it = compute_latency_by_worker_.find(key);
  if (it == compute_latency_by_worker_.end()) {
    const bool overflow = compute_latency_by_worker_.size() >= kMaxLabelSeries;
    WorkerKey map_key = overflow ? WorkerKey::Overflow() : std::move(key);
    auto [inserted_it, inserted] =
        compute_latency_by_worker_.try_emplace(std::move(map_key), nullptr);
    it = inserted_it;
    if (inserted) {
      const std::string worker_id_label =
          overflow ? kOverflowLabel : std::to_string(worker_id);
      const std::string device_label =
          overflow ? kOverflowLabel : std::to_string(device_id);
      const std::string worker_type_label =
          overflow ? kOverflowLabel : escape_label_value(worker_type);
      it->second = &inference_compute_latency_by_worker_family_->Add(
          {{"worker_id", worker_id_label},
           {"device", device_label},
           {"worker_type", worker_type_label}},
          kInferenceLatencyMsBuckets);
    }
  }
  if (it->second != nullptr) {
    it->second->Observe(latency_ms);
  }
}

void
MetricsRegistry::observe_task_runtime_by_worker(
    int worker_id, int device_id, std::string_view worker_type,
    double latency_ms)
{
  if (latency_ms < 0.0 || starpu_task_runtime_by_worker_family_ == nullptr) {
    return;
  }
  WorkerKey key{worker_id, device_id, std::string(worker_type)};
  std::lock_guard<std::mutex> lock(worker_metrics_mutex_);
  auto it = task_runtime_by_worker_.find(key);
  if (it == task_runtime_by_worker_.end()) {
    const bool overflow = task_runtime_by_worker_.size() >= kMaxLabelSeries;
    WorkerKey map_key = overflow ? WorkerKey::Overflow() : std::move(key);
    auto [inserted_it, inserted] =
        task_runtime_by_worker_.try_emplace(std::move(map_key), nullptr);
    it = inserted_it;
    if (inserted) {
      const std::string worker_id_label =
          overflow ? kOverflowLabel : std::to_string(worker_id);
      const std::string device_label =
          overflow ? kOverflowLabel : std::to_string(device_id);
      const std::string worker_type_label =
          overflow ? kOverflowLabel : escape_label_value(worker_type);
      it->second = &starpu_task_runtime_by_worker_family_->Add(
          {{"worker_id", worker_id_label},
           {"device", device_label},
           {"worker_type", worker_type_label}},
          kTaskRuntimeMsBuckets);
    }
  }
  if (it->second != nullptr) {
    it->second->Observe(latency_ms);
  }
}

void
MetricsRegistry::set_worker_inflight_gauge(
    int worker_id, int device_id, std::string_view worker_type,
    std::size_t value)
{
  if (starpu_worker_inflight_family_ == nullptr) {
    return;
  }
  WorkerKey key{worker_id, device_id, std::string(worker_type)};
  std::lock_guard<std::mutex> lock(worker_metrics_mutex_);
  auto it = worker_inflight_gauges_.find(key);
  if (it == worker_inflight_gauges_.end()) {
    const bool overflow = worker_inflight_gauges_.size() >= kMaxLabelSeries;
    WorkerKey map_key = overflow ? WorkerKey::Overflow() : std::move(key);
    auto [inserted_it, inserted] =
        worker_inflight_gauges_.try_emplace(std::move(map_key), nullptr);
    it = inserted_it;
    if (inserted) {
      const std::string worker_id_label =
          overflow ? kOverflowLabel : std::to_string(worker_id);
      const std::string device_label =
          overflow ? kOverflowLabel : std::to_string(device_id);
      const std::string worker_type_label =
          overflow ? kOverflowLabel : escape_label_value(worker_type);
      it->second = &starpu_worker_inflight_family_->Add(
          {{"worker_id", worker_id_label},
           {"device", device_label},
           {"worker_type", worker_type_label}});
    }
  }
  if (it->second != nullptr) {
    it->second->Set(static_cast<double>(value));
  }
}

void
MetricsRegistry::observe_io_copy_latency(
    std::string_view direction, int worker_id, int device_id,
    std::string_view worker_type, double duration_ms)
{
  if (duration_ms < 0.0 || io_copy_latency_family_ == nullptr) {
    return;
  }
  IoKey key{
      std::string(direction), worker_id, device_id, std::string(worker_type)};
  std::lock_guard<std::mutex> lock(io_metrics_mutex_);
  auto it = io_copy_latency_.find(key);
  if (it == io_copy_latency_.end()) {
    const bool overflow = io_copy_latency_.size() >= kMaxLabelSeries;
    IoKey map_key = overflow ? IoKey::Overflow() : std::move(key);
    auto [inserted_it, inserted] =
        io_copy_latency_.try_emplace(std::move(map_key), nullptr);
    it = inserted_it;
    if (inserted) {
      const std::string direction_label =
          overflow ? kOverflowLabel : escape_label_value(direction);
      const std::string worker_id_label =
          overflow ? kOverflowLabel : std::to_string(worker_id);
      const std::string device_label =
          overflow ? kOverflowLabel : std::to_string(device_id);
      const std::string worker_type_label =
          overflow ? kOverflowLabel : escape_label_value(worker_type);
      it->second = &io_copy_latency_family_->Add(
          {{"direction", direction_label},
           {"worker_id", worker_id_label},
           {"device", device_label},
           {"worker_type", worker_type_label}},
          kInferenceLatencyMsBuckets);
    }
  }
  if (it->second != nullptr) {
    it->second->Observe(duration_ms);
  }
}

void
MetricsRegistry::increment_transfer_bytes(
    std::string_view direction, int worker_id, int device_id,
    std::string_view worker_type, std::size_t bytes)
{
  if (bytes == 0 || transfer_bytes_family_ == nullptr) {
    return;
  }
  IoKey key{
      std::string(direction), worker_id, device_id, std::string(worker_type)};
  std::lock_guard<std::mutex> lock(io_metrics_mutex_);
  auto it = transfer_bytes_.find(key);
  if (it == transfer_bytes_.end()) {
    const bool overflow = transfer_bytes_.size() >= kMaxLabelSeries;
    IoKey map_key = overflow ? IoKey::Overflow() : std::move(key);
    auto [inserted_it, inserted] =
        transfer_bytes_.try_emplace(std::move(map_key), nullptr);
    it = inserted_it;
    if (inserted) {
      const std::string direction_label =
          overflow ? kOverflowLabel : escape_label_value(direction);
      const std::string worker_id_label =
          overflow ? kOverflowLabel : std::to_string(worker_id);
      const std::string device_label =
          overflow ? kOverflowLabel : std::to_string(device_id);
      const std::string worker_type_label =
          overflow ? kOverflowLabel : escape_label_value(worker_type);
      it->second = &transfer_bytes_family_->Add(
          {{"direction", direction_label},
           {"worker_id", worker_id_label},
           {"device", device_label},
           {"worker_type", worker_type_label}});
    }
  }
  if (it->second != nullptr) {
    it->second->Increment(static_cast<double>(bytes));
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
  queue_capacity_.store(capacity, std::memory_order_release);
}

auto
MetricsRegistry::queue_capacity_value() const -> std::size_t
{
  return queue_capacity_.load(std::memory_order_acquire);
}

#if defined(STARPU_TESTING)
void
MetricsRegistry::run_sampling_request_nb()
{
  std::scoped_lock<std::mutex> lock(sampling_mutex_);
  perform_sampling_request_nb();
}
#endif

void
MetricsRegistry::sample_cpu_usage()
{
  if (system_cpu_usage_percent_ == nullptr) {
    return;
  }
  if (!cpu_usage_provider_) {
    system_cpu_usage_percent_->Set(std::numeric_limits<double>::quiet_NaN());
    return;
  }

  try {
    auto usage = cpu_usage_provider_();
    if (usage.has_value()) {
      system_cpu_usage_percent_->Set(*usage);
    } else {
      system_cpu_usage_percent_->Set(std::numeric_limits<double>::quiet_NaN());
    }
  }
  catch (const std::exception& e) {
    if (should_log_sampling_error(cpu_sampling_error_log_ts())) {
      log_error(std::format("CPU metrics sampling failed: {}", e.what()));
    }
    system_cpu_usage_percent_->Set(std::numeric_limits<double>::quiet_NaN());
  }
  catch (...) {
    if (should_log_sampling_error(cpu_sampling_error_log_ts())) {
      log_error("CPU metrics sampling failed due to an unknown error");
    }
    system_cpu_usage_percent_->Set(std::numeric_limits<double>::quiet_NaN());
  }
}

void
MetricsRegistry::sample_inference_throughput()
{
  if (inference_throughput_gauge_ == nullptr) {
    return;
  }

  if (auto snap = perf_observer::snapshot()) {
    inference_throughput_gauge_->Set(snap->throughput);
  }
}

void
MetricsRegistry::sample_process_resident_memory()
{
  if (process_resident_memory_bytes_ == nullptr) {
    return;
  }

  if (auto rss_bytes = monitoring::detail::read_process_rss_bytes()) {
    process_resident_memory_bytes_->Set(*rss_bytes);
  } else {
    process_resident_memory_bytes_->Set(
        std::numeric_limits<double>::quiet_NaN());
  }
}

void
MetricsRegistry::sample_process_open_fds()
{
  if (process_open_fds_ == nullptr) {
    return;
  }

  if (auto fds = monitoring::detail::read_process_open_fds()) {
    process_open_fds_->Set(*fds);
  } else {
    process_open_fds_->Set(std::numeric_limits<double>::quiet_NaN());
  }
}

void
MetricsRegistry::sample_gpu_stats()
{
  if (!gpu_stats_provider_) {
    return;
  }

  try {
    auto gstats = gpu_stats_provider_();
    std::unordered_set<int> seen_indices;
    seen_indices.reserve(gstats.size());
    for (const auto& stats : gstats) {
      seen_indices.insert(stats.index);
      const std::string label = std::to_string(stats.index);

      const auto ensure_gauge = [&](auto& gauges,
                                    auto* family) -> prometheus::Gauge* {
        auto [it, inserted] = gauges.try_emplace(stats.index, nullptr);
        if (inserted) {
          it->second = &family->Add({{"gpu", label}});
        }
        return it->second;
      };

      ensure_gauge(gpu_utilization_gauges_, gpu_utilization_family_)
          ->Set(stats.util_percent);
      ensure_gauge(gpu_memory_used_gauges_, gpu_memory_used_bytes_family_)
          ->Set(stats.mem_used_bytes);
      ensure_gauge(gpu_memory_total_gauges_, gpu_memory_total_bytes_family_)
          ->Set(stats.mem_total_bytes);
      const auto set_or_clear_nan = [&](auto& gauges, auto* family,
                                        double value) {
        if (std::isnan(value)) {
          auto it = gauges.find(stats.index);
          if (it != gauges.end()) {
            if (family != nullptr) {
              family->Remove(it->second);
            }
            gauges.erase(it);
          }
          return;
        }
        ensure_gauge(gauges, family)->Set(value);
      };

      set_or_clear_nan(
          gpu_temperature_gauges_, gpu_temperature_family_,
          stats.temperature_celsius);
      set_or_clear_nan(gpu_power_gauges_, gpu_power_family_, stats.power_watts);
    }

    const auto clear_missing = [&](auto& gauges, auto* family) {
      for (auto it = gauges.begin(); it != gauges.end();) {
        if (!seen_indices.contains(it->first)) {
          if (family != nullptr) {
            family->Remove(it->second);
          }
          it = gauges.erase(it);
        } else {
          ++it;
        }
      }
    };

    clear_missing(gpu_utilization_gauges_, gpu_utilization_family_);
    clear_missing(gpu_memory_used_gauges_, gpu_memory_used_bytes_family_);
    clear_missing(gpu_memory_total_gauges_, gpu_memory_total_bytes_family_);
    clear_missing(gpu_temperature_gauges_, gpu_temperature_family_);
    clear_missing(gpu_power_gauges_, gpu_power_family_);
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
      std::scoped_lock<std::mutex> lock(sampling_mutex_);
      perform_sampling_request_nb();
    }
    for (auto slept = 0ms; slept < next_sleep && !stop.stop_requested();
         slept += 50ms) {
      std::this_thread::sleep_for(50ms);
    }
  }
}

}  // namespace starpu_server

#if defined(STARPU_TESTING)
void
starpu_server::MetricsRegistry::TestAccessor::ClearCpuUsageProvider(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.cpu_usage_provider_ = {};
}

void
starpu_server::MetricsRegistry::TestAccessor::ClearSystemCpuUsageGauge(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.system_cpu_usage_percent_ = nullptr;
}

void
starpu_server::MetricsRegistry::TestAccessor::ClearProcessOpenFdsGauge(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.process_open_fds_ = nullptr;
}

void
starpu_server::MetricsRegistry::TestAccessor::ClearProcessResidentMemoryGauge(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.process_resident_memory_bytes_ = nullptr;
}

void
starpu_server::MetricsRegistry::TestAccessor::ClearInferenceThroughputGauge(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.inference_throughput_gauge_ = nullptr;
}

void
starpu_server::MetricsRegistry::TestAccessor::ClearGpuStatsProvider(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.gpu_stats_provider_ = {};
}

auto
starpu_server::MetricsRegistry::TestAccessor::ProcessOpenFdsGauge(
    starpu_server::MetricsRegistry& metrics) -> prometheus::Gauge*
{
  return metrics.process_open_fds_;
}

auto
starpu_server::MetricsRegistry::TestAccessor::ProcessResidentMemoryGauge(
    starpu_server::MetricsRegistry& metrics) -> prometheus::Gauge*
{
  return metrics.process_resident_memory_bytes_;
}

auto
starpu_server::MetricsRegistry::TestAccessor::InferenceThroughputGauge(
    starpu_server::MetricsRegistry& metrics) -> prometheus::Gauge*
{
  return metrics.inference_throughput_gauge_;
}

auto
starpu_server::MetricsRegistry::TestAccessor::GpuUtilizationGaugeCount(
    const starpu_server::MetricsRegistry& metrics) -> std::size_t
{
  std::scoped_lock<std::mutex> lock(metrics.sampling_mutex_);
  return metrics.gpu_utilization_gauges_.size();
}

auto
starpu_server::MetricsRegistry::TestAccessor::GpuMemoryUsedGaugeCount(
    const starpu_server::MetricsRegistry& metrics) -> std::size_t
{
  std::scoped_lock<std::mutex> lock(metrics.sampling_mutex_);
  return metrics.gpu_memory_used_gauges_.size();
}

auto
starpu_server::MetricsRegistry::TestAccessor::GpuMemoryTotalGaugeCount(
    const starpu_server::MetricsRegistry& metrics) -> std::size_t
{
  std::scoped_lock<std::mutex> lock(metrics.sampling_mutex_);
  return metrics.gpu_memory_total_gauges_.size();
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
  metrics.starpu_worker_inflight_family_ = nullptr;
}

void
starpu_server::MetricsRegistry::TestAccessor::
    ClearStarpuTaskRuntimeByWorkerFamily(
        starpu_server::MetricsRegistry& metrics)
{
  metrics.starpu_task_runtime_by_worker_family_ = nullptr;
}

void
starpu_server::MetricsRegistry::TestAccessor::
    ClearInferenceComputeLatencyByWorkerFamily(
        starpu_server::MetricsRegistry& metrics)
{
  metrics.inference_compute_latency_by_worker_family_ = nullptr;
}

void
starpu_server::MetricsRegistry::TestAccessor::ClearIoCopyLatencyFamily(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.io_copy_latency_family_ = nullptr;
}

void
starpu_server::MetricsRegistry::TestAccessor::ClearTransferBytesFamily(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.transfer_bytes_family_ = nullptr;
}

void
starpu_server::MetricsRegistry::TestAccessor::ClearModelsLoadedFamily(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.models_loaded_family_ = nullptr;
}

void
starpu_server::MetricsRegistry::TestAccessor::ClearModelLoadFailuresFamily(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.model_load_failures_family_ = nullptr;
}

void
starpu_server::MetricsRegistry::TestAccessor::ClearInferenceFailuresFamily(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.inference_failures_family_ = nullptr;
}

void
starpu_server::MetricsRegistry::TestAccessor::ClearInferenceCompletedFamily(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.inference_completed_family_ = nullptr;
}

void
starpu_server::MetricsRegistry::TestAccessor::ClearRequestsByStatusFamily(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.requests_by_status_family_ = nullptr;
}
#endif
