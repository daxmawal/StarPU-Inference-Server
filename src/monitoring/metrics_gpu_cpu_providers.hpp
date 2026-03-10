#pragma once

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "monitoring/metrics.hpp"

#ifdef STARPU_HAVE_NVML
#include <nvml.h>
#endif

#include "metrics_constants.hpp"
#include "metrics_labels.hpp"
#include "utils/logger.hpp"

namespace starpu_server::inline metrics_internal_detail {

using monitoring::detail::CpuTotals;

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
}  // namespace starpu_server::inline metrics_internal_detail

namespace starpu_server::monitoring::detail {

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

}  // namespace starpu_server::monitoring::detail

namespace starpu_server {

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

}  // namespace starpu_server
