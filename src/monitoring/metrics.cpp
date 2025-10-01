#include "monitoring/metrics.hpp"

#include <prometheus/exposer.h>
#include <prometheus/histogram.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef STARPU_HAVE_NVML
#include <nvml.h>
#endif

#include "utils/logger.hpp"

namespace starpu_server {

namespace monitoring::detail {

auto
read_total_cpu_times(std::istream& input, CpuTotals& out) -> bool
{
  std::string cpu{};
  if (!(input >> cpu)) {
    return false;
  }
  if (!cpu.starts_with("cpu")) {
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

const prometheus::Histogram::BucketBoundaries kInferenceLatencyMsBuckets{
    1, 5, 10, 25, 50, 100, 250, 500, 1000};

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

  const auto totald = static_cast<double>(curr_total - prev_total);
  const auto idled = static_cast<double>(curr_idle - prev_idle);
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

using GpuSample = MetricsRegistry::GpuSample;
using GpuStatsProvider = MetricsRegistry::GpuStatsProvider;
using CpuUsageProvider = MetricsRegistry::CpuUsageProvider;

#ifdef STARPU_HAVE_NVML

class NvmlWrapper {
 public:
  static auto instance() -> NvmlWrapper&;

  auto query_stats() -> std::vector<GpuSample>;

 private:
  NvmlWrapper();
  ~NvmlWrapper();

  NvmlWrapper(const NvmlWrapper&) = delete;
  NvmlWrapper& operator=(const NvmlWrapper&) = delete;

  static auto error_string(nvmlReturn_t rc) -> const char*;

  bool initialized_{false};
  std::mutex mutex_{};
};

auto
NvmlWrapper::instance() -> NvmlWrapper&
{
  static NvmlWrapper wrapper;
  return wrapper;
}

NvmlWrapper::NvmlWrapper()
{
  const nvmlReturn_t status = nvmlInit();
  if (status != NVML_SUCCESS) {
    log_warning(
        std::string("Failed to initialize NVML: ") + error_string(status));
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
  std::lock_guard<std::mutex> guard(mutex_);
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
      log_warning(
          std::string("nvmlDeviceGetHandleByIndex failed for GPU ") +
          std::to_string(idx) + ": " + error_string(status));
      continue;
    }

    nvmlUtilization_t utilization{};
    status = nvmlDeviceGetUtilizationRates(device, &utilization);
    if (status != NVML_SUCCESS) {
      log_warning(
          std::string("nvmlDeviceGetUtilizationRates failed for GPU ") +
          std::to_string(idx) + ": " + error_string(status));
      continue;
    }

    nvmlMemory_t memory_info{};
    status = nvmlDeviceGetMemoryInfo(device, &memory_info);
    if (status != NVML_SUCCESS) {
      log_warning(
          std::string("nvmlDeviceGetMemoryInfo failed for GPU ") +
          std::to_string(idx) + ": " + error_string(status));
      continue;
    }

    GpuSample stat;
    stat.index = static_cast<int>(idx);
    stat.util_percent = static_cast<double>(utilization.gpu);
    stat.mem_used_bytes = static_cast<double>(memory_info.used);
    stat.mem_total_bytes = static_cast<double>(memory_info.total);
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
make_cpu_usage_provider(std::function<bool(CpuTotals&)> reader)
    -> MetricsRegistry::CpuUsageProvider
{
  CpuTotals prev_cpu{};
  bool have_prev_cpu = reader(prev_cpu);
  return [reader = std::move(reader), prev_cpu,
          have_prev_cpu]() mutable -> std::optional<double> {
    CpuTotals cur_cpu{};
    if (!reader(cur_cpu)) {
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
    : registry(std::make_shared<prometheus::Registry>()),
      requests_total(nullptr), inference_latency(nullptr),
      queue_size_gauge(nullptr), exposer_(nullptr),
      gpu_stats_provider_(std::move(gpu_provider)),
      cpu_usage_provider_(std::move(cpu_provider))
{
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
      auto exposer = std::make_unique<prometheus::Exposer>(
          "0.0.0.0:" + std::to_string(port));
      exposer_handle =
          std::make_unique<PrometheusExposerHandle>(std::move(exposer));
    }
    exposer_handle->RegisterCollectable(registry);
    exposer_ = std::move(exposer_handle);
  }
  catch (const std::exception& e) {
    log_error(std::string("Failed to initialize metrics exposer: ") + e.what());
    throw;
  }

  auto& counter_family = prometheus::BuildCounter()
                             .Name("requests_total")
                             .Help("Total requests received")
                             .Register(*registry);
  requests_total = &counter_family.Add({});

  auto& histogram_family = prometheus::BuildHistogram()
                               .Name("inference_latency_ms")
                               .Help("Inference latency in milliseconds")
                               .Register(*registry);
  inference_latency = &histogram_family.Add({}, kInferenceLatencyMsBuckets);

  auto& gauge_family = prometheus::BuildGauge()
                           .Name("inference_queue_size")
                           .Help("Number of jobs in the inference queue")
                           .Register(*registry);
  queue_size_gauge = &gauge_family.Add({});

  auto& cpu_family = prometheus::BuildGauge()
                         .Name("system_cpu_usage_percent")
                         .Help("System-wide CPU utilization percentage (0-100)")
                         .Register(*registry);
  system_cpu_usage_percent = &cpu_family.Add({});

  gpu_utilization_family =
      &prometheus::BuildGauge()
           .Name("gpu_utilization_percent")
           .Help("GPU utilization percentage per GPU (0-100)")
           .Register(*registry);
  gpu_memory_used_bytes_family = &prometheus::BuildGauge()
                                      .Name("gpu_memory_used_bytes")
                                      .Help("Used GPU memory in bytes per GPU")
                                      .Register(*registry);
  gpu_memory_total_bytes_family =
      &prometheus::BuildGauge()
           .Name("gpu_memory_total_bytes")
           .Help("Total GPU memory in bytes per GPU")
           .Register(*registry);

  if (start_sampler_thread) {
    sampler_thread_ = std::jthread(
        [this](const std::stop_token& stop) { this->sampling_loop(stop); });
  }
}

MetricsRegistry::~MetricsRegistry() noexcept
{
  request_stop();
  if (exposer_ && registry) {
    try {
      exposer_->RemoveCollectable(registry);
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
}  // namespace

auto
init_metrics(int port) -> bool
{
  std::shared_ptr<MetricsRegistry> expected{nullptr};

  try {
    auto new_metrics = std::make_shared<MetricsRegistry>(port);

#ifndef STARPU_HAVE_NVML
    std::call_once(nvml_warning_flag(), [] {
      log_warning_critical(
          "NVML support is not available; GPU metrics collection is "
          "disabled.");
    });
#endif

    if (!metrics_atomic().compare_exchange_strong(
            expected, new_metrics, std::memory_order_acq_rel,
            std::memory_order_acquire)) {
      log_warning("Metrics were previously initialized");
      return false;
    }

    set_queue_size(0);
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
  if (metrics_ptr && metrics_ptr->queue_size_gauge != nullptr) {
    metrics_ptr->queue_size_gauge->Set(static_cast<double>(size));
  }
}

void
MetricsRegistry::request_stop()
{
  if (sampler_thread_.joinable()) {
    sampler_thread_.request_stop();
  }
}

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

void
MetricsRegistry::run_sampling_iteration()
{
  perform_sampling_iteration();
}

void
MetricsRegistry::perform_sampling_iteration()
{
  if (system_cpu_usage_percent != nullptr && cpu_usage_provider_) {
    try {
      auto usage = cpu_usage_provider_();
      if (usage.has_value()) {
        system_cpu_usage_percent->Set(*usage);
      }
    }
    catch (const std::exception& e) {
      log_error(std::string("CPU metrics sampling failed: ") + e.what());
    }
    catch (...) {
      log_error("CPU metrics sampling failed due to an unknown error");
    }
  }

  if (!gpu_stats_provider_) {
    return;
  }

  try {
    auto gstats = gpu_stats_provider_();
    for (const auto& stats : gstats) {
      const std::string label = std::to_string(stats.index);
      if (gpu_utilization_gauges_.find(stats.index) ==
          gpu_utilization_gauges_.end()) {
        gpu_utilization_gauges_[stats.index] =
            &gpu_utilization_family->Add({{"gpu", label}});
      }
      if (gpu_memory_used_gauges_.find(stats.index) ==
          gpu_memory_used_gauges_.end()) {
        gpu_memory_used_gauges_[stats.index] =
            &gpu_memory_used_bytes_family->Add({{"gpu", label}});
      }
      if (gpu_memory_total_gauges_.find(stats.index) ==
          gpu_memory_total_gauges_.end()) {
        gpu_memory_total_gauges_[stats.index] =
            &gpu_memory_total_bytes_family->Add({{"gpu", label}});
      }
      gpu_utilization_gauges_[stats.index]->Set(stats.util_percent);
      gpu_memory_used_gauges_[stats.index]->Set(stats.mem_used_bytes);
      gpu_memory_total_gauges_[stats.index]->Set(stats.mem_total_bytes);
    }
  }
  catch (const std::exception& e) {
    log_error(std::string("GPU metrics sampling failed: ") + e.what());
  }
  catch (...) {
    log_error("GPU metrics sampling failed due to an unknown error");
  }
}

void
MetricsRegistry::sampling_loop(const std::stop_token& stop)
{
  using namespace std::chrono_literals;
  auto next_sleep = 1000ms;
  while (!stop.stop_requested()) {
    perform_sampling_iteration();
    for (auto slept = 0ms; slept < next_sleep && !stop.stop_requested();
         slept += 50ms) {
      std::this_thread::sleep_for(50ms);
    }
  }
}

}  // namespace starpu_server
