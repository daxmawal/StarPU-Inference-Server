#include "monitoring/metrics.hpp"

#include <prometheus/exposer.h>
#include <prometheus/histogram.h>
#include <unistd.h>

#include <chrono>
#include <exception>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef STARPU_HAVE_NVML
#include <nvml.h>
#endif

#include "utils/logger.hpp"

namespace starpu_server {

namespace {
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

struct CpuTotals {
  unsigned long long user{0}, nice{0}, system{0}, idle{0}, iowait{0}, irq{0},
      softirq{0}, steal{0};
};

auto
read_total_cpu_times(CpuTotals& out) -> bool
{
  std::ifstream function{"/proc/stat"};
  if (!function.is_open()) {
    return false;
  }
  std::string cpu;
  function >> cpu;
  if (!cpu.starts_with("cpu")) {
    return false;
  }
  function >> out.user >> out.nice >> out.system >> out.idle >> out.iowait >>
      out.irq >> out.softirq >> out.steal;
  return true;
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

struct GpuStat {
  int index{0};
  double util_percent{0.0};
  double mem_used_bytes{0.0};
  double mem_total_bytes{0.0};
};

#ifdef STARPU_HAVE_NVML

class NvmlWrapper {
 public:
  static auto instance() -> NvmlWrapper&;

  auto query_stats() -> std::vector<GpuStat>;

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
NvmlWrapper::query_stats() -> std::vector<GpuStat>
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

  std::vector<GpuStat> stats;
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

    GpuStat stat;
    stat.index = static_cast<int>(idx);
    stat.util_percent = static_cast<double>(utilization.gpu);
    stat.mem_used_bytes = static_cast<double>(memory_info.used);
    stat.mem_total_bytes = static_cast<double>(memory_info.total);
    stats.push_back(stat);
  }

  return stats;
}

auto
query_gpu_stats_nvml() -> std::vector<GpuStat>
{
  return NvmlWrapper::instance().query_stats();
}

#else

auto
query_gpu_stats_nvml() -> std::vector<GpuStat>
{
  return {};
}

#endif  // STARPU_HAVE_NVML
}  // namespace

MetricsRegistry::MetricsRegistry(int port)
    : registry(std::make_shared<prometheus::Registry>()),
      requests_total(nullptr), inference_latency(nullptr),
      queue_size_gauge(nullptr), exposer_(nullptr)
{
  try {
    exposer_ = std::make_unique<prometheus::Exposer>(
        "0.0.0.0:" + std::to_string(port));
    exposer_->RegisterCollectable(registry);
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

  sampler_thread_ = std::jthread(
      [this](const std::stop_token& stop) { this->sampling_loop(stop); });
}

MetricsRegistry::~MetricsRegistry() noexcept
{
  if (sampler_thread_.joinable()) {
    sampler_thread_.request_stop();
  }
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
MetricsRegistry::sampling_loop(const std::stop_token& stop)
{
  using namespace std::chrono_literals;

  CpuTotals prev_cpu{};
  bool have_prev_cpu = read_total_cpu_times(prev_cpu);

  auto next_sleep = 1000ms;
  while (!stop.stop_requested()) {
    CpuTotals cur_cpu{};
    if (read_total_cpu_times(cur_cpu) && have_prev_cpu &&
        system_cpu_usage_percent != nullptr) {
      const double usage = cpu_usage_percent(prev_cpu, cur_cpu);
      system_cpu_usage_percent->Set(usage);
    }
    prev_cpu = cur_cpu;
    have_prev_cpu = true;

    try {
      auto gstats = query_gpu_stats_nvml();
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
      log_error(
          std::string("GPU metrics sampling failed: ") + e.what());
    }
    catch (...) {
      log_error("GPU metrics sampling failed due to an unknown error");
    }

    for (auto slept = 0ms; slept < next_sleep && !stop.stop_requested();
         slept += 50ms) {
      std::this_thread::sleep_for(50ms);
    }
  }
}

}  // namespace starpu_server
