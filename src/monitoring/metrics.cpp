#include "monitoring/metrics.hpp"

#include <prometheus/exposer.h>
#include <prometheus/histogram.h>

#include <atomic>
#include <chrono>
#include <exception>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
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
const prometheus::Histogram::BucketBoundaries kBatchSizeBuckets{
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};

auto
status_code_label(int code) -> std::string
{
  // Common gRPC status codes; fall back to numeric.
  switch (code) {
    case 0:
      return "OK";
    case 3:
      return "INVALID_ARGUMENT";
    case 4:
      return "DEADLINE_EXCEEDED";
    case 5:
      return "NOT_FOUND";
    case 7:
      return "PERMISSION_DENIED";
    case 8:
      return "RESOURCE_EXHAUSTED";
    case 9:
      return "FAILED_PRECONDITION";
    case 10:
      return "ABORTED";
    case 11:
      return "OUT_OF_RANGE";
    case 12:
      return "UNIMPLEMENTED";
    case 13:
      return "INTERNAL";
    case 14:
      return "UNAVAILABLE";
    case 16:
      return "UNAUTHENTICATED";
    default:
      return std::to_string(code);
  }
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
          std::format("0.0.0.0:{}", port));
      exposer_handle =
          std::make_unique<PrometheusExposerHandle>(std::move(exposer));
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

  auto& cpu_family = prometheus::BuildGauge()
                         .Name("system_cpu_usage_percent")
                         .Help("System-wide CPU utilization percentage (0-100)")
                         .Register(*registry_);
  system_cpu_usage_percent_ = &cpu_family.Add({});

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

  if (start_sampler_thread) {
    sampler_thread_ = std::jthread(
        [this](const std::stop_token& stop) { this->sampling_loop(stop); });
  }
}

MetricsRegistry::~MetricsRegistry() noexcept
{
  request_stop();
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
      const double ratio = static_cast<double>(size) / capacity;
      metrics_ptr->queue_fill_ratio_gauge()->Set(ratio);
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
  if (metrics_ptr->queue_fill_ratio_gauge() != nullptr) {
    metrics_ptr->queue_fill_ratio_gauge()->Set(0.0);
  }
}

void
set_queue_fill_ratio(std::size_t size, std::size_t capacity)
{
  auto metrics_ptr = metrics_atomic().load(std::memory_order_acquire);
  if (metrics_ptr == nullptr ||
      metrics_ptr->queue_fill_ratio_gauge() == nullptr || capacity == 0) {
    return;
  }
  metrics_ptr->queue_fill_ratio_gauge()->Set(
      static_cast<double>(size) / capacity);
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
      status_code_label(status_code), model_name);
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
MetricsRegistry::system_cpu_usage_percent() const -> prometheus::Gauge*
{
  return system_cpu_usage_percent_;
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
MetricsRegistry::requests_by_status_family() const
    -> prometheus::Family<prometheus::Counter>*
{
  return requests_by_status_family_;
}

void
MetricsRegistry::increment_status_counter(
    std::string_view code_label, std::string_view model_label)
{
  if (requests_by_status_family_ == nullptr) {
    return;
  }

  const std::string code{code_label};
  const std::string model{model_label};
  const std::string key = std::format("{}|{}", code, model);

  std::lock_guard<std::mutex> lock(status_mutex_);
  auto [it, inserted] = status_counters_.try_emplace(key, nullptr);
  if (inserted) {
    it->second =
        &requests_by_status_family_->Add({{"code", code}, {"model", model}});
  }
  it->second->Increment();
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

void
MetricsRegistry::run_sampling_request_nb()
{
  perform_sampling_request_nb();
}

void
MetricsRegistry::perform_sampling_request_nb()
{
  if (system_cpu_usage_percent_ != nullptr && cpu_usage_provider_) {
    try {
      auto usage = cpu_usage_provider_();
      if (usage.has_value()) {
        system_cpu_usage_percent_->Set(*usage);
      }
    }
    catch (const std::exception& e) {
      log_error(std::format("CPU metrics sampling failed: {}", e.what()));
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
    }
  }
  catch (const std::exception& e) {
    log_error(std::format("GPU metrics sampling failed: {}", e.what()));
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
    perform_sampling_request_nb();
    for (auto slept = 0ms; slept < next_sleep && !stop.stop_requested();
         slept += 50ms) {
      std::this_thread::sleep_for(50ms);
    }
  }
}

}  // namespace starpu_server
