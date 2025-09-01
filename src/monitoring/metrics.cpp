#include "monitoring/metrics.hpp"

#include <prometheus/exposer.h>
#include <prometheus/histogram.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

#include "utils/logger.hpp"

namespace starpu_server {

namespace {
const prometheus::Histogram::BucketBoundaries kInferenceLatencyMsBuckets{
    1, 5, 10, 25, 50, 100, 250, 500, 1000};

struct CpuTotals {
  unsigned long long user{0}, nice{0}, system{0}, idle{0}, iowait{0}, irq{0},
      softirq{0}, steal{0};
};

static bool
read_total_cpu_times(CpuTotals& out)
{
  std::ifstream f{"/proc/stat"};
  if (!f.is_open())
    return false;
  std::string cpu;
  f >> cpu;
  if (cpu.rfind("cpu", 0) != 0)
    return false;
  f >> out.user >> out.nice >> out.system >> out.idle >> out.iowait >>
      out.irq >> out.softirq >> out.steal;
  return true;
}

static double
cpu_usage_percent(const CpuTotals& prev, const CpuTotals& curr)
{
  const unsigned long long prev_idle = prev.idle + prev.iowait;
  const unsigned long long curr_idle = curr.idle + curr.iowait;
  const unsigned long long prev_non_idle = prev.user + prev.nice + prev.system +
                                           prev.irq + prev.softirq + prev.steal;
  const unsigned long long curr_non_idle = curr.user + curr.nice + curr.system +
                                           curr.irq + curr.softirq + curr.steal;

  const unsigned long long prev_total = prev_idle + prev_non_idle;
  const unsigned long long curr_total = curr_idle + curr_non_idle;

  const double totald = static_cast<double>(curr_total - prev_total);
  const double idled = static_cast<double>(curr_idle - prev_idle);
  if (totald <= 0.0)
    return 0.0;
  const double usage = (totald - idled) / totald * 100.0;
  if (usage < 0.0)
    return 0.0;
  if (usage > 100.0)
    return 100.0;
  return usage;
}

struct GpuStat {
  int index{0};
  double util_percent{0.0};
  double mem_used_bytes{0.0};
  double mem_total_bytes{0.0};
};

static std::vector<GpuStat>
query_gpu_stats_nvidia_smi()
{
  std::vector<GpuStat> stats;
  FILE* pipe = popen(
      "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total "
      "--format=csv,noheader,nounits 2>/dev/null",
      "r");
  if (!pipe) {
    return stats;
  }
  char buffer[512];
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    std::string line(buffer);
    // Trim trailing newline
    if (!line.empty() && (line.back() == '\n' || line.back() == '\r'))
      line.pop_back();
    std::istringstream iss(line);
    std::string idx_s, util_s, used_s, total_s;
    if (!std::getline(iss, idx_s, ','))
      continue;
    if (!std::getline(iss, util_s, ','))
      continue;
    if (!std::getline(iss, used_s, ','))
      continue;
    if (!std::getline(iss, total_s, ','))
      continue;

    auto to_int = [](std::string s) {
      size_t start = s.find_first_not_of(" \t");
      size_t end = s.find_last_not_of(" \t");
      if (start == std::string::npos)
        return 0;
      s = s.substr(start, end - start + 1);
      try {
        return std::stoi(s);
      }
      catch (...) {
        return 0;
      }
    };

    const int idx = to_int(idx_s);
    const int util = to_int(util_s);
    const int used_mib = to_int(used_s);
    const int total_mib = to_int(total_s);
    GpuStat st;
    st.index = idx;
    st.util_percent = static_cast<double>(util);
    st.mem_used_bytes = static_cast<double>(used_mib) * 1024.0 * 1024.0;
    st.mem_total_bytes = static_cast<double>(total_mib) * 1024.0 * 1024.0;
    stats.push_back(st);
  }
  pclose(pipe);
  return stats;
}
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

  sampler_thread_ =
      std::jthread([this](std::stop_token st) { this->sampling_loop(st); });
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
MetricsRegistry::sampling_loop(std::stop_token stop)
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
      auto gstats = query_gpu_stats_nvidia_smi();
      for (const auto& st : gstats) {
        const std::string label = std::to_string(st.index);
        if (gpu_utilization_gauges_.find(st.index) ==
            gpu_utilization_gauges_.end()) {
          gpu_utilization_gauges_[st.index] =
              &gpu_utilization_family->Add({{"gpu", label}});
        }
        if (gpu_memory_used_gauges_.find(st.index) ==
            gpu_memory_used_gauges_.end()) {
          gpu_memory_used_gauges_[st.index] =
              &gpu_memory_used_bytes_family->Add({{"gpu", label}});
        }
        if (gpu_memory_total_gauges_.find(st.index) ==
            gpu_memory_total_gauges_.end()) {
          gpu_memory_total_gauges_[st.index] =
              &gpu_memory_total_bytes_family->Add({{"gpu", label}});
        }
        gpu_utilization_gauges_[st.index]->Set(st.util_percent);
        gpu_memory_used_gauges_[st.index]->Set(st.mem_used_bytes);
        gpu_memory_total_gauges_[st.index]->Set(st.mem_total_bytes);
      }
    }
    catch (...) {
    }

    for (auto slept = 0ms; slept < next_sleep && !stop.stop_requested();
         slept += 50ms) {
      std::this_thread::sleep_for(50ms);
    }
  }
}

}  // namespace starpu_server
