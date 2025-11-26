#include "worker_info.hpp"

#include <hwloc.h>
#include <starpu.h>

#include <format>
#include <string>
#include <vector>

#include "utils/logger.hpp"
#include "utils/runtime_config.hpp"

namespace starpu_server {

namespace {

auto
worker_type_label(const enum starpu_worker_archtype type) -> std::string
{
  switch (type) {
    case STARPU_CPU_WORKER:
      return "CPU";
    case STARPU_CUDA_WORKER:
      return "CUDA";
    default:
      return std::format("Other({})", static_cast<int>(type));
  }
}

auto
format_cpu_core_ranges(const std::vector<int>& cpus) -> std::string
{
  if (cpus.empty()) {
    return {};
  }

  std::string result;
  auto flush_range = [&](int start, int end) {
    if (!result.empty()) {
      result.push_back(',');
    }
    if (start == end) {
      result += std::to_string(start);
    } else {
      result += std::format("{}-{}", start, end);
    }
  };

  int range_start = cpus.front();
  int previous = range_start;
  for (std::size_t idx = 1; idx < cpus.size(); ++idx) {
    const int core = cpus[idx];
    if (core == previous + 1) {
      previous = core;
    } else {
      flush_range(range_start, previous);
      range_start = previous = core;
    }
  }
  flush_range(range_start, previous);
  return result;
}

auto
describe_cpu_affinity(int worker_id) -> std::string
{
  hwloc_cpuset_t cpuset = starpu_worker_get_hwloc_cpuset(worker_id);
  if (cpuset == nullptr) {
    return {};
  }

  std::vector<int> cores;
  for (int core = hwloc_bitmap_first(cpuset); core != -1;
       core = hwloc_bitmap_next(cpuset, core)) {
    cores.push_back(core);
  }
  hwloc_bitmap_free(cpuset);
  return format_cpu_core_ranges(cores);
}

}  // namespace

void
log_worker_inventory(const RuntimeConfig& opts)
{
  const auto total_workers = static_cast<int>(starpu_worker_get_count());
  log_info(
      opts.verbosity,
      std::format("Configured {} StarPU worker(s).", total_workers));

  for (int worker_id = 0; worker_id < total_workers; ++worker_id) {
    const auto type = starpu_worker_get_type(worker_id);
    const int device_id = starpu_worker_get_devid(worker_id);
    const std::string device_label =
        device_id >= 0 ? std::to_string(device_id) : "N/A";
    std::string cpu_affinity;
    if (type == STARPU_CPU_WORKER) {
      const std::string affinity = describe_cpu_affinity(worker_id);
      if (!affinity.empty()) {
        cpu_affinity = std::format(", cores={}", affinity);
      }
    }
    log_info(
        opts.verbosity,
        std::format(
            "Worker {:2d}: type={}, device id={}{}", worker_id,
            worker_type_label(type), device_label, cpu_affinity));
  }
}

}  // namespace starpu_server
