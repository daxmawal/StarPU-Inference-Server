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

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
using WorkerCpusetProviderOverrideForTestFn =
    decltype(&starpu_worker_get_hwloc_cpuset);
using HwlocBitmapFirstOverrideForTestFn = decltype(&hwloc_bitmap_first);
using HwlocBitmapNextOverrideForTestFn = decltype(&hwloc_bitmap_next);
using HwlocBitmapFreeOverrideForTestFn = decltype(&hwloc_bitmap_free);

auto
worker_cpuset_provider_override_for_test() noexcept
    -> WorkerCpusetProviderOverrideForTestFn&
{
  struct WorkerCpusetProviderOverrideTag;
  return ::starpu_server::testing::server_main::detail::override_slot_ref<
      WorkerCpusetProviderOverrideTag, WorkerCpusetProviderOverrideForTestFn>();
}

auto
hwloc_bitmap_first_override_for_test() noexcept
    -> HwlocBitmapFirstOverrideForTestFn&
{
  struct HwlocBitmapFirstOverrideTag;
  return ::starpu_server::testing::server_main::detail::override_slot_ref<
      HwlocBitmapFirstOverrideTag, HwlocBitmapFirstOverrideForTestFn>();
}

auto
hwloc_bitmap_next_override_for_test() noexcept
    -> HwlocBitmapNextOverrideForTestFn&
{
  struct HwlocBitmapNextOverrideTag;
  return ::starpu_server::testing::server_main::detail::override_slot_ref<
      HwlocBitmapNextOverrideTag, HwlocBitmapNextOverrideForTestFn>();
}

auto
hwloc_bitmap_free_override_for_test() noexcept
    -> HwlocBitmapFreeOverrideForTestFn&
{
  struct HwlocBitmapFreeOverrideTag;
  return ::starpu_server::testing::server_main::detail::override_slot_ref<
      HwlocBitmapFreeOverrideTag, HwlocBitmapFreeOverrideForTestFn>();
}
#endif  // SONAR_IGNORE_STOP

auto
get_worker_cpuset_for_affinity(int worker_id) -> hwloc_cpuset_t
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto override_fn = worker_cpuset_provider_override_for_test();
      override_fn != nullptr) {
    return override_fn(worker_id);
  }
#endif  // SONAR_IGNORE_STOP
  return starpu_worker_get_hwloc_cpuset(worker_id);
}

auto
bitmap_first_for_affinity(hwloc_const_bitmap_t cpuset) -> int
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto override_fn = hwloc_bitmap_first_override_for_test();
      override_fn != nullptr) {
    return override_fn(cpuset);
  }
#endif  // SONAR_IGNORE_STOP
  return hwloc_bitmap_first(cpuset);
}

auto
bitmap_next_for_affinity(hwloc_const_bitmap_t cpuset, int previous_core) -> int
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto override_fn = hwloc_bitmap_next_override_for_test();
      override_fn != nullptr) {
    return override_fn(cpuset, previous_core);
  }
#endif  // SONAR_IGNORE_STOP
  return hwloc_bitmap_next(cpuset, previous_core);
}

void
bitmap_free_for_affinity(hwloc_bitmap_t cpuset)
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto override_fn = hwloc_bitmap_free_override_for_test();
      override_fn != nullptr) {
    override_fn(cpuset);
    return;
  }
#endif  // SONAR_IGNORE_STOP
  hwloc_bitmap_free(cpuset);
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
  hwloc_cpuset_t cpuset = get_worker_cpuset_for_affinity(worker_id);
  if (cpuset == nullptr) {
    return {};
  }

  std::vector<int> cores;
  for (int core = bitmap_first_for_affinity(cpuset); core != -1;
       core = bitmap_next_for_affinity(cpuset, core)) {
    cores.push_back(core);
  }
  bitmap_free_for_affinity(cpuset);
  return format_cpu_core_ranges(cores);
}

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
using WorkerCountOverrideForTestFn = decltype(&starpu_worker_get_count);
using WorkerTypeOverrideForTestFn = decltype(&starpu_worker_get_type);
using WorkerDeviceIdOverrideForTestFn = decltype(&starpu_worker_get_devid);
using DescribeCpuAffinityOverrideForTestFn = std::string (*)(int);

auto
worker_count_override_for_test() noexcept -> WorkerCountOverrideForTestFn&
{
  struct WorkerCountOverrideTag;
  return ::starpu_server::testing::server_main::detail::override_slot_ref<
      WorkerCountOverrideTag, WorkerCountOverrideForTestFn>();
}

auto
worker_type_override_for_test() noexcept -> WorkerTypeOverrideForTestFn&
{
  struct WorkerTypeOverrideTag;
  return ::starpu_server::testing::server_main::detail::override_slot_ref<
      WorkerTypeOverrideTag, WorkerTypeOverrideForTestFn>();
}

auto
worker_device_id_override_for_test() noexcept
    -> WorkerDeviceIdOverrideForTestFn&
{
  struct WorkerDeviceIdOverrideTag;
  return ::starpu_server::testing::server_main::detail::override_slot_ref<
      WorkerDeviceIdOverrideTag, WorkerDeviceIdOverrideForTestFn>();
}

auto
describe_cpu_affinity_override_for_test() noexcept
    -> DescribeCpuAffinityOverrideForTestFn&
{
  struct DescribeCpuAffinityOverrideTag;
  return ::starpu_server::testing::server_main::detail::override_slot_ref<
      DescribeCpuAffinityOverrideTag, DescribeCpuAffinityOverrideForTestFn>();
}
#endif  // SONAR_IGNORE_STOP

auto
worker_count_for_inventory() -> int
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto override_fn = worker_count_override_for_test();
      override_fn != nullptr) {
    return static_cast<int>(override_fn());
  }
#endif  // SONAR_IGNORE_STOP
  return static_cast<int>(starpu_worker_get_count());
}

// clang-format off
auto
worker_type_for_inventory(int worker_id) -> enum starpu_worker_archtype
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  const auto override_fn = worker_type_override_for_test();
  if (override_fn != nullptr) {
    const auto worker_type = override_fn(worker_id);
    return worker_type;
  }
#endif  // SONAR_IGNORE_STOP
  return starpu_worker_get_type(worker_id);
}
// clang-format on

auto
worker_device_id_for_inventory(int worker_id) -> int
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto override_fn = worker_device_id_override_for_test();
      override_fn != nullptr) {
    return override_fn(worker_id);
  }
#endif  // SONAR_IGNORE_STOP
  return starpu_worker_get_devid(worker_id);
}

auto
describe_cpu_affinity_for_inventory(int worker_id) -> std::string
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  if (const auto override_fn = describe_cpu_affinity_override_for_test();
      override_fn != nullptr) {
    return override_fn(worker_id);
  }
#endif  // SONAR_IGNORE_STOP
  return describe_cpu_affinity(worker_id);
}

void
log_worker_inventory(const starpu_server::RuntimeConfig& opts)
{
  const auto total_workers = worker_count_for_inventory();
  starpu_server::log_info(
      opts.verbosity,
      std::format("Configured {} StarPU worker(s).", total_workers));

  for (int worker_id = 0; worker_id < total_workers; ++worker_id) {
    const auto type = worker_type_for_inventory(worker_id);
    const int device_id = worker_device_id_for_inventory(worker_id);
    const std::string device_label =
        device_id >= 0 ? std::to_string(device_id) : "N/A";
    std::string cpu_affinity;
    if (type == STARPU_CPU_WORKER) {
      const std::string affinity =
          describe_cpu_affinity_for_inventory(worker_id);
      if (!affinity.empty()) {
        cpu_affinity = std::format(", cores={}", affinity);
      }
    }
    starpu_server::log_info(
        opts.verbosity,
        std::format(
            "Worker {:2d}: type={}, device id={}{}", worker_id,
            worker_type_label(type), device_label, cpu_affinity));
  }
}
