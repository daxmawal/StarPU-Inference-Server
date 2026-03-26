auto
worker_type_label(starpu_worker_archtype type) -> std::string
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
format_int_list(const std::vector<int>& values) -> std::string
{
  if (values.empty()) {
    return "none";
  }

  std::string result;
  for (std::size_t idx = 0; idx < values.size(); ++idx) {
    if (idx > 0) {
      result += ',';
    }
    result += std::to_string(values[idx]);
  }
  return result;
}

auto
resolve_model_label_for_startup_metrics(
    const starpu_server::RuntimeConfig& opts) -> std::string
{
  if (!opts.model.has_value()) {
    return "default";
  }
  if (!opts.model->name.empty()) {
    return opts.model->name;
  }
  return opts.model->path;
}

// Test-only runtime hook overrides kept together for readability.
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
STARPU_SERVER_DECLARE_TEST_OVERRIDE_SLOT(
    WorkerCpusetProviderOverrideForTestFn,
    worker_cpuset_provider_override_for_test,
    decltype(&starpu_worker_get_hwloc_cpuset))
STARPU_SERVER_DECLARE_TEST_OVERRIDE_SLOT(
    HwlocBitmapFirstOverrideForTestFn, hwloc_bitmap_first_override_for_test,
    decltype(&hwloc_bitmap_first))
STARPU_SERVER_DECLARE_TEST_OVERRIDE_SLOT(
    HwlocBitmapNextOverrideForTestFn, hwloc_bitmap_next_override_for_test,
    decltype(&hwloc_bitmap_next))
STARPU_SERVER_DECLARE_TEST_OVERRIDE_SLOT(
    HwlocBitmapFreeOverrideForTestFn, hwloc_bitmap_free_override_for_test,
    decltype(&hwloc_bitmap_free))
STARPU_SERVER_DECLARE_TEST_OVERRIDE_SLOT(
    WorkerCountOverrideForTestFn, worker_count_override_for_test,
    decltype(&starpu_worker_get_count))
STARPU_SERVER_DECLARE_TEST_OVERRIDE_SLOT(
    WorkerTypeOverrideForTestFn, worker_type_override_for_test,
    decltype(&starpu_worker_get_type))
STARPU_SERVER_DECLARE_TEST_OVERRIDE_SLOT(
    WorkerDeviceIdOverrideForTestFn, worker_device_id_override_for_test,
    decltype(&starpu_worker_get_devid))
STARPU_SERVER_DECLARE_TEST_OVERRIDE_SLOT(
    DescribeCpuAffinityOverrideForTestFn,
    describe_cpu_affinity_override_for_test, std::string (*)(int))
#endif  // SONAR_IGNORE_STOP

auto
get_worker_cpuset_for_affinity(int worker_id) -> hwloc_cpuset_t
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  return ::starpu_server::testing::server_main::detail::call_override_or(
      worker_cpuset_provider_override_for_test,
      [](int id) { return starpu_worker_get_hwloc_cpuset(id); }, worker_id);
#else
  return starpu_worker_get_hwloc_cpuset(worker_id);
#endif  // SONAR_IGNORE_STOP
}

auto
bitmap_first_for_affinity(hwloc_const_bitmap_t cpuset) -> int
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  return ::starpu_server::testing::server_main::detail::call_override_or(
      hwloc_bitmap_first_override_for_test,
      [](hwloc_const_bitmap_t bitmap) { return hwloc_bitmap_first(bitmap); },
      cpuset);
#else
  return hwloc_bitmap_first(cpuset);
#endif  // SONAR_IGNORE_STOP
}

auto
bitmap_next_for_affinity(hwloc_const_bitmap_t cpuset, int previous_core) -> int
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  return ::starpu_server::testing::server_main::detail::call_override_or(
      hwloc_bitmap_next_override_for_test,
      [](hwloc_const_bitmap_t bitmap, int previous) {
        return hwloc_bitmap_next(bitmap, previous);
      },
      cpuset, previous_core);
#else
  return hwloc_bitmap_next(cpuset, previous_core);
#endif  // SONAR_IGNORE_STOP
}

void
bitmap_free_for_affinity(hwloc_bitmap_t cpuset)
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  ::starpu_server::testing::server_main::detail::call_override_or(
      hwloc_bitmap_free_override_for_test,
      [](hwloc_bitmap_t bitmap) { hwloc_bitmap_free(bitmap); }, cpuset);
#else
  hwloc_bitmap_free(cpuset);
#endif  // SONAR_IGNORE_STOP
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

auto
worker_count_for_inventory() -> int
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  return ::starpu_server::testing::server_main::detail::call_override_or(
      worker_count_override_for_test,
      []() { return static_cast<int>(starpu_worker_get_count()); });
#else
  return static_cast<int>(starpu_worker_get_count());
#endif  // SONAR_IGNORE_STOP
}

auto
worker_type_for_inventory(int worker_id) -> starpu_worker_archtype
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  return ::starpu_server::testing::server_main::detail::call_override_or(
      worker_type_override_for_test,
      [](int id) { return starpu_worker_get_type(id); }, worker_id);
#else
  return starpu_worker_get_type(worker_id);
#endif  // SONAR_IGNORE_STOP
}

auto
worker_device_id_for_inventory(int worker_id) -> int
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  return ::starpu_server::testing::server_main::detail::call_override_or(
      worker_device_id_override_for_test,
      [](int id) { return starpu_worker_get_devid(id); }, worker_id);
#else
  return starpu_worker_get_devid(worker_id);
#endif  // SONAR_IGNORE_STOP
}

auto
describe_cpu_affinity_for_inventory(int worker_id) -> std::string
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  return ::starpu_server::testing::server_main::detail::call_override_or(
      describe_cpu_affinity_override_for_test,
      [](int id) { return describe_cpu_affinity(id); }, worker_id);
#else
  return describe_cpu_affinity(worker_id);
#endif  // SONAR_IGNORE_STOP
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

void
report_gpu_replication_startup(
    const starpu_server::RuntimeConfig& opts, std::size_t loaded_gpu_replicas)
{
  if (!opts.devices.use_cuda || opts.devices.ids.empty()) {
    return;
  }

  const std::string model_label = resolve_model_label_for_startup_metrics(opts);
  const std::string replication_policy =
      std::string(to_string(opts.devices.gpu_model_replication));

  starpu_server::set_gpu_model_replication_policy(
      model_label, replication_policy);
  starpu_server::set_gpu_model_replicas_total(model_label, loaded_gpu_replicas);

  try {
    const auto assignments =
        starpu_server::detail::build_gpu_replica_assignments(opts);
    const auto workers_by_device =
        starpu_server::StarPUSetup::get_cuda_workers_by_device(
            opts.devices.ids);

    std::map<int, std::size_t> replicas_by_device;
    const std::size_t replica_limit = loaded_gpu_replicas < assignments.size()
                                          ? loaded_gpu_replicas
                                          : assignments.size();
    for (std::size_t idx = 0; idx < replica_limit; ++idx) {
      ++replicas_by_device[assignments[idx].device_id];
    }

    starpu_server::log_info(
        opts.verbosity,
        std::format(
            "GPU model replication summary: policy={}, total_replicas={}, "
            "configured_cuda_devices={}.",
            replication_policy, loaded_gpu_replicas, workers_by_device.size()));

    for (const int device_id : opts.devices.ids) {
      if (device_id < 0) {
        continue;
      }

      const auto workers_it = workers_by_device.find(device_id);
      const std::vector<int> workers = workers_it != workers_by_device.end()
                                           ? workers_it->second
                                           : std::vector<int>{};
      const std::size_t replica_count = replicas_by_device.contains(device_id)
                                            ? replicas_by_device[device_id]
                                            : 0U;
      for (const int worker_id : workers) {
        starpu_server::set_starpu_cuda_worker_info(worker_id, device_id, true);
      }

      starpu_server::log_info(
          opts.verbosity,
          std::format(
              "CUDA device {} -> workers=[{}], model replicas={}", device_id,
              format_int_list(workers), replica_count));
    }
  }
  catch (const std::exception& error) {
    starpu_server::log_warning(std::format(
        "Failed to summarize GPU model replication startup state: {}",
        error.what()));
  }
}
