#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
void
starpu_server::testing::MetricsRegistryTestAccessor::ClearCpuUsageProvider(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.providers_.cpu_usage_provider = {};
}

void
starpu_server::testing::MetricsRegistryTestAccessor::ClearQueueSizeGauge(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.gauges_.queue_size = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::ClearQueueFillRatioGauge(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.gauges_.queue_fill_ratio = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::ClearSystemCpuUsageGauge(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.gauges_.system_cpu_usage_percent = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::ClearProcessOpenFdsGauge(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.gauges_.process_open_fds = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    ClearProcessResidentMemoryGauge(starpu_server::MetricsRegistry& metrics)
{
  metrics.gauges_.process_resident_memory_bytes = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    ClearInferenceThroughputGauge(starpu_server::MetricsRegistry& metrics)
{
  metrics.gauges_.inference_throughput = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::ClearGpuStatsProvider(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.providers_.gpu_stats_provider = {};
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::ProcessOpenFdsGauge(
    starpu_server::MetricsRegistry& metrics) -> prometheus::Gauge*
{
  return metrics.gauges_.process_open_fds;
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::ProcessResidentMemoryGauge(
    starpu_server::MetricsRegistry& metrics) -> prometheus::Gauge*
{
  return metrics.gauges_.process_resident_memory_bytes;
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::InferenceThroughputGauge(
    starpu_server::MetricsRegistry& metrics) -> prometheus::Gauge*
{
  return metrics.gauges_.inference_throughput;
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::GpuUtilizationGaugeCount(
    const starpu_server::MetricsRegistry& metrics) -> std::size_t
{
  std::scoped_lock<std::mutex> lock(metrics.mutexes_.sampling);
  return metrics.caches_.gpu.utilization.size();
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::GpuMemoryUsedGaugeCount(
    const starpu_server::MetricsRegistry& metrics) -> std::size_t
{
  std::scoped_lock<std::mutex> lock(metrics.mutexes_.sampling);
  return metrics.caches_.gpu.memory_used.size();
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::GpuMemoryTotalGaugeCount(
    const starpu_server::MetricsRegistry& metrics) -> std::size_t
{
  std::scoped_lock<std::mutex> lock(metrics.mutexes_.sampling);
  return metrics.caches_.gpu.memory_total.size();
}

void
starpu_server::testing::MetricsRegistryTestAccessor::SampleProcessOpenFds(
    starpu_server::MetricsRegistry& metrics)
{
  if (metrics.sampler_ != nullptr) {
    metrics.sampler_->sample_process_open_fds();
  }
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    SampleProcessResidentMemory(starpu_server::MetricsRegistry& metrics)
{
  if (metrics.sampler_ != nullptr) {
    metrics.sampler_->sample_process_resident_memory();
  }
}

void
starpu_server::testing::MetricsRegistryTestAccessor::SampleInferenceThroughput(
    starpu_server::MetricsRegistry& metrics)
{
  if (metrics.sampler_ != nullptr) {
    metrics.sampler_->sample_inference_throughput();
  }
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    ClearStarpuWorkerInflightFamily(starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.starpu_worker_inflight = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    ClearStarpuTaskRuntimeByWorkerFamily(
        starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.starpu_task_runtime_by_worker = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    ClearInferenceComputeLatencyByWorkerFamily(
        starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.inference_compute_latency_by_worker = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::ClearIoCopyLatencyFamily(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.io_copy_latency = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::ClearTransferBytesFamily(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.transfer_bytes = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::ClearModelsLoadedFamily(
    starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.models_loaded = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    ClearGpuModelReplicationPolicyInfoFamily(
        starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.gpu_model_replication_policy_info = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    ClearGpuModelReplicasTotalFamily(starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.gpu_model_replicas_total = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    ClearStarpuCudaWorkerInfoFamily(starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.starpu_cuda_worker_info = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    ClearModelLoadFailuresFamily(starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.model_load_failures = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    ClearInferenceFailuresFamily(starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.inference_failures = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    ClearInferenceCompletedFamily(starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.inference_completed = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    ClearRequestsReceivedFamily(starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.requests_received = nullptr;
}

void
starpu_server::testing::MetricsRegistryTestAccessor::
    ClearRequestsByStatusFamily(starpu_server::MetricsRegistry& metrics)
{
  metrics.families_.requests_by_status = nullptr;
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::FailureKeyOverflowIsEmpty()
    -> bool
{
  const auto key = MetricsRegistry::FailureKey::Overflow();
  return key.overflow && key.stage.empty() && key.reason.empty() &&
         key.model.empty();
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::FailureKeyEquals(
    std::string_view stage_lhs, std::string_view reason_lhs,
    std::string_view model_lhs, bool overflow_lhs, std::string_view stage_rhs,
    std::string_view reason_rhs, std::string_view model_rhs,
    bool overflow_rhs) -> bool
{
  MetricsRegistry::FailureKey lhs{
      std::string(stage_lhs), std::string(reason_lhs), std::string(model_lhs),
      overflow_lhs};
  MetricsRegistry::FailureKey rhs{
      std::string(stage_rhs), std::string(reason_rhs), std::string(model_rhs),
      overflow_rhs};
  return lhs == rhs;
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::ModelKeyOverflowIsEmpty()
    -> bool
{
  const auto key = MetricsRegistry::ModelKey::Overflow();
  return key.overflow && key.model.empty();
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::ModelKeyEquals(
    std::string_view model_lhs, bool overflow_lhs, std::string_view model_rhs,
    bool overflow_rhs) -> bool
{
  MetricsRegistry::ModelKey lhs{std::string(model_lhs), overflow_lhs};
  MetricsRegistry::ModelKey rhs{std::string(model_rhs), overflow_rhs};
  return lhs == rhs;
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::
    ModelPolicyKeyOverflowIsEmpty() -> bool
{
  const auto key = MetricsRegistry::ModelPolicyKey::Overflow();
  return key.overflow && key.model.empty() && key.policy.empty();
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::ModelPolicyKeyEquals(
    std::string_view model_lhs, std::string_view policy_lhs, bool overflow_lhs,
    std::string_view model_rhs, std::string_view policy_rhs,
    bool overflow_rhs) -> bool
{
  MetricsRegistry::ModelPolicyKey lhs{
      std::string(model_lhs), std::string(policy_lhs), overflow_lhs};
  MetricsRegistry::ModelPolicyKey rhs{
      std::string(model_rhs), std::string(policy_rhs), overflow_rhs};
  return lhs == rhs;
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::
    ModelDeviceKeyOverflowIsEmpty() -> bool
{
  const auto key = MetricsRegistry::ModelDeviceKey::Overflow();
  return key.overflow && key.model.empty() && key.device.empty();
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::ModelDeviceKeyEquals(
    std::string_view model_lhs, std::string_view device_lhs, bool overflow_lhs,
    std::string_view model_rhs, std::string_view device_rhs,
    bool overflow_rhs) -> bool
{
  MetricsRegistry::ModelDeviceKey lhs{
      std::string(model_lhs), std::string(device_lhs), overflow_lhs};
  MetricsRegistry::ModelDeviceKey rhs{
      std::string(model_rhs), std::string(device_rhs), overflow_rhs};
  return lhs == rhs;
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::IoKeyOverflowIsEmpty()
    -> bool
{
  const auto key = MetricsRegistry::IoKey::Overflow();
  return key.overflow && key.direction.empty() && key.worker_id == 0 &&
         key.device_id == 0 && key.worker_type.empty();
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::IoKeyEquals(
    std::string_view direction_lhs, int worker_id_lhs, int device_id_lhs,
    std::string_view worker_type_lhs, bool overflow_lhs,
    std::string_view direction_rhs, int worker_id_rhs, int device_id_rhs,
    std::string_view worker_type_rhs, bool overflow_rhs) -> bool
{
  MetricsRegistry::IoKey lhs{
      std::string(direction_lhs), worker_id_lhs, device_id_lhs,
      std::string(worker_type_lhs), overflow_lhs};
  MetricsRegistry::IoKey rhs{
      std::string(direction_rhs), worker_id_rhs, device_id_rhs,
      std::string(worker_type_rhs), overflow_rhs};
  return lhs == rhs;
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::WorkerKeyOverflowIsEmpty()
    -> bool
{
  const auto key = MetricsRegistry::WorkerKey::Overflow();
  return key.overflow && key.worker_id == 0 && key.device_id == 0 &&
         key.worker_type.empty();
}

auto
starpu_server::testing::MetricsRegistryTestAccessor::WorkerKeyEquals(
    int worker_id_lhs, int device_id_lhs, std::string_view worker_type_lhs,
    bool overflow_lhs, int worker_id_rhs, int device_id_rhs,
    std::string_view worker_type_rhs, bool overflow_rhs) -> bool
{
  MetricsRegistry::WorkerKey lhs{
      worker_id_lhs, device_id_lhs, std::string(worker_type_lhs), overflow_lhs};
  MetricsRegistry::WorkerKey rhs{
      worker_id_rhs, device_id_rhs, std::string(worker_type_rhs), overflow_rhs};
  return lhs == rhs;
}
#endif  // SONAR_IGNORE_END
