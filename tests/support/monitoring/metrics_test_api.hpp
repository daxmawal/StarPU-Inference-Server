#pragma once

#if !defined(STARPU_TESTING)
#error "metrics_test_api.hpp is test-only and requires STARPU_TESTING"
#endif

#include "monitoring/metrics.hpp"

namespace starpu_server::testing {

class MetricsRegistryTestAccessor {
 public:
  static void ClearCpuUsageProvider(MetricsRegistry& metrics);
  static void ClearQueueSizeGauge(MetricsRegistry& metrics);
  static void ClearQueueFillRatioGauge(MetricsRegistry& metrics);
  static void ClearSystemCpuUsageGauge(MetricsRegistry& metrics);
  static void ClearProcessOpenFdsGauge(MetricsRegistry& metrics);
  static void ClearProcessResidentMemoryGauge(MetricsRegistry& metrics);
  static void ClearInferenceThroughputGauge(MetricsRegistry& metrics);
  static void ClearGpuStatsProvider(MetricsRegistry& metrics);
  static auto ProcessOpenFdsGauge(MetricsRegistry& metrics)
      -> prometheus::Gauge*;
  static auto ProcessResidentMemoryGauge(MetricsRegistry& metrics)
      -> prometheus::Gauge*;
  static auto InferenceThroughputGauge(MetricsRegistry& metrics)
      -> prometheus::Gauge*;
  static auto GpuUtilizationGaugeCount(const MetricsRegistry& metrics)
      -> std::size_t;
  static auto GpuMemoryUsedGaugeCount(const MetricsRegistry& metrics)
      -> std::size_t;
  static auto GpuMemoryTotalGaugeCount(const MetricsRegistry& metrics)
      -> std::size_t;
  static void SampleProcessOpenFds(MetricsRegistry& metrics);
  static void SampleProcessResidentMemory(MetricsRegistry& metrics);
  static void SampleInferenceThroughput(MetricsRegistry& metrics);
  static void ClearStarpuWorkerInflightFamily(MetricsRegistry& metrics);
  static void ClearStarpuTaskRuntimeByWorkerFamily(MetricsRegistry& metrics);
  static void ClearInferenceComputeLatencyByWorkerFamily(
      MetricsRegistry& metrics);
  static void ClearIoCopyLatencyFamily(MetricsRegistry& metrics);
  static void ClearTransferBytesFamily(MetricsRegistry& metrics);
  static void ClearModelsLoadedFamily(MetricsRegistry& metrics);
  static void ClearGpuModelReplicationPolicyInfoFamily(
      MetricsRegistry& metrics);
  static void ClearGpuModelReplicasTotalFamily(MetricsRegistry& metrics);
  static void ClearStarpuCudaWorkerInfoFamily(MetricsRegistry& metrics);
  static void ClearModelLoadFailuresFamily(MetricsRegistry& metrics);
  static void ClearInferenceFailuresFamily(MetricsRegistry& metrics);
  static void ClearInferenceCompletedFamily(MetricsRegistry& metrics);
  static void ClearRequestsReceivedFamily(MetricsRegistry& metrics);
  static void ClearRequestsByStatusFamily(MetricsRegistry& metrics);
  static auto FailureKeyOverflowIsEmpty() -> bool;
  static auto FailureKeyEquals(
      std::string_view stage_lhs, std::string_view reason_lhs,
      std::string_view model_lhs, bool overflow_lhs, std::string_view stage_rhs,
      std::string_view reason_rhs, std::string_view model_rhs,
      bool overflow_rhs) -> bool;
  static auto ModelKeyOverflowIsEmpty() -> bool;
  static auto ModelKeyEquals(
      std::string_view model_lhs, bool overflow_lhs, std::string_view model_rhs,
      bool overflow_rhs) -> bool;
  static auto ModelPolicyKeyOverflowIsEmpty() -> bool;
  static auto ModelPolicyKeyEquals(
      std::string_view model_lhs, std::string_view policy_lhs,
      bool overflow_lhs, std::string_view model_rhs,
      std::string_view policy_rhs, bool overflow_rhs) -> bool;
  static auto ModelDeviceKeyOverflowIsEmpty() -> bool;
  static auto ModelDeviceKeyEquals(
      std::string_view model_lhs, std::string_view device_lhs,
      bool overflow_lhs, std::string_view model_rhs,
      std::string_view device_rhs, bool overflow_rhs) -> bool;
  static auto IoKeyOverflowIsEmpty() -> bool;
  static auto IoKeyEquals(
      std::string_view direction_lhs, int worker_id_lhs, int device_id_lhs,
      std::string_view worker_type_lhs, bool overflow_lhs,
      std::string_view direction_rhs, int worker_id_rhs, int device_id_rhs,
      std::string_view worker_type_rhs, bool overflow_rhs) -> bool;
  static auto WorkerKeyOverflowIsEmpty() -> bool;
  static auto WorkerKeyEquals(
      int worker_id_lhs, int device_id_lhs, std::string_view worker_type_lhs,
      bool overflow_lhs, int worker_id_rhs, int device_id_rhs,
      std::string_view worker_type_rhs, bool overflow_rhs) -> bool;
};

}  // namespace starpu_server::testing

namespace starpu_server::monitoring::detail {

void set_process_open_fds_reader_override(
    std::function<std::optional<double>()> reader);
void set_process_rss_bytes_reader_override(
    std::function<std::optional<double>()> reader);
void set_metrics_init_failure_for_test(bool fail);
auto metrics_init_failure_for_test() -> bool;
void set_metrics_request_stop_skip_join_for_test(bool skip_join);
auto metrics_request_stop_skip_join_for_test() -> bool;
void set_process_fd_path_for_test(std::filesystem::path path);
void reset_process_fd_path_for_test();
auto process_fd_path_for_test() -> const std::filesystem::path&;
using ProcessFdDirectoryIteratorFactory =
    std::function<std::filesystem::directory_iterator(
        const std::filesystem::path&)>;
void set_process_fd_directory_iterator_for_test(
    ProcessFdDirectoryIteratorFactory factory);
void reset_process_fd_directory_iterator_for_test();
auto process_fd_directory_iterator_for_test()
    -> ProcessFdDirectoryIteratorFactory;
void set_process_rss_bytes_path_for_test(std::filesystem::path path);
void reset_process_rss_bytes_path_for_test();
auto process_rss_bytes_path_for_test() -> const std::filesystem::path&;
using ProcessPageSizeProvider = std::function<long()>;
void set_process_page_size_provider_for_test(ProcessPageSizeProvider provider);
void reset_process_page_size_provider_for_test();
auto process_page_size_provider_for_test() -> const ProcessPageSizeProvider&;
auto should_log_sampling_error_for_test(std::atomic<std::int64_t>& last_log)
    -> bool;
auto status_code_label_for_test(int code) -> std::string;

}  // namespace starpu_server::monitoring::detail
