#pragma once

#include <prometheus/counter.h>
#include <prometheus/gauge.h>
#include <prometheus/histogram.h>
#include <prometheus/registry.h>

#include <atomic>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <iosfwd>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace prometheus {
class Exposer;
class Collectable;
template <typename T>
class Family;
}  // namespace prometheus

namespace starpu_server {

class MetricsRegistry {
 public:
  struct ExposerHandle {
    ExposerHandle() = default;
    ExposerHandle(const ExposerHandle&) = delete;
    auto operator=(const ExposerHandle&) -> ExposerHandle& = delete;
    ExposerHandle(ExposerHandle&&) = delete;
    auto operator=(ExposerHandle&&) -> ExposerHandle& = delete;
    virtual ~ExposerHandle() = default;
    virtual void RegisterCollectable(
        const std::shared_ptr<prometheus::Collectable>& collectable) = 0;
    virtual void RemoveCollectable(
        const std::shared_ptr<prometheus::Collectable>& collectable) = 0;
  };

  struct GpuSample {
    int index{0};
    double util_percent{0.0};
    double mem_used_bytes{0.0};
    double mem_total_bytes{0.0};
    double temperature_celsius{std::numeric_limits<double>::quiet_NaN()};
    double power_watts{std::numeric_limits<double>::quiet_NaN()};
  };

  struct StatusCodeLabel {
    std::string_view value;
  };

  struct FailureStageLabel {
    std::string_view value;
  };

  struct FailureReasonLabel {
    std::string_view value;
  };

  struct ModelLabel {
    std::string_view value;
  };

  struct DeviceLabel {
    std::string_view value;
  };

  using GpuStatsProvider = std::function<std::vector<GpuSample>()>;
  using CpuUsageProvider = std::function<std::optional<double>()>;

  explicit MetricsRegistry(int port);
  MetricsRegistry(
      int port, GpuStatsProvider gpu_provider, CpuUsageProvider cpu_provider,
      bool start_sampler_thread = true,
      std::unique_ptr<ExposerHandle> exposer_handle = nullptr);
  ~MetricsRegistry() noexcept;
  MetricsRegistry(const MetricsRegistry&) = delete;
  auto operator=(const MetricsRegistry&) -> MetricsRegistry& = delete;
  MetricsRegistry(MetricsRegistry&&) = delete;
  auto operator=(MetricsRegistry&&) -> MetricsRegistry& = delete;

  void run_sampling_request_nb();
  void request_stop();

  auto has_gpu_stats_provider() const -> bool;
  auto has_cpu_usage_provider() const -> bool;

  [[nodiscard]] auto registry() const -> std::shared_ptr<prometheus::Registry>;
  [[nodiscard]] auto requests_total() const -> prometheus::Counter*;
  [[nodiscard]] auto requests_rejected_total() const -> prometheus::Counter*;
  [[nodiscard]] auto inference_latency() const -> prometheus::Histogram*;
  [[nodiscard]] auto queue_size_gauge() const -> prometheus::Gauge*;
  [[nodiscard]] auto inflight_tasks_gauge() const -> prometheus::Gauge*;
  [[nodiscard]] auto max_inflight_tasks_gauge() const -> prometheus::Gauge*;
  [[nodiscard]] auto system_cpu_usage_percent() const -> prometheus::Gauge*;
  [[nodiscard]] auto server_health_state_gauge() const -> prometheus::Gauge*;
  [[nodiscard]] auto queue_fill_ratio_gauge() const -> prometheus::Gauge*;
  [[nodiscard]] auto queue_capacity_gauge() const -> prometheus::Gauge*;
  [[nodiscard]] auto queue_latency_histogram() const -> prometheus::Histogram*;
  [[nodiscard]] auto batch_collect_latency_histogram() const
      -> prometheus::Histogram*;
  [[nodiscard]] auto batch_efficiency_histogram() const
      -> prometheus::Histogram*;
  [[nodiscard]] auto batch_pending_jobs_gauge() const -> prometheus::Gauge*;
  [[nodiscard]] auto submit_latency_histogram() const -> prometheus::Histogram*;
  [[nodiscard]] auto scheduling_latency_histogram() const
      -> prometheus::Histogram*;
  [[nodiscard]] auto codelet_latency_histogram() const
      -> prometheus::Histogram*;
  [[nodiscard]] auto inference_compute_latency_histogram() const
      -> prometheus::Histogram*;
  [[nodiscard]] auto callback_latency_histogram() const
      -> prometheus::Histogram*;
  [[nodiscard]] auto preprocess_latency_histogram() const
      -> prometheus::Histogram*;
  [[nodiscard]] auto postprocess_latency_histogram() const
      -> prometheus::Histogram*;
  [[nodiscard]] auto batch_size_histogram() const -> prometheus::Histogram*;
  [[nodiscard]] auto logical_batch_size_histogram() const
      -> prometheus::Histogram*;
  [[nodiscard]] auto gpu_utilization_family() const
      -> prometheus::Family<prometheus::Gauge>*;
  [[nodiscard]] auto gpu_memory_used_bytes_family() const
      -> prometheus::Family<prometheus::Gauge>*;
  [[nodiscard]] auto gpu_memory_total_bytes_family() const
      -> prometheus::Family<prometheus::Gauge>*;
  [[nodiscard]] auto gpu_temperature_family() const
      -> prometheus::Family<prometheus::Gauge>*;
  [[nodiscard]] auto gpu_power_family() const
      -> prometheus::Family<prometheus::Gauge>*;
  [[nodiscard]] auto requests_by_status_family() const
      -> prometheus::Family<prometheus::Counter>*;
  [[nodiscard]] auto inference_completed_family() const
      -> prometheus::Family<prometheus::Counter>*;
  [[nodiscard]] auto inference_failures_family() const
      -> prometheus::Family<prometheus::Counter>*;
  [[nodiscard]] auto model_load_failures_family() const
      -> prometheus::Family<prometheus::Counter>*;
  [[nodiscard]] auto models_loaded_family() const
      -> prometheus::Family<prometheus::Gauge>*;
  [[nodiscard]] auto starpu_worker_busy_ratio_gauge() const
      -> prometheus::Gauge*;
  [[nodiscard]] auto starpu_prepared_queue_depth_gauge() const
      -> prometheus::Gauge*;
  [[nodiscard]] auto inference_throughput_gauge() const -> prometheus::Gauge*;
  [[nodiscard]] auto process_resident_memory_gauge() const
      -> prometheus::Gauge*;
  [[nodiscard]] auto process_open_fds_gauge() const -> prometheus::Gauge*;
  [[nodiscard]] auto model_load_duration_histogram() const
      -> prometheus::Histogram*;
  [[nodiscard]] auto starpu_task_runtime_histogram() const
      -> prometheus::Histogram*;
  [[nodiscard]] auto inference_compute_latency_by_worker_family() const
      -> prometheus::Family<prometheus::Histogram>*;
  [[nodiscard]] auto starpu_task_runtime_by_worker_family() const
      -> prometheus::Family<prometheus::Histogram>*;
  [[nodiscard]] auto starpu_worker_inflight_family() const
      -> prometheus::Family<prometheus::Gauge>*;
  [[nodiscard]] auto io_copy_latency_family() const
      -> prometheus::Family<prometheus::Histogram>*;
  [[nodiscard]] auto transfer_bytes_family() const
      -> prometheus::Family<prometheus::Counter>*;

  void increment_status_counter(
      StatusCodeLabel code_label, ModelLabel model_label);
  void increment_completed_counter(
      std::string_view model_label, std::size_t logical_jobs);
  void increment_failure_counter(
      FailureStageLabel stage_label, FailureReasonLabel reason_label,
      ModelLabel model_label, std::size_t count);
  void increment_model_load_failure_counter(std::string_view model_label);
  void set_model_loaded_flag(
      ModelLabel model_label, DeviceLabel device_label, bool loaded);
  void set_queue_capacity(std::size_t capacity);
  [[nodiscard]] auto queue_capacity_value() const -> std::size_t;
  void observe_compute_latency_by_worker(
      int worker_id, int device_id, std::string_view worker_type,
      double latency_ms);
  void observe_task_runtime_by_worker(
      int worker_id, int device_id, std::string_view worker_type,
      double latency_ms);
  void set_worker_inflight_gauge(
      int worker_id, int device_id, std::string_view worker_type,
      std::size_t value);
  void observe_io_copy_latency(
      std::string_view direction, int worker_id, int device_id,
      std::string_view worker_type, double duration_ms);
  void increment_transfer_bytes(
      std::string_view direction, int worker_id, int device_id,
      std::string_view worker_type, std::size_t bytes);

#if defined(STARPU_TESTING)
  struct TestAccessor {
    static void ClearCpuUsageProvider(MetricsRegistry& metrics);
    static void ClearStarpuWorkerInflightFamily(MetricsRegistry& metrics);
    static void ClearStarpuTaskRuntimeByWorkerFamily(MetricsRegistry& metrics);
    static void ClearInferenceComputeLatencyByWorkerFamily(
        MetricsRegistry& metrics);
    static void ClearIoCopyLatencyFamily(MetricsRegistry& metrics);
    static void ClearTransferBytesFamily(MetricsRegistry& metrics);
    static void ClearModelsLoadedFamily(MetricsRegistry& metrics);
    static void ClearModelLoadFailuresFamily(MetricsRegistry& metrics);
    static void ClearInferenceFailuresFamily(MetricsRegistry& metrics);
    static void ClearInferenceCompletedFamily(MetricsRegistry& metrics);
    static void ClearRequestsByStatusFamily(MetricsRegistry& metrics);
  };
#endif

 private:
  void initialize(
      int port, bool start_sampler_thread,
      std::unique_ptr<ExposerHandle> exposer_handle);
  void sample_cpu_usage();
  void sample_inference_throughput();
  void sample_process_resident_memory();
  void sample_process_open_fds();
  void sample_gpu_stats();
  void perform_sampling_request_nb();
  void sampling_loop(const std::stop_token& stop);

  std::shared_ptr<prometheus::Registry> registry_;
  prometheus::Counter* requests_total_{nullptr};
  prometheus::Counter* requests_rejected_total_{nullptr};
  prometheus::Histogram* inference_latency_{nullptr};
  prometheus::Gauge* queue_size_gauge_{nullptr};
  prometheus::Gauge* inflight_tasks_gauge_{nullptr};
  prometheus::Gauge* max_inflight_tasks_gauge_{nullptr};
  prometheus::Gauge* system_cpu_usage_percent_{nullptr};
  prometheus::Gauge* server_health_state_{nullptr};
  prometheus::Gauge* starpu_worker_busy_ratio_{nullptr};
  prometheus::Gauge* starpu_prepared_queue_depth_{nullptr};
  prometheus::Gauge* inference_throughput_gauge_{nullptr};
  prometheus::Gauge* process_resident_memory_bytes_{nullptr};
  prometheus::Gauge* process_open_fds_{nullptr};
  prometheus::Gauge* queue_fill_ratio_gauge_{nullptr};
  prometheus::Gauge* queue_capacity_gauge_{nullptr};
  prometheus::Histogram* queue_latency_histogram_{nullptr};
  prometheus::Histogram* batch_collect_latency_histogram_{nullptr};
  prometheus::Gauge* batch_pending_jobs_gauge_{nullptr};
  prometheus::Histogram* batch_efficiency_histogram_{nullptr};
  prometheus::Histogram* submit_latency_histogram_{nullptr};
  prometheus::Histogram* scheduling_latency_histogram_{nullptr};
  prometheus::Histogram* codelet_latency_histogram_{nullptr};
  prometheus::Histogram* inference_compute_latency_histogram_{nullptr};
  prometheus::Histogram* callback_latency_histogram_{nullptr};
  prometheus::Histogram* preprocess_latency_histogram_{nullptr};
  prometheus::Histogram* postprocess_latency_histogram_{nullptr};
  prometheus::Histogram* batch_size_histogram_{nullptr};
  prometheus::Histogram* logical_batch_size_histogram_{nullptr};
  prometheus::Histogram* model_load_duration_histogram_{nullptr};
  prometheus::Histogram* starpu_task_runtime_histogram_{nullptr};
  prometheus::Family<prometheus::Histogram>*
      inference_compute_latency_by_worker_family_{nullptr};
  prometheus::Family<prometheus::Histogram>*
      starpu_task_runtime_by_worker_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* starpu_worker_inflight_family_{
      nullptr};
  prometheus::Family<prometheus::Histogram>* io_copy_latency_family_{nullptr};
  prometheus::Family<prometheus::Counter>* transfer_bytes_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* gpu_utilization_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* gpu_memory_used_bytes_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* gpu_memory_total_bytes_family_{
      nullptr};
  prometheus::Family<prometheus::Gauge>* gpu_temperature_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* gpu_power_family_{nullptr};
  prometheus::Family<prometheus::Counter>* requests_by_status_family_{nullptr};
  prometheus::Family<prometheus::Counter>* inference_completed_family_{nullptr};
  prometheus::Family<prometheus::Counter>* inference_failures_family_{nullptr};
  prometheus::Family<prometheus::Counter>* model_load_failures_family_{nullptr};
  prometheus::Family<prometheus::Gauge>* models_loaded_family_{nullptr};

  std::unique_ptr<ExposerHandle> exposer_;
  std::jthread sampler_thread_;
  GpuStatsProvider gpu_stats_provider_;
  CpuUsageProvider cpu_usage_provider_;
  std::atomic<std::size_t> queue_capacity_{0};
  std::unordered_map<int, prometheus::Gauge*> gpu_utilization_gauges_;
  std::unordered_map<int, prometheus::Gauge*> gpu_memory_used_gauges_;
  std::unordered_map<int, prometheus::Gauge*> gpu_memory_total_gauges_;
  std::unordered_map<int, prometheus::Gauge*> gpu_temperature_gauges_;
  std::unordered_map<int, prometheus::Gauge*> gpu_power_gauges_;
  std::unordered_map<std::string, prometheus::Counter*> status_counters_;
  std::unordered_map<std::string, prometheus::Counter*>
      inference_completed_counters_;
  std::unordered_map<std::string, prometheus::Counter*>
      inference_failure_counters_;
  std::unordered_map<std::string, prometheus::Counter*>
      model_load_failure_counters_;
  std::unordered_map<std::string, prometheus::Gauge*> models_loaded_gauges_;
  std::unordered_map<std::string, prometheus::Histogram*>
      compute_latency_by_worker_;
  std::unordered_map<std::string, prometheus::Histogram*>
      task_runtime_by_worker_;
  std::unordered_map<std::string, prometheus::Gauge*> worker_inflight_gauges_;
  std::unordered_map<std::string, prometheus::Histogram*> io_copy_latency_;
  std::unordered_map<std::string, prometheus::Counter*> transfer_bytes_;
  std::mutex sampling_mutex_;
  std::mutex status_mutex_;
};

auto init_metrics(int port) -> bool;
void shutdown_metrics();
void set_queue_size(std::size_t size);
void set_inflight_tasks(std::size_t size);
void set_starpu_worker_busy_ratio(double ratio);
void set_max_inflight_tasks(std::size_t max_tasks);
void set_queue_capacity(std::size_t capacity);
void set_queue_fill_ratio(std::size_t size, std::size_t capacity);
void set_server_health(bool ready);
void set_starpu_prepared_queue_depth(std::size_t depth);
void set_batch_pending_jobs(std::size_t pending);
void increment_request_status(int status_code, std::string_view model_name);
void increment_inference_completed(
    std::string_view model_name, std::size_t logical_jobs);
void increment_inference_failure(
    std::string_view stage, std::string_view reason,
    std::string_view model_name, std::size_t count = 1);
void observe_batch_size(std::size_t batch_size);
void observe_logical_batch_size(std::size_t logical_jobs);
void observe_batch_efficiency(double ratio);
void observe_latency_breakdown(
    double queue_ms, double batch_ms, double submit_ms, double scheduling_ms,
    double codelet_ms, double inference_ms, double callback_ms,
    double preprocess_ms, double postprocess_ms);
void observe_starpu_task_runtime(double runtime_ms);
void observe_model_load_duration(double duration_ms);
void set_model_loaded(
    std::string_view model_name, std::string_view device_label, bool loaded);
void increment_model_load_failure(std::string_view model_name);
void increment_rejected_requests();
void observe_compute_latency_by_worker(
    int worker_id, int device_id, std::string_view worker_type,
    double latency_ms);
void observe_task_runtime_by_worker(
    int worker_id, int device_id, std::string_view worker_type,
    double latency_ms);
void set_worker_inflight_gauge(
    int worker_id, int device_id, std::string_view worker_type,
    std::size_t value);
void observe_io_copy_latency(
    std::string_view direction, int worker_id, int device_id,
    std::string_view worker_type, double duration_ms);
void increment_transfer_bytes(
    std::string_view direction, int worker_id, int device_id,
    std::string_view worker_type, std::size_t bytes);
auto get_metrics() -> std::shared_ptr<MetricsRegistry>;

}  // namespace starpu_server

namespace starpu_server::monitoring::detail {

struct CpuTotals {
  unsigned long long user{0};
  unsigned long long nice{0};
  unsigned long long system{0};
  unsigned long long idle{0};
  unsigned long long iowait{0};
  unsigned long long irq{0};
  unsigned long long softirq{0};
  unsigned long long steal{0};
};

auto read_total_cpu_times(std::istream& input, CpuTotals& out) -> bool;
auto read_total_cpu_times(const std::filesystem::path& path, CpuTotals& out)
    -> bool;
auto make_cpu_usage_provider(std::function<bool(CpuTotals&)> reader)
    -> starpu_server::MetricsRegistry::CpuUsageProvider;
auto cpu_usage_percent(const CpuTotals& prev, const CpuTotals& curr) -> double;
auto read_process_open_fds() -> std::optional<double>;
auto read_process_rss_bytes() -> std::optional<double>;
#if defined(STARPU_TESTING)
void set_process_open_fds_reader_override(
    std::function<std::optional<double>()> reader);
void set_process_rss_bytes_reader_override(
    std::function<std::optional<double>()> reader);
void set_metrics_init_failure_for_test(bool fail);
auto metrics_init_failure_for_test() -> bool;
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
#endif

}  // namespace starpu_server::monitoring::detail
