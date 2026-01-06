#pragma once

#include <prometheus/counter.h>
#include <prometheus/gauge.h>
#include <prometheus/histogram.h>
#include <prometheus/registry.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
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

  struct CounterMetrics {
    prometheus::Counter* requests_total{nullptr};
    prometheus::Counter* requests_rejected_total{nullptr};
  };

  struct GaugeMetrics {
    prometheus::Gauge* queue_size{nullptr};
    prometheus::Gauge* inflight_tasks{nullptr};
    prometheus::Gauge* max_inflight_tasks{nullptr};
    prometheus::Gauge* system_cpu_usage_percent{nullptr};
    prometheus::Gauge* server_health_state{nullptr};
    prometheus::Gauge* starpu_worker_busy_ratio{nullptr};
    prometheus::Gauge* starpu_prepared_queue_depth{nullptr};
    prometheus::Gauge* inference_throughput{nullptr};
    prometheus::Gauge* process_resident_memory_bytes{nullptr};
    prometheus::Gauge* process_open_fds{nullptr};
    prometheus::Gauge* queue_fill_ratio{nullptr};
    prometheus::Gauge* queue_capacity{nullptr};
    prometheus::Gauge* batch_pending_jobs{nullptr};
  };

  struct HistogramMetrics {
    prometheus::Histogram* inference_latency{nullptr};
    prometheus::Histogram* queue_latency{nullptr};
    prometheus::Histogram* batch_collect_latency{nullptr};
    prometheus::Histogram* batch_efficiency{nullptr};
    prometheus::Histogram* submit_latency{nullptr};
    prometheus::Histogram* scheduling_latency{nullptr};
    prometheus::Histogram* codelet_latency{nullptr};
    prometheus::Histogram* inference_compute_latency{nullptr};
    prometheus::Histogram* callback_latency{nullptr};
    prometheus::Histogram* preprocess_latency{nullptr};
    prometheus::Histogram* postprocess_latency{nullptr};
    prometheus::Histogram* batch_size{nullptr};
    prometheus::Histogram* logical_batch_size{nullptr};
    prometheus::Histogram* model_load_duration{nullptr};
    prometheus::Histogram* starpu_task_runtime{nullptr};
  };

  struct FamilyMetrics {
    prometheus::Family<prometheus::Histogram>*
        inference_compute_latency_by_worker{nullptr};
    prometheus::Family<prometheus::Histogram>* starpu_task_runtime_by_worker{
        nullptr};
    prometheus::Family<prometheus::Gauge>* starpu_worker_inflight{nullptr};
    prometheus::Family<prometheus::Histogram>* io_copy_latency{nullptr};
    prometheus::Family<prometheus::Counter>* transfer_bytes{nullptr};
    prometheus::Family<prometheus::Gauge>* gpu_utilization{nullptr};
    prometheus::Family<prometheus::Gauge>* gpu_memory_used_bytes{nullptr};
    prometheus::Family<prometheus::Gauge>* gpu_memory_total_bytes{nullptr};
    prometheus::Family<prometheus::Gauge>* gpu_temperature{nullptr};
    prometheus::Family<prometheus::Gauge>* gpu_power{nullptr};
    prometheus::Family<prometheus::Counter>* requests_by_status{nullptr};
    prometheus::Family<prometheus::Counter>* inference_completed{nullptr};
    prometheus::Family<prometheus::Counter>* inference_failures{nullptr};
    prometheus::Family<prometheus::Counter>* model_load_failures{nullptr};
    prometheus::Family<prometheus::Gauge>* models_loaded{nullptr};
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

  void request_stop();

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  void run_sampling_request_nb();
  auto has_gpu_stats_provider() const -> bool;
  auto has_cpu_usage_provider() const -> bool;
#endif  // SONAR_IGNORE_END

  [[nodiscard]] auto registry() const -> std::shared_ptr<prometheus::Registry>;
  auto counters() -> CounterMetrics& { return counters_; }
  auto gauges() -> GaugeMetrics& { return gauges_; }
  auto histograms() -> HistogramMetrics& { return histograms_; }
  auto families() -> FamilyMetrics& { return families_; }

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

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  struct TestAccessor {
    static void ClearCpuUsageProvider(MetricsRegistry& metrics);
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
    static void ClearModelLoadFailuresFamily(MetricsRegistry& metrics);
    static void ClearInferenceFailuresFamily(MetricsRegistry& metrics);
    static void ClearInferenceCompletedFamily(MetricsRegistry& metrics);
    static void ClearRequestsByStatusFamily(MetricsRegistry& metrics);
    static auto FailureKeyOverflowIsEmpty() -> bool;
    static auto FailureKeyEquals(
        std::string_view stage_lhs, std::string_view reason_lhs,
        std::string_view model_lhs, bool overflow_lhs,
        std::string_view stage_rhs, std::string_view reason_rhs,
        std::string_view model_rhs, bool overflow_rhs) -> bool;
    static auto ModelKeyOverflowIsEmpty() -> bool;
    static auto ModelKeyEquals(
        std::string_view model_lhs, bool overflow_lhs,
        std::string_view model_rhs, bool overflow_rhs) -> bool;
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
#endif  // SONAR_IGNORE_END

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

  static constexpr std::size_t kHashCombineMagic = 0x9e3779b97f4a7c15ULL;
  static constexpr std::size_t kHashCombineShiftLeft = 6U;
  static constexpr std::size_t kHashCombineShiftRight = 2U;

  static auto HashCombine(std::size_t seed, std::size_t value) noexcept
      -> std::size_t
  {
    return seed ^ (value + kHashCombineMagic + (seed << kHashCombineShiftLeft) +
                   (seed >> kHashCombineShiftRight));
  }

  struct StatusKey {
    std::string code;
    std::string model;
    bool overflow{false};

    static auto Overflow() -> StatusKey
    {
      StatusKey key;
      key.overflow = true;
      return key;
    }

    auto operator==(const StatusKey& other) const noexcept -> bool
    {
      return overflow == other.overflow && code == other.code &&
             model == other.model;
    }
  };

  struct StatusKeyHash {
    auto operator()(const StatusKey& key) const noexcept -> std::size_t
    {
      std::size_t seed = 0;
      seed = MetricsRegistry::HashCombine(
          seed, std::hash<std::string>{}(key.code));
      seed = MetricsRegistry::HashCombine(
          seed, std::hash<std::string>{}(key.model));
      seed = MetricsRegistry::HashCombine(
          seed, static_cast<std::size_t>(key.overflow));
      return seed;
    }
  };

  struct FailureKey {
    std::string stage;
    std::string reason;
    std::string model;
    bool overflow{false};

    static auto Overflow() -> FailureKey
    {
      FailureKey key;
      key.overflow = true;
      return key;
    }

    auto operator==(const FailureKey& other) const noexcept -> bool
    {
      return overflow == other.overflow && stage == other.stage &&
             reason == other.reason && model == other.model;
    }
  };

  struct FailureKeyHash {
    auto operator()(const FailureKey& key) const noexcept -> std::size_t
    {
      std::size_t seed = 0;
      seed = MetricsRegistry::HashCombine(
          seed, std::hash<std::string>{}(key.stage));
      seed = MetricsRegistry::HashCombine(
          seed, std::hash<std::string>{}(key.reason));
      seed = MetricsRegistry::HashCombine(
          seed, std::hash<std::string>{}(key.model));
      seed = MetricsRegistry::HashCombine(
          seed, static_cast<std::size_t>(key.overflow));
      return seed;
    }
  };

  struct ModelKey {
    std::string model;
    bool overflow{false};

    static auto Overflow() -> ModelKey
    {
      ModelKey key;
      key.overflow = true;
      return key;
    }

    auto operator==(const ModelKey& other) const noexcept -> bool
    {
      return overflow == other.overflow && model == other.model;
    }
  };

  struct ModelKeyHash {
    auto operator()(const ModelKey& key) const noexcept -> std::size_t
    {
      std::size_t seed = 0;
      seed = MetricsRegistry::HashCombine(
          seed, std::hash<std::string>{}(key.model));
      seed = MetricsRegistry::HashCombine(
          seed, static_cast<std::size_t>(key.overflow));
      return seed;
    }
  };

  struct ModelDeviceKey {
    std::string model;
    std::string device;
    bool overflow{false};

    static auto Overflow() -> ModelDeviceKey
    {
      ModelDeviceKey key;
      key.overflow = true;
      return key;
    }

    auto operator==(const ModelDeviceKey& other) const noexcept -> bool
    {
      return overflow == other.overflow && model == other.model &&
             device == other.device;
    }
  };

  struct ModelDeviceKeyHash {
    auto operator()(const ModelDeviceKey& key) const noexcept -> std::size_t
    {
      std::size_t seed = 0;
      seed = MetricsRegistry::HashCombine(
          seed, std::hash<std::string>{}(key.model));
      seed = MetricsRegistry::HashCombine(
          seed, std::hash<std::string>{}(key.device));
      seed = MetricsRegistry::HashCombine(
          seed, static_cast<std::size_t>(key.overflow));
      return seed;
    }
  };

  struct WorkerKey {
    int worker_id{0};
    int device_id{0};
    std::string worker_type;
    bool overflow{false};

    static auto Overflow() -> WorkerKey
    {
      WorkerKey key;
      key.overflow = true;
      return key;
    }

    auto operator==(const WorkerKey& other) const noexcept -> bool
    {
      return overflow == other.overflow && worker_id == other.worker_id &&
             device_id == other.device_id && worker_type == other.worker_type;
    }
  };

  struct WorkerKeyHash {
    auto operator()(const WorkerKey& key) const noexcept -> std::size_t
    {
      std::size_t seed = 0;
      seed =
          MetricsRegistry::HashCombine(seed, std::hash<int>{}(key.worker_id));
      seed =
          MetricsRegistry::HashCombine(seed, std::hash<int>{}(key.device_id));
      seed = MetricsRegistry::HashCombine(
          seed, std::hash<std::string>{}(key.worker_type));
      seed = MetricsRegistry::HashCombine(
          seed, static_cast<std::size_t>(key.overflow));
      return seed;
    }
  };

  struct IoKey {
    std::string direction;
    int worker_id{0};
    int device_id{0};
    std::string worker_type;
    bool overflow{false};

    static auto Overflow() -> IoKey
    {
      IoKey key;
      key.overflow = true;
      return key;
    }

    auto operator==(const IoKey& other) const noexcept -> bool
    {
      return overflow == other.overflow && direction == other.direction &&
             worker_id == other.worker_id && device_id == other.device_id &&
             worker_type == other.worker_type;
    }
  };

  struct IoKeyHash {
    auto operator()(const IoKey& key) const noexcept -> std::size_t
    {
      std::size_t seed = 0;
      seed = MetricsRegistry::HashCombine(
          seed, std::hash<std::string>{}(key.direction));
      seed =
          MetricsRegistry::HashCombine(seed, std::hash<int>{}(key.worker_id));
      seed =
          MetricsRegistry::HashCombine(seed, std::hash<int>{}(key.device_id));
      seed = MetricsRegistry::HashCombine(
          seed, std::hash<std::string>{}(key.worker_type));
      seed = MetricsRegistry::HashCombine(
          seed, static_cast<std::size_t>(key.overflow));
      return seed;
    }
  };

  struct RegistryState {
    std::shared_ptr<prometheus::Registry> registry;
    std::unique_ptr<ExposerHandle> exposer;
    std::jthread sampler_thread;
    std::atomic<std::size_t> queue_capacity{0};
  };

  struct Providers {
    GpuStatsProvider gpu_stats_provider;
    CpuUsageProvider cpu_usage_provider;
  };

  struct GpuGaugeCache {
    std::unordered_map<int, prometheus::Gauge*> utilization;
    std::unordered_map<int, prometheus::Gauge*> memory_used;
    std::unordered_map<int, prometheus::Gauge*> memory_total;
    std::unordered_map<int, prometheus::Gauge*> temperature;
    std::unordered_map<int, prometheus::Gauge*> power;
  };

  struct StatusCountersCache {
    std::unordered_map<StatusKey, prometheus::Counter*, StatusKeyHash> counters;
  };

  struct ModelMetricsCache {
    std::unordered_map<ModelKey, prometheus::Counter*, ModelKeyHash>
        inference_completed;
    std::unordered_map<FailureKey, prometheus::Counter*, FailureKeyHash>
        inference_failures;
    std::unordered_map<ModelKey, prometheus::Counter*, ModelKeyHash>
        model_load_failures;
    std::unordered_map<ModelDeviceKey, prometheus::Gauge*, ModelDeviceKeyHash>
        models_loaded;
  };

  struct WorkerMetricsCache {
    std::unordered_map<WorkerKey, prometheus::Histogram*, WorkerKeyHash>
        compute_latency;
    std::unordered_map<WorkerKey, prometheus::Histogram*, WorkerKeyHash>
        task_runtime;
    std::unordered_map<WorkerKey, prometheus::Gauge*, WorkerKeyHash> inflight;
  };

  struct IoMetricsCache {
    std::unordered_map<IoKey, prometheus::Histogram*, IoKeyHash> copy_latency;
    std::unordered_map<IoKey, prometheus::Counter*, IoKeyHash> transfer_bytes;
  };

  struct MetricCaches {
    GpuGaugeCache gpu;
    StatusCountersCache status;
    ModelMetricsCache model;
    WorkerMetricsCache worker;
    IoMetricsCache io;
  };

  struct MetricMutexes {
    mutable std::mutex sampling;
    std::mutex status;
    std::mutex model_metrics;
    std::mutex worker_metrics;
    std::mutex io_metrics;
  };

  RegistryState registry_state_;
  CounterMetrics counters_;
  GaugeMetrics gauges_;
  HistogramMetrics histograms_;
  FamilyMetrics families_;
  Providers providers_;
  MetricCaches caches_;
  MetricMutexes mutexes_;
};

auto init_metrics(int port) -> bool;
void shutdown_metrics();
void set_queue_size(std::size_t size);
void set_inflight_tasks(std::size_t size);
void set_starpu_worker_busy_ratio(double ratio);
void set_max_inflight_tasks(std::size_t max_tasks);
void set_queue_capacity(std::size_t capacity);
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
void set_queue_fill_ratio(std::size_t size, std::size_t capacity);
#endif  // SONAR_IGNORE_END
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
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
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
auto should_log_sampling_error_for_test(std::atomic<std::int64_t>& last_log)
    -> bool;
auto status_code_label_for_test(int code) -> std::string;
#endif  // SONAR_IGNORE_END

}  // namespace starpu_server::monitoring::detail
