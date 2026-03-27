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
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace prometheus {
class Exposer;
class Collectable;
template <typename T>
class Family;
}  // namespace prometheus

namespace starpu_server {
namespace testing {
class MetricsRegistryTestAccessor;
}

class MetricsRecorder;

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
    struct CongestionGaugeMetrics {
      prometheus::Gauge* flag{nullptr};
      prometheus::Gauge* score{nullptr};
      prometheus::Gauge* lambda_rps{nullptr};
      prometheus::Gauge* mu_rps{nullptr};
      prometheus::Gauge* rho_ewma{nullptr};
      prometheus::Gauge* queue_fill_ewma{nullptr};
      prometheus::Gauge* queue_growth_rate{nullptr};
      prometheus::Gauge* queue_p95_ms{nullptr};
      prometheus::Gauge* queue_p99_ms{nullptr};
      prometheus::Gauge* e2e_p95_ms{nullptr};
      prometheus::Gauge* e2e_p99_ms{nullptr};
      prometheus::Gauge* rejection_rps{nullptr};
    };

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
    CongestionGaugeMetrics congestion{};
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
    prometheus::Family<prometheus::Gauge>* gpu_model_replication_policy_info{
        nullptr};
    prometheus::Family<prometheus::Gauge>* gpu_model_replicas_total{nullptr};
    prometheus::Family<prometheus::Gauge>* starpu_cuda_worker_info{nullptr};
    prometheus::Family<prometheus::Histogram>* io_copy_latency{nullptr};
    prometheus::Family<prometheus::Counter>* transfer_bytes{nullptr};
    prometheus::Family<prometheus::Gauge>* gpu_utilization{nullptr};
    prometheus::Family<prometheus::Gauge>* gpu_memory_used_bytes{nullptr};
    prometheus::Family<prometheus::Gauge>* gpu_memory_total_bytes{nullptr};
    prometheus::Family<prometheus::Gauge>* gpu_temperature{nullptr};
    prometheus::Family<prometheus::Gauge>* gpu_power{nullptr};
    prometheus::Family<prometheus::Counter>* requests_by_status{nullptr};
    prometheus::Family<prometheus::Counter>* requests_received{nullptr};
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
  void increment_received_counter(std::string_view model_label);
  void increment_failure_counter(
      FailureStageLabel stage_label, FailureReasonLabel reason_label,
      ModelLabel model_label, std::size_t count);
  void increment_model_load_failure_counter(std::string_view model_label);
  void set_model_loaded_flag(
      ModelLabel model_label, DeviceLabel device_label, bool loaded);
  void set_gpu_model_replication_policy_flag(
      ModelLabel model_label, std::string_view policy_label);
  void set_gpu_model_replicas_total_gauge(
      ModelLabel model_label, std::size_t replicas);
  void set_starpu_cuda_worker_info_gauge(
      int worker_id, int device_id, bool active);
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

 private:
  friend class testing::MetricsRegistryTestAccessor;

  class Sampler;

  void initialize(
      [[maybe_unused]] int port, bool start_sampler_thread,
      std::unique_ptr<ExposerHandle> exposer_handle);

  static constexpr std::size_t kHashCombineMagic = 0x9e3779b97f4a7c15ULL;
  static constexpr std::size_t kHashCombineShiftLeft = 6U;
  static constexpr std::size_t kHashCombineShiftRight = 2U;

  static auto HashCombine(std::size_t seed, std::size_t value) noexcept
      -> std::size_t
  {
    return seed ^ (value + kHashCombineMagic + (seed << kHashCombineShiftLeft) +
                   (seed >> kHashCombineShiftRight));
  }

  template <typename... Values>
  static auto HashMany(const Values&... values) noexcept -> std::size_t
  {
    std::size_t seed = 0;
    ((seed = HashCombine(seed, std::hash<std::decay_t<Values>>{}(values))),
     ...);
    return seed;
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
      return MetricsRegistry::HashMany(key.code, key.model, key.overflow);
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
      return MetricsRegistry::HashMany(
          key.stage, key.reason, key.model, key.overflow);
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
      return MetricsRegistry::HashMany(key.model, key.overflow);
    }
  };

  struct ModelPolicyKey {
    std::string model;
    std::string policy;
    bool overflow{false};

    static auto Overflow() -> ModelPolicyKey
    {
      ModelPolicyKey key;
      key.overflow = true;
      return key;
    }

    auto operator==(const ModelPolicyKey& other) const noexcept -> bool
    {
      return overflow == other.overflow && model == other.model &&
             policy == other.policy;
    }
  };

  struct ModelPolicyKeyHash {
    auto operator()(const ModelPolicyKey& key) const noexcept -> std::size_t
    {
      return MetricsRegistry::HashMany(key.model, key.policy, key.overflow);
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
      return MetricsRegistry::HashMany(key.model, key.device, key.overflow);
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
      return MetricsRegistry::HashMany(
          key.worker_id, key.device_id, key.worker_type, key.overflow);
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
      return MetricsRegistry::HashMany(
          key.direction, key.worker_id, key.device_id, key.worker_type,
          key.overflow);
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
        requests_received;
    std::unordered_map<ModelKey, prometheus::Counter*, ModelKeyHash>
        inference_completed;
    std::unordered_map<FailureKey, prometheus::Counter*, FailureKeyHash>
        inference_failures;
    std::unordered_map<ModelKey, prometheus::Counter*, ModelKeyHash>
        model_load_failures;
    std::unordered_map<ModelDeviceKey, prometheus::Gauge*, ModelDeviceKeyHash>
        models_loaded;
    std::unordered_map<ModelPolicyKey, prometheus::Gauge*, ModelPolicyKeyHash>
        gpu_replication_policy;
    std::unordered_map<ModelKey, prometheus::Gauge*, ModelKeyHash>
        gpu_replicas_total;
  };

  struct WorkerMetricsCache {
    std::unordered_map<WorkerKey, prometheus::Histogram*, WorkerKeyHash>
        compute_latency;
    std::unordered_map<WorkerKey, prometheus::Histogram*, WorkerKeyHash>
        task_runtime;
    std::unordered_map<WorkerKey, prometheus::Gauge*, WorkerKeyHash> inflight;
    std::unordered_map<WorkerKey, prometheus::Gauge*, WorkerKeyHash>
        cuda_worker_info;
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
  std::unique_ptr<Sampler> sampler_;
};

struct LatencyBreakdownMetrics {
  double queue_ms{0.0};
  double batch_ms{0.0};
  double submit_ms{0.0};
  double scheduling_ms{0.0};
  double codelet_ms{0.0};
  double inference_ms{0.0};
  double callback_ms{0.0};
  double preprocess_ms{0.0};
  double postprocess_ms{0.0};
};

namespace metrics_recorder_detail {

class Access {
 protected:
  Access() = default;
  explicit Access(std::shared_ptr<MetricsRegistry> registry)
      : registry_(std::move(registry))
  {
  }

  [[nodiscard]] auto registry_ptr() const
      -> const std::shared_ptr<MetricsRegistry>&
  {
    return registry_;
  }

 private:
  std::shared_ptr<MetricsRegistry> registry_;
};

class Core : protected Access {
 protected:
  using Access::Access;
  using Access::registry_ptr;

 public:
  [[nodiscard]] auto enabled() const -> bool
  {
    return registry_ptr() != nullptr;
  }
  [[nodiscard]] auto registry() const -> std::shared_ptr<MetricsRegistry>
  {
    return registry_ptr();
  }

  void increment_requests_total() const;
  void observe_inference_latency(double latency_ms) const;
};

class QueueRuntime : protected Access {
 protected:
  using Access::Access;
  using Access::registry_ptr;

 public:
  void set_queue_size(std::size_t size) const;
  void set_inflight_tasks(std::size_t size) const;
  void set_starpu_worker_busy_ratio(double ratio) const;
  void set_max_inflight_tasks(std::size_t max_tasks) const;
  void set_queue_capacity(std::size_t capacity) const;
  void set_server_health(bool ready) const;
  void set_starpu_prepared_queue_depth(std::size_t depth) const;
  void set_batch_pending_jobs(std::size_t pending) const;
};

class Congestion : protected Access {
 protected:
  using Access::Access;
  using Access::registry_ptr;

 public:
  void set_congestion_flag(bool congested) const;
  void set_congestion_score(double score) const;
  void set_congestion_arrival_rate(double rps) const;
  void set_congestion_completion_rate(double rps) const;
  void set_congestion_rejection_rate(double rps) const;
  void set_congestion_rho(double rho) const;
  void set_congestion_fill_ewma(double fill) const;
  void set_congestion_queue_growth_rate(double rate) const;
  void set_congestion_queue_latency_p95(double latency_ms) const;
  void set_congestion_queue_latency_p99(double latency_ms) const;
  void set_congestion_e2e_latency_p95(double latency_ms) const;
  void set_congestion_e2e_latency_p99(double latency_ms) const;
};

class Requests : protected Access {
 protected:
  using Access::Access;
  using Access::registry_ptr;

 public:
  void increment_request_status(
      int status_code, std::string_view model_name) const;
  void increment_requests_received(std::string_view model_name) const;
  void increment_inference_completed(
      std::string_view model_name, std::size_t logical_jobs) const;
  void increment_inference_failure(
      std::string_view stage, std::string_view reason,
      std::string_view model_name, std::size_t count = 1) const;
  void observe_batch_size(std::size_t batch_size) const;
  void observe_logical_batch_size(std::size_t logical_jobs) const;
  void observe_batch_efficiency(double ratio) const;
  void observe_latency_breakdown(
      const LatencyBreakdownMetrics& breakdown) const;
  void observe_starpu_task_runtime(double runtime_ms) const;
  void observe_model_load_duration(double duration_ms) const;
  void increment_rejected_requests() const;
};

class Models : protected Access {
 protected:
  using Access::Access;
  using Access::registry_ptr;

 public:
  void set_model_loaded(
      std::string_view model_name, std::string_view device_label,
      bool loaded) const;
  void set_gpu_model_replication_policy(
      std::string_view model_name, std::string_view policy_label) const;
  void set_gpu_model_replicas_total(
      std::string_view model_name, std::size_t replicas) const;
  void increment_model_load_failure(std::string_view model_name) const;
};

class Workers : protected Access {
 protected:
  using Access::Access;
  using Access::registry_ptr;

 public:
  void set_starpu_cuda_worker_info(
      int worker_id, int device_id, bool active) const;
  void observe_compute_latency_by_worker(
      int worker_id, int device_id, std::string_view worker_type,
      double latency_ms) const;
  void observe_task_runtime_by_worker(
      int worker_id, int device_id, std::string_view worker_type,
      double latency_ms) const;
  void set_worker_inflight_gauge(
      int worker_id, int device_id, std::string_view worker_type,
      std::size_t value) const;
  void observe_io_copy_latency(
      std::string_view direction, int worker_id, int device_id,
      std::string_view worker_type, double duration_ms) const;
  void increment_transfer_bytes(
      std::string_view direction, int worker_id, int device_id,
      std::string_view worker_type, std::size_t bytes) const;
};

}  // namespace metrics_recorder_detail

class MetricsRecorder : public metrics_recorder_detail::Core,
                        public metrics_recorder_detail::QueueRuntime,
                        public metrics_recorder_detail::Congestion,
                        public metrics_recorder_detail::Requests,
                        public metrics_recorder_detail::Models,
                        public metrics_recorder_detail::Workers {
 public:
  MetricsRecorder() = default;
  explicit MetricsRecorder(std::shared_ptr<MetricsRegistry> registry)
      : Core(registry), QueueRuntime(registry), Congestion(registry),
        Requests(registry), Models(registry), Workers(std::move(registry))
  {
  }
};

auto create_metrics_recorder(int port) -> std::shared_ptr<MetricsRecorder>;
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
void set_congestion_flag(bool congested);
void set_congestion_score(double score);
void set_congestion_arrival_rate(double rps);
void set_congestion_completion_rate(double rps);
void set_congestion_rejection_rate(double rps);
void set_congestion_rho(double rho);
void set_congestion_fill_ewma(double fill);
void set_congestion_queue_growth_rate(double rate);
void set_congestion_queue_latency_p95(double latency_ms);
void set_congestion_queue_latency_p99(double latency_ms);
void set_congestion_e2e_latency_p95(double latency_ms);
void set_congestion_e2e_latency_p99(double latency_ms);
void increment_request_status(int status_code, std::string_view model_name);
void increment_requests_received(std::string_view model_name);
void increment_inference_completed(
    std::string_view model_name, std::size_t logical_jobs);
void increment_inference_failure(
    std::string_view stage, std::string_view reason,
    std::string_view model_name, std::size_t count = 1);
void observe_batch_size(std::size_t batch_size);
void observe_logical_batch_size(std::size_t logical_jobs);
void observe_batch_efficiency(double ratio);
void observe_inference_latency(double latency_ms);
void observe_latency_breakdown(const LatencyBreakdownMetrics& breakdown);
void observe_starpu_task_runtime(double runtime_ms);
void observe_model_load_duration(double duration_ms);
void set_model_loaded(
    std::string_view model_name, std::string_view device_label, bool loaded);
void set_gpu_model_replication_policy(
    std::string_view model_name, std::string_view policy_label);
void set_gpu_model_replicas_total(
    std::string_view model_name, std::size_t replicas);
void set_starpu_cuda_worker_info(int worker_id, int device_id, bool active);
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

}  // namespace starpu_server::monitoring::detail
