#pragma once

#include <grpcpp/grpcpp.h>
#include <torch/torch.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

#include "grpc_service.grpc.pb.h"
#include "starpu_task_worker/inference_queue.hpp"
#include "utils/logger.hpp"
#include "utils/monotonic_clock.hpp"

namespace starpu_server {
namespace detail {
struct TimingInfo;
}
namespace testing {
class InferenceServiceTestAccessor;
}

class MetricsRegistry;
class MetricsRecorder;
struct RuntimeObservability;

class InferenceServiceImpl final
    : public inference::GRPCInferenceService::Service {
 public:
  struct InputShapeConfig {
    std::vector<std::vector<int64_t>> expected_input_dims;
    int max_batch_size = 0;
  };

  struct ServiceOptions {
    std::string default_model_name;
    std::vector<std::string> expected_input_names;
    std::vector<std::string> expected_output_names;
    std::string server_name;
    std::string server_version;
  };

  InferenceServiceImpl(
      InferenceQueue* queue,
      const std::vector<torch::Tensor>* reference_outputs,
      std::vector<at::ScalarType> expected_input_types,
      InputShapeConfig input_shape_config, ServiceOptions service_options = {},
      std::shared_ptr<RuntimeObservability> observability = {});

  InferenceServiceImpl(
      InferenceQueue* queue,
      const std::vector<torch::Tensor>* reference_outputs,
      std::vector<at::ScalarType> expected_input_types,
      ServiceOptions service_options = {},
      std::shared_ptr<RuntimeObservability> observability = {});

  auto ServerLive(
      grpc::ServerContext* context, const inference::ServerLiveRequest* request,
      inference::ServerLiveResponse* reply) -> grpc::Status override;

  auto ServerReady(
      grpc::ServerContext* context,
      const inference::ServerReadyRequest* request,
      inference::ServerReadyResponse* reply) -> grpc::Status override;

  auto ModelReady(
      grpc::ServerContext* context, const inference::ModelReadyRequest* request,
      inference::ModelReadyResponse* reply) -> grpc::Status override;

  auto ServerMetadata(
      grpc::ServerContext* context,
      const inference::ServerMetadataRequest* request,
      inference::ServerMetadataResponse* reply) -> grpc::Status override;

  auto ModelMetadata(
      grpc::ServerContext* context,
      const inference::ModelMetadataRequest* request,
      inference::ModelMetadataResponse* reply) -> grpc::Status override;

  auto ModelConfig(
      grpc::ServerContext* context,
      const inference::ModelConfigRequest* request,
      inference::ModelConfigResponse* reply) -> grpc::Status override;

  auto ModelStatistics(
      grpc::ServerContext* context,
      const inference::ModelStatisticsRequest* request,
      inference::ModelStatisticsResponse* reply) -> grpc::Status override;

  auto ModelStreamInfer(
      grpc::ServerContext* context,
      grpc::ServerReaderWriter<
          inference::ModelStreamInferResponse, inference::ModelInferRequest>*
          stream) -> grpc::Status override;

  auto RepositoryIndex(
      grpc::ServerContext* context,
      const inference::RepositoryIndexRequest* request,
      inference::RepositoryIndexResponse* reply) -> grpc::Status override;

  auto RepositoryModelLoad(
      grpc::ServerContext* context,
      const inference::RepositoryModelLoadRequest* request,
      inference::RepositoryModelLoadResponse* reply) -> grpc::Status override;

  auto RepositoryModelUnload(
      grpc::ServerContext* context,
      const inference::RepositoryModelUnloadRequest* request,
      inference::RepositoryModelUnloadResponse* reply) -> grpc::Status override;

  auto SystemSharedMemoryStatus(
      grpc::ServerContext* context,
      const inference::SystemSharedMemoryStatusRequest* request,
      inference::SystemSharedMemoryStatusResponse* reply)
      -> grpc::Status override;

  auto SystemSharedMemoryRegister(
      grpc::ServerContext* context,
      const inference::SystemSharedMemoryRegisterRequest* request,
      inference::SystemSharedMemoryRegisterResponse* reply)
      -> grpc::Status override;

  auto SystemSharedMemoryUnregister(
      grpc::ServerContext* context,
      const inference::SystemSharedMemoryUnregisterRequest* request,
      inference::SystemSharedMemoryUnregisterResponse* reply)
      -> grpc::Status override;

  auto CudaSharedMemoryStatus(
      grpc::ServerContext* context,
      const inference::CudaSharedMemoryStatusRequest* request,
      inference::CudaSharedMemoryStatusResponse* reply)
      -> grpc::Status override;

  auto CudaSharedMemoryRegister(
      grpc::ServerContext* context,
      const inference::CudaSharedMemoryRegisterRequest* request,
      inference::CudaSharedMemoryRegisterResponse* reply)
      -> grpc::Status override;

  auto CudaSharedMemoryUnregister(
      grpc::ServerContext* context,
      const inference::CudaSharedMemoryUnregisterRequest* request,
      inference::CudaSharedMemoryUnregisterResponse* reply)
      -> grpc::Status override;

  auto TraceSetting(
      grpc::ServerContext* context,
      const inference::TraceSettingRequest* request,
      inference::TraceSettingResponse* reply) -> grpc::Status override;

  auto LogSettings(
      grpc::ServerContext* context,
      const inference::LogSettingsRequest* request,
      inference::LogSettingsResponse* reply) -> grpc::Status override;

  // Validation / convert block.
  struct LatencyBreakdown {
    double preprocess_ms = 0.0;
    double queue_ms = 0.0;
    double batch_ms = 0.0;
    double submit_ms = 0.0;
    double scheduling_ms = 0.0;
    double codelet_ms = 0.0;
    double inference_ms = 0.0;
    double callback_ms = 0.0;
    double postprocess_ms = 0.0;
    double total_ms = 0.0;
    double overall_ms = 0.0;
  };

  struct PopulateResponseOptions {
    std::string_view model_name_override;
    bool set_prepost_overall = true;
    std::span<const std::string> output_names;
  };

  // Async coordination block.
  struct AsyncFailureInfo {
    std::string stage;
    std::string reason;
    bool metrics_reported = false;
  };

  static auto populate_response(
      const inference::ModelInferRequest* request,
      inference::ModelInferResponse* reply,
      const std::vector<torch::Tensor>& outputs, int64_t recv_ms,
      const LatencyBreakdown& breakdown) -> grpc::Status;
  static auto populate_response(
      const inference::ModelInferRequest* request,
      inference::ModelInferResponse* reply,
      const std::vector<torch::Tensor>& outputs, int64_t recv_ms,
      const LatencyBreakdown& breakdown,
      PopulateResponseOptions options) -> grpc::Status;

  using AsyncJobCallback = std::function<void(
      grpc::Status, std::vector<torch::Tensor>, LatencyBreakdown,
      detail::TimingInfo, std::optional<AsyncFailureInfo>)>;

  auto submit_job_async(
      const std::vector<torch::Tensor>& inputs, AsyncJobCallback on_complete,
      std::vector<std::shared_ptr<const void>> input_lifetimes = {},
      std::shared_ptr<std::atomic<bool>> cancel_flag = {},
      MonotonicClock::time_point receive_time = MonotonicClock::now(),
      std::string model_name = {},
      std::optional<AsyncFailureInfo>* submit_failure_info = nullptr)
      -> grpc::Status;

  void HandleModelInferAsync(
      grpc::ServerContext* context, const inference::ModelInferRequest* request,
      inference::ModelInferResponse* reply,
      std::function<void(grpc::Status)> on_done,
      std::shared_ptr<void> call_guard = {});

  auto validate_and_convert_inputs(
      const inference::ModelInferRequest* request,
      std::vector<torch::Tensor>& inputs,
      std::vector<std::shared_ptr<const void>>* input_lifetimes = nullptr)
      -> grpc::Status;

 private:
  friend class testing::InferenceServiceTestAccessor;

  // Async coordination internals.
  class CallbackHandle {
   public:
    explicit CallbackHandle(std::function<void(grpc::Status)> callback);
    auto TryAcquire() -> bool;
    auto Invoke(grpc::Status status) -> bool;

   private:
    std::mutex mutex_;
    std::function<void(grpc::Status)> callback_;
    bool consumed_ = false;
  };

  struct AsyncInferCompletionContext {
    const inference::ModelInferRequest* request;
    inference::ModelInferResponse* reply;
    std::shared_ptr<CallbackHandle> callback_handle;
    std::shared_ptr<MetricsRecorder> metrics;
    MonotonicClock::time_point recv_tp;
    int64_t recv_ms;
    std::string resolved_model_name;
    InferenceServiceImpl* impl = nullptr;
    const std::vector<std::string>* output_names = nullptr;
    std::shared_ptr<std::atomic<bool>> cancel_flag;
    std::shared_ptr<std::atomic<bool>> terminal_flag;
    std::optional<AsyncFailureInfo> failure_info;
  };

  struct AsyncCancellationContext {
    std::shared_ptr<std::atomic<bool>> cancel_flag;
    std::shared_ptr<std::atomic<bool>> terminal_flag;
    std::shared_ptr<CallbackHandle> callback_handle;
    std::string_view resolved_model_name;
    InferenceServiceImpl* service = nullptr;
    const inference::ModelInferRequest* request = nullptr;
    MonotonicClock::time_point recv_tp;
  };

  struct AsyncTerminalState {
    std::shared_ptr<std::atomic<bool>> cancel_flag;
    std::shared_ptr<std::atomic<bool>> terminal_flag;
  };

  struct AsyncInternalErrorDetails {
    std::string_view stage;
    std::string_view reason;
    std::string_view log_context;
  };

  struct AsyncTerminalCompletionDetails {
    InferenceServiceImpl* service = nullptr;
    std::string_view resolved_model_name;
    const inference::ModelInferRequest* request = nullptr;
    MonotonicClock::time_point recv_tp;
    std::string_view stage;
    std::string_view reason;
    const std::optional<AsyncFailureInfo>* failure_info = nullptr;
    std::string_view log_context;
    bool check_cancel_flag = true;
    bool record_status_metric = true;
    bool record_before_callback = true;
    bool require_callback_invoked_for_record = false;
  };

  class AsyncOps {
   public:
    static auto try_mark_terminal(
        const std::shared_ptr<std::atomic<bool>>& terminal_flag) -> bool;
    static auto enter_async_terminal_once(
        const AsyncTerminalState& terminal_state,
        bool check_cancel_flag = true) -> bool;
    static auto invoke_async_callback(
        const std::shared_ptr<CallbackHandle>& callback_handle,
        const grpc::Status& status) -> bool;
    static void record_async_terminal_failure(
        const AsyncTerminalCompletionDetails& details,
        const grpc::Status& status);
    static auto complete_async_terminal_with_status(
        const AsyncTerminalState& terminal_state,
        const std::shared_ptr<CallbackHandle>& callback_handle,
        const grpc::Status& status,
        const AsyncTerminalCompletionDetails& details) -> bool;
    static auto is_async_cancelled(const AsyncInferCompletionContext& context)
        -> bool;
    static auto prepare_async_completion(
        const AsyncInferCompletionContext& context,
        const std::shared_ptr<CallbackHandle>& callback_handle) -> bool;
    static auto handle_job_failure(
        const AsyncInferCompletionContext& context,
        const grpc::Status& job_status,
        const std::shared_ptr<CallbackHandle>& callback_handle) -> bool;
    static void finalize_successful_completion(
        const AsyncInferCompletionContext& context,
        const std::vector<torch::Tensor>& outs, LatencyBreakdown breakdown,
        const detail::TimingInfo& timing_info);

    static void handle_async_infer_completion(
        const AsyncInferCompletionContext& context,
        const grpc::Status& job_status, const std::vector<torch::Tensor>& outs,
        LatencyBreakdown breakdown, detail::TimingInfo timing_info);
    static void handle_submit_job_completion(
        const AsyncInferCompletionContext& context,
        const grpc::Status& job_status, const std::vector<torch::Tensor>& outs,
        LatencyBreakdown breakdown, detail::TimingInfo timing_info,
        std::optional<AsyncFailureInfo> failure_info);

    static auto build_latency_breakdown(
        const detail::TimingInfo& info, double latency_ms) -> LatencyBreakdown;

    static auto setup_async_cancellation(
        grpc::ServerContext* context, std::shared_ptr<void>& call_guard,
        const AsyncCancellationContext& cancellation_context) -> bool;
    static void handle_async_cancellation(
        const grpc::ServerContext* context,
        const AsyncTerminalState& terminal_state,
        const std::shared_ptr<CallbackHandle>& callback_handle,
        InferenceServiceImpl* service, std::string_view resolved_model_name,
        const inference::ModelInferRequest* request,
        MonotonicClock::time_point recv_tp);

    static auto handle_input_validation_failure(
        const grpc::Status& status, const AsyncTerminalState& terminal_state,
        const std::shared_ptr<CallbackHandle>& callback_handle,
        InferenceServiceImpl* service, std::string_view resolved_model_name,
        const inference::ModelInferRequest* request,
        MonotonicClock::time_point recv_tp) -> bool;

    static auto handle_submit_failure(
        const grpc::Status& status, const AsyncTerminalState& terminal_state,
        const std::shared_ptr<CallbackHandle>& callback_handle,
        const AsyncTerminalCompletionDetails& details) -> bool;

    static void handle_async_internal_error(
        const AsyncTerminalState& terminal_state,
        const std::shared_ptr<CallbackHandle>& callback_handle,
        InferenceServiceImpl* service, std::string_view resolved_model_name,
        const inference::ModelInferRequest* request,
        MonotonicClock::time_point recv_tp,
        const AsyncInternalErrorDetails& details);

    static void notify_cancel_flag_created(
        const std::shared_ptr<std::atomic<bool>>& cancel_flag);

    static void notify_submit_job_async_done(
        const std::shared_ptr<std::atomic<bool>>& cancel_flag,
        const grpc::Status& status);
  };

  void record_success(
      const inference::ModelInferRequest* request,
      const LatencyBreakdown& breakdown, MonotonicClock::time_point recv_tp,
      std::string_view resolved_model_name);

  void record_failure(
      const inference::ModelInferRequest* request,
      MonotonicClock::time_point recv_tp, std::string_view resolved_model_name);

 public:
  void validate_schema_or_throw() const;
  [[nodiscard]] auto metrics_recorder() const
      -> std::shared_ptr<MetricsRecorder>;
  [[nodiscard]] auto observability() const
      -> const std::shared_ptr<RuntimeObservability>&
  {
    return observability_;
  }
  void record_terminal_metrics(
      std::string_view model_name, const grpc::Status& status,
      std::string_view default_stage, std::string_view default_reason = {},
      const std::optional<AsyncFailureInfo>& failure_info = std::nullopt,
      bool record_status_metric = true) const;

 private:
  [[nodiscard]] auto resolve_model_name(std::string model_name) const
      -> std::string;
  auto next_request_id() -> int;

  // Shared service state.
  InferenceQueue* queue_;
  const std::vector<torch::Tensor>* reference_outputs_;
  std::vector<at::ScalarType> expected_input_types_;
  std::vector<std::vector<int64_t>> expected_input_dims_;
  std::vector<std::string> expected_input_names_;
  std::vector<std::string> expected_output_names_;
  int max_batch_size_ = 0;
  std::string default_model_name_;
  std::string server_name_;
  std::string server_version_;
  std::shared_ptr<RuntimeObservability> observability_;
  struct StatisticDurationState {
    uint64_t count = 0;
    uint64_t ns = 0;
  };
  struct InferStatisticsState {
    StatisticDurationState success{};
    StatisticDurationState fail{};
    StatisticDurationState queue{};
    StatisticDurationState compute_input{};
    StatisticDurationState compute_infer{};
    StatisticDurationState compute_output{};
  };
  struct ModelStatsState {
    uint64_t last_inference_ms = 0;
    uint64_t inference_count = 0;
    uint64_t execution_count = 0;
    InferStatisticsState inference_stats{};
  };
  struct ModelStatsKey {
    std::string name;
    std::string version;

    auto operator==(const ModelStatsKey& other) const -> bool = default;
  };
  struct ModelStatsKeyHash {
    auto operator()(const ModelStatsKey& key) const noexcept -> std::size_t
    {
      constexpr std::size_t kHashCombineMagic = 0x9e3779b9U;
      constexpr unsigned kHashCombineLeftShift = 6U;
      constexpr unsigned kHashCombineRightShift = 2U;
      const std::size_t name_hash = std::hash<std::string>{}(key.name);
      const std::size_t version_hash = std::hash<std::string>{}(key.version);
      return name_hash ^ (version_hash + kHashCombineMagic +
                          (name_hash << kHashCombineLeftShift) +
                          (name_hash >> kHashCombineRightShift));
    }
  };
  mutable std::mutex model_stats_mutex_;
  std::unordered_map<ModelStatsKey, ModelStatsState, ModelStatsKeyHash>
      model_stats_;
  std::atomic<int> next_request_id_{0};
};

// Async server block.
class AsyncServerContext {
 public:
  AsyncServerContext(
      inference::GRPCInferenceService::AsyncService& async_service,
      InferenceServiceImpl& impl);

  void configure(grpc::ServerBuilder& builder);
  void start();
  void shutdown();
  [[nodiscard]] auto started() const -> bool { return started_; }
  [[nodiscard]] auto thread_count() const -> std::size_t
  {
    return threads_.size();
  }

 private:
  void poll_events();

  inference::GRPCInferenceService::AsyncService* async_service_;
  InferenceServiceImpl* impl_;
  std::unique_ptr<grpc::ServerCompletionQueue> completion_queue_;
  std::vector<std::jthread> threads_;
  bool started_ = false;
};

inline constexpr std::size_t kDefaultGrpcThreads = 4;
inline constexpr std::size_t kMinGrpcThreads = 2;
inline constexpr std::size_t kMaxGrpcThreads = 8;

auto compute_thread_count_from(unsigned concurrency) -> std::size_t;

struct GrpcServerOptions {
  std::string address;
  std::size_t max_message_bytes;
  VerbosityLevel verbosity;
  std::string default_model_name;
  std::string server_name;
  std::string server_version;
};

struct GrpcModelSpec {
  std::span<const at::ScalarType> expected_input_types;
  std::span<const std::vector<int64_t>> expected_input_dims;
  std::span<const std::string> expected_input_names;
  std::span<const std::string> expected_output_names;
  int max_batch_size = 0;
};

struct GrpcModelNamesSpec {
  std::span<const std::string> expected_input_names;
  std::span<const std::string> expected_output_names;
};

struct GrpcServerLifecycleHooks {
  std::function<void(grpc::Server* server)> on_started;
  std::function<void()> on_stopped;
};

void RunGrpcServer(
    InferenceQueue& queue, const std::vector<torch::Tensor>& reference_outputs,
    const GrpcModelSpec& model_spec, const GrpcServerOptions& options,
    std::unique_ptr<grpc::Server>& server,
    const GrpcServerLifecycleHooks& hooks = {},
    std::shared_ptr<RuntimeObservability> observability = {});

void RunGrpcServer(
    InferenceQueue& queue, const std::vector<torch::Tensor>& reference_outputs,
    const std::vector<at::ScalarType>& expected_input_types,
    GrpcModelNamesSpec model_names, const GrpcServerOptions& options,
    std::unique_ptr<grpc::Server>& server,
    const GrpcServerLifecycleHooks& hooks = {},
    std::shared_ptr<RuntimeObservability> observability = {});

void StopServer(grpc::Server* server);
}  // namespace starpu_server
