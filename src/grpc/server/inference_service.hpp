#pragma once

#include <grpcpp/grpcpp.h>
#include <torch/torch.h>

#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "grpc_service.grpc.pb.h"
#include "starpu_task_worker/inference_queue.hpp"
#include "utils/logger.hpp"

namespace starpu_server {
namespace detail {
struct TimingInfo;
}

class MetricsRegistry;

class InferenceServiceImpl final
    : public inference::GRPCInferenceService::Service {
 public:
  InferenceServiceImpl(
      InferenceQueue* queue,
      const std::vector<torch::Tensor>* reference_outputs,
      std::vector<at::ScalarType> expected_input_types,
      std::vector<std::vector<int64_t>> expected_input_dims, int max_batch_size,
      std::string default_model_name = {});

  InferenceServiceImpl(
      InferenceQueue* queue,
      const std::vector<torch::Tensor>* reference_outputs,
      std::vector<at::ScalarType> expected_input_types,
      std::string default_model_name = {});

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

  auto ModelInfer(
      grpc::ServerContext* context, const inference::ModelInferRequest* request,
      inference::ModelInferResponse* reply) -> grpc::Status override;

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

  static auto populate_response(
      const inference::ModelInferRequest* request,
      inference::ModelInferResponse* reply,
      const std::vector<torch::Tensor>& outputs, int64_t recv_ms,
      const LatencyBreakdown& breakdown,
      std::string_view model_name_override = {}) -> grpc::Status;

  using AsyncJobCallback = std::function<void(
      grpc::Status, std::vector<torch::Tensor>, LatencyBreakdown,
      detail::TimingInfo)>;

  auto submit_job_async(
      const std::vector<torch::Tensor>& inputs, AsyncJobCallback on_complete,
      std::vector<std::shared_ptr<const void>> input_lifetimes = {},
      std::chrono::high_resolution_clock::time_point receive_time =
          std::chrono::high_resolution_clock::now(),
      std::string model_name = {}) -> grpc::Status;

  auto submit_job_and_wait(
      const std::vector<torch::Tensor>& inputs,
      std::vector<torch::Tensor>& outputs, LatencyBreakdown& breakdown,
      detail::TimingInfo& timing_info,
      std::vector<std::shared_ptr<const void>> input_lifetimes = {})
      -> grpc::Status;

  void HandleModelInferAsync(
      grpc::ServerContext* context, const inference::ModelInferRequest* request,
      inference::ModelInferResponse* reply,
      std::function<void(grpc::Status)> on_done);

  auto validate_and_convert_inputs(
      const inference::ModelInferRequest* request,
      std::vector<torch::Tensor>& inputs,
      std::vector<std::shared_ptr<const void>>* input_lifetimes = nullptr)
      -> grpc::Status;

 private:
  class CallbackHandle {
   public:
    explicit CallbackHandle(std::function<void(grpc::Status)> callback);
    auto TryAcquire() -> bool;
    void Invoke(grpc::Status status);

   private:
    std::mutex mutex_;
    std::function<void(grpc::Status)> callback_;
    bool consumed_ = false;
  };

  struct AsyncInferCompletionContext {
    const inference::ModelInferRequest* request;
    inference::ModelInferResponse* reply;
    std::shared_ptr<CallbackHandle> callback_handle;
    std::shared_ptr<MetricsRegistry> metrics;
    std::chrono::high_resolution_clock::time_point recv_tp;
    int64_t recv_ms;
    std::string resolved_model_name;
  };

  static void handle_async_infer_completion(
      const AsyncInferCompletionContext& context,
      const grpc::Status& job_status, const std::vector<torch::Tensor>& outs,
      LatencyBreakdown breakdown, detail::TimingInfo timing_info);

  static auto build_latency_breakdown(
      const detail::TimingInfo& info, double latency_ms) -> LatencyBreakdown;

  [[nodiscard]] auto resolve_model_name(std::string model_name) const
      -> std::string;

  InferenceQueue* queue_;
  const std::vector<torch::Tensor>* reference_outputs_;
  std::vector<at::ScalarType> expected_input_types_;
  std::vector<std::vector<int64_t>> expected_input_dims_;
  int max_batch_size_ = 0;
  std::string default_model_name_;
  std::atomic<int> next_request_id_{0};
};

class AsyncServerContext {
 public:
  AsyncServerContext(
      inference::GRPCInferenceService::AsyncService& async_service,
      InferenceServiceImpl& impl);

  void configure(grpc::ServerBuilder& builder);
  void start();
  void shutdown();

  [[nodiscard]] auto started() const -> bool;
  [[nodiscard]] auto thread_count() const -> std::size_t;

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
};

void RunGrpcServer(
    InferenceQueue& queue, const std::vector<torch::Tensor>& reference_outputs,
    const std::vector<at::ScalarType>& expected_input_types,
    const std::vector<std::vector<int64_t>>& expected_input_dims,
    int max_batch_size, const GrpcServerOptions& options,
    std::unique_ptr<grpc::Server>& server);

void RunGrpcServer(
    InferenceQueue& queue, const std::vector<torch::Tensor>& reference_outputs,
    const std::vector<at::ScalarType>& expected_input_types,
    const GrpcServerOptions& options, std::unique_ptr<grpc::Server>& server);

void StopServer(grpc::Server* server);
}  // namespace starpu_server
