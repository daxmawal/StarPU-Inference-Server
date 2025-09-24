#pragma once

#include <grpcpp/grpcpp.h>
#include <torch/torch.h>

#include <cstddef>
#include <memory>

#include "grpc_service.grpc.pb.h"
#include "starpu_task_worker/inference_queue.hpp"
#include "utils/logger.hpp"

namespace starpu_server {
namespace detail {
struct TimingInfo;
}

class InferenceServiceImpl final
    : public inference::GRPCInferenceService::Service {
 public:
  InferenceServiceImpl(
      InferenceQueue* queue,
      const std::vector<torch::Tensor>* reference_outputs,
      std::vector<at::ScalarType> expected_input_types,
      std::vector<std::vector<int64_t>> expected_input_dims,
      int max_batch_size);

  InferenceServiceImpl(
      InferenceQueue* queue,
      const std::vector<torch::Tensor>* reference_outputs,
      std::vector<at::ScalarType> expected_input_types);

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
      const LatencyBreakdown& breakdown) -> grpc::Status;

  auto submit_job_and_wait(
      const std::vector<torch::Tensor>& inputs,
      std::vector<torch::Tensor>& outputs, LatencyBreakdown& breakdown,
      detail::TimingInfo& timing_info) -> grpc::Status;

  auto validate_and_convert_inputs(
      const inference::ModelInferRequest* request,
      std::vector<torch::Tensor>& inputs) -> grpc::Status;

 private:
  InferenceQueue* queue_;
  const std::vector<torch::Tensor>* reference_outputs_;
  std::vector<at::ScalarType> expected_input_types_;
  std::vector<std::vector<int64_t>> expected_input_dims_;
  int max_batch_size_ = 0;
  std::atomic<int> next_job_id_{0};
};

void RunGrpcServer(
    InferenceQueue& queue, const std::vector<torch::Tensor>& reference_outputs,
    const std::vector<at::ScalarType>& expected_input_types,
    const std::vector<std::vector<int64_t>>& expected_input_dims,
    int max_batch_size, const std::string& address,
    std::size_t max_message_bytes, VerbosityLevel verbosity,
    std::unique_ptr<grpc::Server>& server);

void RunGrpcServer(
    InferenceQueue& queue, const std::vector<torch::Tensor>& reference_outputs,
    const std::vector<at::ScalarType>& expected_input_types,
    const std::string& address, std::size_t max_message_bytes,
    VerbosityLevel verbosity, std::unique_ptr<grpc::Server>& server);

void StopServer(std::unique_ptr<grpc::Server>& server);
}  // namespace starpu_server
