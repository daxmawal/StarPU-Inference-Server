#pragma once

#include <grpcpp/grpcpp.h>
#include <torch/torch.h>

#include "grpc_service.grpc.pb.h"
#include "starpu_task_worker/inference_queue.hpp"

namespace starpu_server {
class InferenceServiceImpl final
    : public inference::GRPCInferenceService::Service {
 public:
  InferenceServiceImpl(
      InferenceQueue* queue,
      const std::vector<torch::Tensor>* reference_outputs);

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

  static void populate_response(
      const inference::ModelInferRequest* request,
      inference::ModelInferResponse* reply,
      const std::vector<torch::Tensor>& outputs, int64_t recv_ms,
      int64_t send_ms);

  auto submit_job_and_wait(
      const std::vector<torch::Tensor>& inputs,
      std::vector<torch::Tensor>& outputs) -> grpc::Status;

  static auto validate_and_convert_inputs(
      const inference::ModelInferRequest* request,
      std::vector<torch::Tensor>& inputs) -> grpc::Status;

 private:
  InferenceQueue* queue_;
  const std::vector<torch::Tensor>* reference_outputs_;
  std::atomic<int> next_job_id_{0};
};

void RunGrpcServer(
    InferenceQueue& queue, const std::vector<torch::Tensor>& reference_outputs,
    const std::string& address, int max_message_bytes);

void StopServer();
}  // namespace starpu_server