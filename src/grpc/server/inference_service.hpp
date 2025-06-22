#pragma once

#include <grpcpp/grpcpp.h>
#include <torch/torch.h>

#include "grpc_service.grpc.pb.h"
#include "server/inference_queue.hpp"

class InferenceServiceImpl final
    : public inference::GRPCInferenceService::Service {
 public:
  InferenceServiceImpl(
      InferenceQueue* queue,
      const std::vector<torch::Tensor>* reference_outputs);

  auto ServerLive(
      grpc::ServerContext* context, const inference::ServerLiveRequest* request,
      inference::ServerLiveResponse* reply) -> grpc::Status override;

  auto ModelInfer(
      grpc::ServerContext* context, const inference::ModelInferRequest* request,
      inference::ModelInferResponse* reply) -> grpc::Status override;

 private:
  InferenceQueue* queue_;
  const std::vector<torch::Tensor>* reference_outputs_;
  std::atomic<unsigned int> next_job_id_{0};
};

void RunServer(
    InferenceQueue& queue, const std::vector<torch::Tensor>& reference_outputs,
    const std::string& address, int max_message_bytes);
