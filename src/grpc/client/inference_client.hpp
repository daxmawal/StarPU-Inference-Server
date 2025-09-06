#pragma once

#include <grpcpp/grpcpp.h>
#include <torch/script.h>

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "client_args.hpp"
#include "grpc_service.grpc.pb.h"
#include "utils/logger.hpp"
#include "utils/time_utils.hpp"

namespace starpu_server {
struct AsyncClientCall;

struct ModelId {
  std::string name;
  std::string version;
};

class InferenceClient {
 public:
  explicit InferenceClient(
      std::shared_ptr<grpc::Channel>& channel, VerbosityLevel verbosity);

  auto ServerIsLive() -> bool;
  auto ServerIsReady() -> bool;
  auto ModelIsReady(const ModelId& model) -> bool;
  void AsyncModelInfer(
      const std::vector<torch::Tensor>& tensors, const ClientConfig& cfg);
  void AsyncCompleteRpc();
  void Shutdown();

 private:
  std::unique_ptr<inference::GRPCInferenceService::Stub> stub_;
  grpc::CompletionQueue cq_;
  std::atomic<int> next_request_id_{0};
  VerbosityLevel verbosity_;
};
}  // namespace starpu_server
