#pragma once

#include <grpcpp/grpcpp.h>
#include <torch/script.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "client_args.hpp"
#include "grpc_service.grpc.pb.h"
#include "utils/logger.hpp"

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
  struct LatencyRecords {
    std::vector<double> roundtrip_ms;
    std::vector<double> server_overall_ms;
    std::vector<double> server_preprocess_ms;
    std::vector<double> server_queue_ms;
    std::vector<double> server_batch_ms;
    std::vector<double> server_submit_ms;
    std::vector<double> server_scheduling_ms;
    std::vector<double> server_codelet_ms;
    std::vector<double> server_inference_ms;
    std::vector<double> server_callback_ms;
    std::vector<double> server_postprocess_ms;
    std::vector<double> server_job_total_ms;
    std::vector<double> request_latency_ms;
    std::vector<double> response_latency_ms;
    std::vector<double> client_overhead_ms;

    [[nodiscard]] auto empty() const -> bool { return roundtrip_ms.empty(); }
  };

  std::unique_ptr<inference::GRPCInferenceService::Stub> stub_;
  grpc::CompletionQueue cq_;
  std::atomic<int> next_request_id_{0};
  VerbosityLevel verbosity_;
  LatencyRecords latency_records_;
  std::optional<std::chrono::high_resolution_clock::time_point>
      first_request_time_;
  std::optional<std::chrono::high_resolution_clock::time_point>
      last_response_time_;
  std::optional<std::size_t> last_batch_size_;
  std::size_t total_inference_count_ = 0;

  void record_latency(
      double roundtrip_ms, double server_overall_ms, double preprocess_ms,
      double queue_ms, double batch_ms, double submit_ms, double scheduling_ms,
      double codelet_ms, double inference_ms, double callback_ms,
      double postprocess_ms, double job_total_ms, double request_latency_ms,
      double response_latency_ms, double client_overhead_ms);
  void log_latency_summary() const;
  static auto determine_inference_count(const ClientConfig& cfg) -> std::size_t;
};
}  // namespace starpu_server
