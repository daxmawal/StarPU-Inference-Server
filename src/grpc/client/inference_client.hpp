#pragma once

#include <grpcpp/grpcpp.h>
#include <torch/script.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "client_args.hpp"
#include "grpc_service.grpc.pb.h"
#include "utils/logger.hpp"

namespace starpu_server {
struct AsyncClientCall;
struct InferenceClientTestAccess;

struct ModelId {
  std::string name;
  std::string version;
};

class InferenceClient {
 public:
  using OutputSummary = std::vector<std::vector<double>>;

  struct LatencySummary {
    double mean_ms;
    double p50_ms;
    double p85_ms;
    double p95_ms;
    double p100_ms;
  };

  struct Summary {
    struct ServerLatencySummary {
      std::optional<LatencySummary> overall;
      std::optional<LatencySummary> preprocess;
      std::optional<LatencySummary> queue;
      std::optional<LatencySummary> batching;
      std::optional<LatencySummary> submit;
      std::optional<LatencySummary> scheduling;
      std::optional<LatencySummary> codelet;
      std::optional<LatencySummary> inference;
      std::optional<LatencySummary> callback;
      std::optional<LatencySummary> postprocess;
      std::optional<LatencySummary> job_total;
    };

    std::size_t requests_sent = 0;
    std::size_t requests_handled = 0;
    std::size_t requests_ok = 0;
    std::size_t requests_rejected = 0;
    std::size_t inference_count = 0;
    std::size_t response_count = 0;
    std::optional<double> elapsed_seconds;
    std::optional<double> throughput_rps;
    std::optional<LatencySummary> roundtrip_latency;
    ServerLatencySummary server_latency;
    std::optional<LatencySummary> request_latency;
    std::optional<LatencySummary> response_latency;
    std::optional<LatencySummary> client_overhead_latency;
  };

  explicit InferenceClient(
      std::shared_ptr<grpc::Channel>& channel, VerbosityLevel verbosity);

  auto ServerIsLive() -> bool;
  auto ServerIsReady() -> bool;
  auto ModelIsReady(const ModelId& model) -> bool;
  void AsyncModelInfer(
      const std::vector<torch::Tensor>& tensors, const ClientConfig& cfg,
      std::optional<OutputSummary> expected_outputs = std::nullopt);
  void AsyncCompleteRpc();
  void Shutdown();
  [[nodiscard]] auto summary() const -> Summary;
  auto write_summary_json(const std::filesystem::path& path) const -> bool;

 private:
  friend struct InferenceClientTestAccess;

  struct LatencySample {
    double roundtrip_ms;
    double server_overall_ms;
    double server_preprocess_ms;
    double server_queue_ms;
    double server_batch_ms;
    double server_submit_ms;
    double server_scheduling_ms;
    double server_codelet_ms;
    double server_inference_ms;
    double server_callback_ms;
    double server_postprocess_ms;
    double server_job_total_ms;
    double request_latency_ms;
    double response_latency_ms;
    double client_overhead_ms;
  };

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
  std::optional<std::chrono::system_clock::time_point> first_request_time_;
  std::optional<std::chrono::system_clock::time_point> last_response_time_;
  std::atomic<std::size_t> total_inference_count_{0};
  std::atomic<std::size_t> total_requests_sent_{0};
  std::atomic<std::size_t> success_requests_{0};
  std::atomic<std::size_t> rejected_requests_{0};

  void record_latency(const LatencySample& sample);
  [[nodiscard]] static auto summarize_latencies(
      const std::vector<double>& values) -> std::optional<LatencySummary>;
  void log_latency_summary() const;
  void log_request_totals() const;
  static auto determine_inference_count(const ClientConfig& cfg) -> std::size_t;
  void validate_server_response(const AsyncClientCall& call) const;
};
}  // namespace starpu_server
