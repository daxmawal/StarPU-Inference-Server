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

// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
struct InferenceClientTestAccess;
#endif  // SONAR_IGNORE_END
// GCOVR_EXCL_STOP

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
    std::size_t requests_sent = 0;
    std::size_t requests_handled = 0;
    std::size_t requests_ok = 0;
    std::size_t requests_rejected = 0;
    std::size_t inference_count = 0;
    std::size_t response_count = 0;
    std::optional<double> elapsed_seconds;
    std::optional<double> throughput_rps;
    std::optional<LatencySummary> roundtrip_latency;
    std::optional<LatencySummary> server_overall_latency;
    std::optional<LatencySummary> server_preprocess_latency;
    std::optional<LatencySummary> server_queue_latency;
    std::optional<LatencySummary> server_batching_latency;
    std::optional<LatencySummary> server_submit_latency;
    std::optional<LatencySummary> server_scheduling_latency;
    std::optional<LatencySummary> server_codelet_latency;
    std::optional<LatencySummary> server_inference_latency;
    std::optional<LatencySummary> server_callback_latency;
    std::optional<LatencySummary> server_postprocess_latency;
    std::optional<LatencySummary> server_job_total_latency;
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
// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  friend struct InferenceClientTestAccess;
#endif  // SONAR_IGNORE_END
  // GCOVR_EXCL_STOP

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
  std::size_t total_inference_count_ = 0;
  std::size_t total_requests_sent_ = 0;
  std::size_t success_requests_ = 0;
  std::size_t rejected_requests_ = 0;

  void record_latency(const LatencySample& sample);
  [[nodiscard]] static auto summarize_latencies(
      const std::vector<double>& values) -> std::optional<LatencySummary>;
  void log_latency_summary() const;
  void log_request_totals() const;
  static auto determine_inference_count(const ClientConfig& cfg) -> std::size_t;
  void validate_server_response(const AsyncClientCall& call) const;
};

// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
struct InferenceClientTestAccess {
  using LatencySample = InferenceClient::LatencySample;
  using LatencyRecords = InferenceClient::LatencyRecords;
  using TimePoint = std::chrono::system_clock::time_point;

  static auto determine_inference_count(const ClientConfig& cfg) -> std::size_t
  {
    return InferenceClient::determine_inference_count(cfg);
  }

  static void set_verbosity(InferenceClient& client, VerbosityLevel level)
  {
    client.verbosity_ = level;
  }

  static auto latency_records(InferenceClient& client) -> LatencyRecords&
  {
    return client.latency_records_;
  }

  static void set_first_request_time(InferenceClient& client, TimePoint tp)
  {
    client.first_request_time_ = tp;
  }

  static void set_last_response_time(InferenceClient& client, TimePoint tp)
  {
    client.last_response_time_ = tp;
  }

  static void set_total_inference_count(
      InferenceClient& client, std::size_t count)
  {
    client.total_inference_count_ = count;
  }

  static void set_total_requests_sent(
      InferenceClient& client, std::size_t count)
  {
    client.total_requests_sent_ = count;
  }

  static void set_success_requests(InferenceClient& client, std::size_t count)
  {
    client.success_requests_ = count;
  }

  static void set_rejected_requests(InferenceClient& client, std::size_t count)
  {
    client.rejected_requests_ = count;
  }

  static void record_latency(
      InferenceClient& client, const LatencySample& sample)
  {
    client.record_latency(sample);
  }

  static void log_latency_summary(const InferenceClient& client)
  {
    client.log_latency_summary();
  }
};
#endif  // SONAR_IGNORE_END
// GCOVR_EXCL_STOP
}  // namespace starpu_server
