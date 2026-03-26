#pragma once

#include <chrono>
#include <cstddef>
#include <vector>

#include "grpc/client/inference_client.hpp"

namespace starpu_server {
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
    client.total_inference_count_.store(count, std::memory_order_relaxed);
  }

  static void set_total_requests_sent(
      InferenceClient& client, std::size_t count)
  {
    client.total_requests_sent_.store(count, std::memory_order_relaxed);
  }

  static void set_success_requests(InferenceClient& client, std::size_t count)
  {
    client.success_requests_.store(count, std::memory_order_relaxed);
  }

  static void set_rejected_requests(InferenceClient& client, std::size_t count)
  {
    client.rejected_requests_.store(count, std::memory_order_relaxed);
  }

  static auto handled_requests(const InferenceClient& client) -> std::size_t
  {
    return client.success_requests_.load(std::memory_order_relaxed) +
           client.rejected_requests_.load(std::memory_order_relaxed);
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
}  // namespace starpu_server
