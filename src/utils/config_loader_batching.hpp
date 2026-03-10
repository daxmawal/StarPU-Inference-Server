#pragma once

#include <yaml-cpp/yaml.h>

#include <limits>
#include <stdexcept>

#include "config_loader_helpers.hpp"

namespace starpu_server { namespace {

void
parse_network_and_delay(const YAML::Node& root, RuntimeConfig& cfg)
{
  if (root["batch_coalesce_timeout_ms"]) {
    cfg.batching.batch_coalesce_timeout_ms = parse_scalar<int>(
        root["batch_coalesce_timeout_ms"], "batch_coalesce_timeout_ms",
        "an integer");
    if (cfg.batching.batch_coalesce_timeout_ms < 0) {
      throw std::invalid_argument("batch_coalesce_timeout_ms must be >= 0");
    }
  }
  if (root["address"]) {
    cfg.server_address = parse_scalar<std::string>(
        root["address"], "address", "a scalar string");
  }
  if (root["metrics_port"]) {
    cfg.metrics_port =
        parse_scalar<int>(root["metrics_port"], "metrics_port", "an integer");
    if (cfg.metrics_port < kMinPort || cfg.metrics_port > kMaxPort) {
      throw std::invalid_argument("metrics_port must be between 1 and 65535");
    }
  }
}

void
parse_message_and_batching(const YAML::Node& root, RuntimeConfig& cfg)
{
  if (root["max_message_bytes"]) {
    const auto tmp = parse_scalar<long long>(
        root["max_message_bytes"], "max_message_bytes", "an integer");
    if (tmp < 0 || static_cast<unsigned long long>(tmp) >
                       std::numeric_limits<std::size_t>::max()) {
      throw std::invalid_argument(
          "max_message_bytes must be >= 0 and fit in size_t");
    }
    cfg.batching.max_message_bytes = static_cast<std::size_t>(tmp);
  }
  if (root["max_batch_size"]) {
    cfg.batching.max_batch_size = parse_scalar<int>(
        root["max_batch_size"], "max_batch_size", "an integer");
    if (cfg.batching.max_batch_size <= 0) {
      throw std::invalid_argument("max_batch_size must be > 0");
    }
  }
  if (root["dynamic_batching"]) {
    cfg.batching.dynamic_batching = parse_scalar<bool>(
        root["dynamic_batching"], "dynamic_batching", "a boolean");
  }
  if (root["pool_size"]) {
    cfg.batching.pool_size =
        parse_scalar<int>(root["pool_size"], "pool_size", "an integer");
    if (cfg.batching.pool_size <= 0) {
      throw std::invalid_argument("pool_size must be > 0");
    }
  }
  if (root["max_inflight_tasks"]) {
    const auto tmp = parse_scalar<long long>(
        root["max_inflight_tasks"], "max_inflight_tasks", "an integer");
    if (tmp < 0 || static_cast<unsigned long long>(tmp) >
                       std::numeric_limits<std::size_t>::max()) {
      throw std::invalid_argument(
          "max_inflight_tasks must be >= 0 and fit in size_t");
    }
    cfg.batching.max_inflight_tasks = static_cast<std::size_t>(tmp);
  }
  if (root["max_queue_size"]) {
    const auto tmp = parse_scalar<long long>(
        root["max_queue_size"], "max_queue_size", "an integer");
    if (tmp <= 0 || static_cast<unsigned long long>(tmp) >
                        std::numeric_limits<std::size_t>::max()) {
      throw std::invalid_argument(
          "max_queue_size must be > 0 and fit in size_t");
    }
    cfg.batching.max_queue_size = static_cast<std::size_t>(tmp);
  }
  if (root["trace_enabled"]) {
    cfg.batching.trace_enabled =
        parse_scalar<bool>(root["trace_enabled"], "trace_enabled", "a boolean");
  }
  if (root["trace_output"]) {
    cfg.batching.trace_output_path =
        resolve_trace_output_directory(parse_scalar<std::string>(
            root["trace_output"], "trace_output", "a scalar string"));
    if (cfg.batching.trace_output_path.empty()) {
      throw std::invalid_argument("trace_output must not be empty");
    }
  }
}

}}  // namespace starpu_server
