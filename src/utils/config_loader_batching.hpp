#pragma once

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <format>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

#include "config_loader_helpers.hpp"

namespace starpu_server::inline config_loader_detail {

struct BatchingConfigPresence {
  bool strategy = false;
  bool adaptive = false;
  bool fixed = false;
};

void
validate_allowed_batching_mapping_keys(
    const YAML::Node& node, std::string_view mapping_name,
    std::initializer_list<std::string_view> allowed_keys)
{
  for (const auto& entry : node) {
    if (!entry.first.IsScalar()) {
      throw std::invalid_argument(
          std::format("{} keys must be scalar strings", mapping_name));
    }

    const auto key = entry.first.as<std::string>();
    bool allowed = false;
    for (const auto allowed_key : allowed_keys) {
      if (key == allowed_key) {
        allowed = true;
        break;
      }
    }

    if (!allowed) {
      throw std::invalid_argument(std::format(
          "Unknown configuration option: {}.{}", mapping_name, key));
    }
  }
}

auto
parse_positive_batching_int(const YAML::Node& node, std::string_view key_path)
    -> int
{
  const auto value = parse_scalar<long long>(node, key_path, "an integer");
  if (value <= 0 || value > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(
        std::format("{} must be > 0 and fit in int", key_path));
  }
  return static_cast<int>(value);
}

auto
parse_nonnegative_batching_size_t(
    const YAML::Node& node, std::string_view key_path) -> std::size_t
{
  const auto value = parse_scalar<long long>(node, key_path, "an integer");
  if (value < 0 || static_cast<unsigned long long>(value) >
                       std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument(
        std::format("{} must be >= 0 and fit in size_t", key_path));
  }
  return static_cast<std::size_t>(value);
}

auto
parse_positive_batching_size_t(
    const YAML::Node& node, std::string_view key_path) -> std::size_t
{
  const auto value = parse_scalar<long long>(node, key_path, "an integer");
  if (value <= 0 || static_cast<unsigned long long>(value) >
                        std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument(
        std::format("{} must be > 0 and fit in size_t", key_path));
  }
  return static_cast<std::size_t>(value);
}

auto
detect_batching_config_presence(const YAML::Node& root)
    -> BatchingConfigPresence
{
  return BatchingConfigPresence{
      .strategy = static_cast<bool>(root["batching_strategy"]),
      .adaptive = static_cast<bool>(root["adaptive_batching"]),
      .fixed = static_cast<bool>(root["fixed_batching"]),
  };
}

void
parse_adaptive_batching(
    const YAML::Node& root, RuntimeConfig::BatchingSettings& batching)
{
  const YAML::Node adaptive_node = root["adaptive_batching"];
  if (!adaptive_node) {
    return;
  }
  if (!adaptive_node.IsMap()) {
    throw std::invalid_argument("adaptive_batching must be a mapping");
  }
  validate_allowed_batching_mapping_keys(
      adaptive_node, "adaptive_batching", {"min_batch_size", "max_batch_size"});

  if (adaptive_node["min_batch_size"]) {
    batching.adaptive.min_batch_size = parse_positive_batching_int(
        adaptive_node["min_batch_size"], "adaptive_batching.min_batch_size");
  }
  if (adaptive_node["max_batch_size"]) {
    batching.adaptive.max_batch_size = parse_positive_batching_int(
        adaptive_node["max_batch_size"], "adaptive_batching.max_batch_size");
  }
}

void
parse_fixed_batching(
    const YAML::Node& root, RuntimeConfig::BatchingSettings& batching)
{
  const YAML::Node fixed_node = root["fixed_batching"];
  if (!fixed_node) {
    return;
  }
  if (!fixed_node.IsMap()) {
    throw std::invalid_argument("fixed_batching must be a mapping");
  }
  validate_allowed_batching_mapping_keys(
      fixed_node, "fixed_batching", {"batch_size"});

  if (fixed_node["batch_size"]) {
    batching.fixed.batch_size = parse_positive_batching_int(
        fixed_node["batch_size"], "fixed_batching.batch_size");
  }
}

void
resolve_disabled_batching(
    const BatchingConfigPresence& presence,
    RuntimeConfig::BatchingSettings& batching)
{
  if (presence.adaptive || presence.fixed) {
    throw std::invalid_argument(
        "batching_strategy 'disabled' does not accept adaptive_batching or "
        "fixed_batching options");
  }

  batching.resolved_max_batch_size = 1;
  batching.adaptive.max_batch_size = 1;
  batching.fixed.batch_size = 1;
}

void
resolve_adaptive_batching(
    const YAML::Node& root, const BatchingConfigPresence& presence,
    RuntimeConfig::BatchingSettings& batching)
{
  if (presence.fixed) {
    throw std::invalid_argument(
        "fixed_batching cannot be used with batching_strategy 'adaptive'");
  }
  if (!root["adaptive_batching"] ||
      !root["adaptive_batching"]["max_batch_size"]) {
    throw std::invalid_argument(
        "batching_strategy 'adaptive' requires "
        "adaptive_batching.max_batch_size");
  }

  batching.resolved_max_batch_size =
      std::max(1, batching.adaptive.max_batch_size);
  batching.fixed.batch_size = std::max(1, batching.resolved_max_batch_size);
}

void
resolve_fixed_batching(
    const YAML::Node& root, const BatchingConfigPresence& presence,
    RuntimeConfig::BatchingSettings& batching)
{
  if (presence.adaptive) {
    throw std::invalid_argument(
        "adaptive_batching cannot be used with batching_strategy 'fixed'");
  }
  if (!root["fixed_batching"] || !root["fixed_batching"]["batch_size"]) {
    throw std::invalid_argument(
        "batching_strategy 'fixed' requires fixed_batching.batch_size");
  }

  batching.resolved_max_batch_size = std::max(1, batching.fixed.batch_size);
  batching.adaptive.max_batch_size =
      std::max(1, batching.resolved_max_batch_size);
}

void
parse_batching_strategy_settings(
    const YAML::Node& root, RuntimeConfig::BatchingSettings& batching)
{
  const auto presence = detect_batching_config_presence(root);
  if (!presence.strategy) {
    throw std::invalid_argument("Missing required key: batching_strategy");
  }

  batching.strategy = parse_batching_strategy_kind(parse_scalar<std::string>(
      root["batching_strategy"], "batching_strategy", "a scalar string"));

  parse_adaptive_batching(root, batching);
  parse_fixed_batching(root, batching);

  switch (batching.strategy) {
    case BatchingStrategyKind::Disabled:
      resolve_disabled_batching(presence, batching);
      break;
    case BatchingStrategyKind::Adaptive:
      resolve_adaptive_batching(root, presence, batching);
      break;
    case BatchingStrategyKind::Fixed:
      resolve_fixed_batching(root, presence, batching);
      break;
  }
}

void
parse_optional_max_message_bytes(
    const YAML::Node& root, RuntimeConfig::BatchingSettings& batching)
{
  if (root["max_message_bytes"]) {
    batching.max_message_bytes = parse_nonnegative_batching_size_t(
        root["max_message_bytes"], "max_message_bytes");
  }
}

void
parse_optional_batching_runtime_limits(
    const YAML::Node& root, RuntimeConfig::BatchingSettings& batching)
{
  if (root["pool_size"]) {
    batching.pool_size =
        parse_scalar<int>(root["pool_size"], "pool_size", "an integer");
    if (batching.pool_size <= 0) {
      throw std::invalid_argument("pool_size must be > 0");
    }
  }

  if (root["max_inflight_tasks"]) {
    batching.max_inflight_tasks = parse_nonnegative_batching_size_t(
        root["max_inflight_tasks"], "max_inflight_tasks");
  }

  if (root["max_queue_size"]) {
    batching.max_queue_size = parse_positive_batching_size_t(
        root["max_queue_size"], "max_queue_size");
  }
}

void
parse_optional_batching_trace_settings(
    const YAML::Node& root, RuntimeConfig::BatchingSettings& batching)
{
  if (root["trace_enabled"]) {
    batching.trace_enabled =
        parse_scalar<bool>(root["trace_enabled"], "trace_enabled", "a boolean");
  }

  if (root["trace_output"]) {
    batching.trace_output_path =
        resolve_trace_output_directory(parse_scalar<std::string>(
            root["trace_output"], "trace_output", "a scalar string"));
    if (batching.trace_output_path.empty()) {
      throw std::invalid_argument("trace_output must not be empty");
    }
  }
}

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
  parse_optional_max_message_bytes(root, cfg.batching);
  parse_batching_strategy_settings(root, cfg.batching);
  parse_optional_batching_runtime_limits(root, cfg.batching);
  parse_optional_batching_trace_settings(root, cfg.batching);
}

}  // namespace starpu_server::inline config_loader_detail
