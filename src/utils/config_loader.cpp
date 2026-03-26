#include "config_loader.hpp"

#include <yaml-cpp/yaml.h>

#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

#include "config_loader_batching.hpp"
#include "config_loader_congestion.hpp"
#include "config_loader_device.hpp"
#include "config_loader_io.hpp"
#include "utils/exceptions.hpp"
#include "utils/logger.hpp"

namespace starpu_server {

inline namespace config_loader_detail {

struct TransparentStringHash {
  using is_transparent = void;

  [[nodiscard]] auto operator()(std::string_view value) const noexcept
      -> std::size_t
  {
    return std::hash<std::string_view>{}(value);
  }
};

// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
using ConfigLoaderPostParseHook = std::function<void(RuntimeConfig&)>;

auto
config_loader_post_parse_hook() -> ConfigLoaderPostParseHook&
{
  static ConfigLoaderPostParseHook hook;
  return hook;
}
#endif  // SONAR_IGNORE_END
// GCOVR_EXCL_STOP

void
parse_verbosity(const YAML::Node& root, RuntimeConfig& cfg)
{
  if (root["verbose"]) {
    cfg.verbosity = parse_verbosity_level(parse_scalar<std::string>(
        root["verbose"], "verbose", "a scalar value"));
  } else if (root["verbosity"]) {
    cfg.verbosity = parse_verbosity_level(parse_scalar<std::string>(
        root["verbosity"], "verbosity", "a scalar value"));
  }
}

void
parse_config_name(const YAML::Node& root, RuntimeConfig& cfg)
{
  const YAML::Node name_node = root["name"];
  if (!name_node) {
    return;
  }
  if (!name_node.IsScalar()) {
    throw std::invalid_argument(
        "Configuration option 'name' must be a scalar string");
  }
  cfg.name = name_node.as<std::string>();
}

void
validate_required_keys(const YAML::Node& root)
{
  const std::vector<std::string> required_keys{
      "name",
      "model",
      "inputs",
      "outputs",
      "pool_size",
      "max_batch_size",
      "batch_coalesce_timeout_ms"};
  std::vector<std::string> missing_keys;
  missing_keys.reserve(required_keys.size());
  for (const auto& key : required_keys) {
    if (!root[key]) {
      missing_keys.push_back(key);
    }
  }
  if (missing_keys.empty()) {
    return;
  }
  if (missing_keys.size() == 1U) {
    throw std::invalid_argument(
        std::string("Missing required key: ") + missing_keys.front());
  }
  std::ostringstream oss;
  oss << "Missing required keys: ";
  for (std::size_t i = 0; i < missing_keys.size(); ++i) {
    if (i != 0U) {
      oss << ", ";
    }
    oss << missing_keys[i];
  }
  throw std::invalid_argument(oss.str());
}

void
validate_allowed_keys(const YAML::Node& root)
{
  static const std::unordered_set<
      std::string, TransparentStringHash, std::equal_to<>>
      kAllowedKeys{
          "verbose",
          "verbosity",
          "name",
          "model",
          "model_name",
          "starpu_env",
          "device_ids",
          "group_cpu_by_numa",
          "gpu_model_replication",
          "inputs",
          "outputs",
          "batch_coalesce_timeout_ms",
          "address",
          "metrics_port",
          "max_message_bytes",
          "max_queue_size",
          "max_inflight_tasks",
          "max_batch_size",
          "dynamic_batching",
          "congestion",
          "pool_size",
          "trace_enabled",
          "trace_output",
          "warmup_pregen_inputs",
          "warmup_request_nb",
          "warmup_batches_per_worker",
          "seed",
          "sync",
          "use_cpu",
          "use_cuda"};

  for (const auto& kvalue : root) {
    if (!kvalue.first.IsScalar()) {
      throw std::invalid_argument("Configuration keys must be scalar strings");
    }
    const auto key = kvalue.first.as<std::string>();
    if (!kAllowedKeys.contains(key)) {
      if (key == "scheduler") {
        throw std::invalid_argument(
            "Unknown configuration option: scheduler (use starpu_env with "
            "STARPU_SCHED)");
      }
      throw std::invalid_argument(
          std::string("Unknown configuration option: ") + key);
    }
  }
}

void
parse_model_node(const YAML::Node& root, RuntimeConfig& cfg)
{
  if (root["model"]) {
    const YAML::Node model_node = root["model"];
    if (!model_node.IsScalar()) {
      throw std::invalid_argument("model must be a scalar string");
    }
    auto& model = ensure_model(cfg);
    model.path = model_node.as<std::string>();
    if (model.path.empty()) {
      throw std::invalid_argument("model must not be empty");
    }
    const std::filesystem::path model_path{model.path};
    if (!std::filesystem::exists(model_path)) {
      throw std::invalid_argument(
          std::string("Model path does not exist: ") + model.path);
    }
    if (!std::filesystem::is_regular_file(model_path)) {
      throw std::invalid_argument(
          std::string("Model path must be a regular file: ") + model.path);
    }
    std::ifstream model_stream(model_path, std::ios::binary);
    if (!model_stream.good()) {
      throw std::invalid_argument(
          std::string("Model path is not readable: ") + model.path);
    }
  }

  const YAML::Node model_name_node = root["model_name"];
  if (model_name_node) {
    if (!model_name_node.IsScalar()) {
      throw std::invalid_argument("model_name must be a scalar string");
    }
    auto& model = ensure_model(cfg);
    model.name = model_name_node.as<std::string>();
  }
}

void
parse_starpu_env(const YAML::Node& root, RuntimeConfig& cfg)
{
  const YAML::Node env_node = root["starpu_env"];
  if (!env_node) {
    return;
  }

  if (!env_node.IsMap()) {
    throw std::invalid_argument(
        "starpu_env must be a mapping of variable names to values");
  }

  for (const auto& item : env_node) {
    if (!item.first.IsScalar()) {
      throw std::invalid_argument("starpu_env entries must have scalar keys");
    }
    const auto key = item.first.as<std::string>();
    if (!item.second.IsScalar()) {
      throw std::invalid_argument(
          std::format("starpu_env entry '{}' must have a scalar value", key));
    }
    cfg.starpu_env[key] = item.second.as<std::string>();
  }
}

void
parse_generation_nodes(const YAML::Node& root, RuntimeConfig& cfg)
{
  if (root["warmup_pregen_inputs"]) {
    const int tmp = parse_scalar<int>(
        root["warmup_pregen_inputs"], "warmup_pregen_inputs", "an integer");
    if (tmp < 0) {
      throw std::invalid_argument("warmup_pregen_inputs must be >= 0");
    }
    cfg.batching.warmup_pregen_inputs = static_cast<size_t>(tmp);
  }
  if (root["warmup_request_nb"]) {
    const int tmp = parse_scalar<int>(
        root["warmup_request_nb"], "warmup_request_nb", "an integer");
    if (tmp < 0) {
      throw std::invalid_argument("warmup_request_nb must be >= 0");
    }
    cfg.batching.warmup_request_nb = tmp;
  }
  if (root["warmup_batches_per_worker"]) {
    const int tmp = parse_scalar<int>(
        root["warmup_batches_per_worker"], "warmup_batches_per_worker",
        "an integer");
    if (tmp < 0) {
      throw std::invalid_argument("warmup_batches_per_worker must be >= 0");
    }
    cfg.batching.warmup_batches_per_worker = tmp;
  }
}

void
parse_seed_tolerances_and_flags(const YAML::Node& root, RuntimeConfig& cfg)
{
  if (root["seed"]) {
    const auto tmp =
        parse_scalar<long long>(root["seed"], "seed", "an integer");
    if (tmp < 0) {
      throw std::invalid_argument("seed must be >= 0");
    }
    cfg.seed = static_cast<uint64_t>(tmp);
  }
  if (root["sync"]) {
    cfg.batching.synchronous =
        parse_scalar<bool>(root["sync"], "sync", "a boolean");
  }
}

}  // namespace config_loader_detail

// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
void
set_config_loader_post_parse_hook(ConfigLoaderPostParseHook hook)
{
  config_loader_post_parse_hook() = std::move(hook);
}

void
reset_config_loader_post_parse_hook()
{
  config_loader_post_parse_hook() = {};
}

auto
parse_tensor_nodes_for_test(
    const YAML::Node& nodes, std::size_t max_inputs, std::string_view label,
    std::size_t max_dims) -> std::vector<TensorConfig>
{
  return parse_tensor_nodes(nodes, max_inputs, label, max_dims);
}

void
parse_congestion_horizons_for_test(
    const YAML::Node& congestion_node, RuntimeConfig& cfg)
{
  parse_congestion_horizons(congestion_node, cfg);
}
#endif  // SONAR_IGNORE_END
// GCOVR_EXCL_STOP

inline namespace config_loader_detail {

void
mark_config_invalid(RuntimeConfig& cfg, const std::string& message)
{
  log_error(std::string("Failed to load config: ") + message);
  cfg.valid = false;
}

void
validate_cross_field_invariants(const RuntimeConfig& cfg)
{
  validate_batching_settings_coherence(cfg.batching);

  if (!cfg.devices.use_cpu && !cfg.devices.use_cuda) {
    throw std::invalid_argument(
        "At least one execution backend must be enabled: set use_cpu: true "
        "and/or configure use_cuda with device_ids");
  }
  if (cfg.devices.use_cuda && cfg.devices.ids.empty()) {
    throw std::invalid_argument(
        "use_cuda is enabled but no CUDA device_ids are configured");
  }
  if (!cfg.devices.use_cuda && !cfg.devices.ids.empty()) {
    throw std::invalid_argument(
        "device_ids are configured but use_cuda is disabled");
  }
}

void
parse_config_file(
    const std::string& path, RuntimeConfig& cfg,
    bool& max_message_bytes_configured)
{
  try {
    YAML::Node root = YAML::LoadFile(path);
    if (!root || !root.IsMap()) {
      throw std::invalid_argument("Config root must be a mapping");
    }

    max_message_bytes_configured = static_cast<bool>(root["max_message_bytes"]);
    parse_verbosity(root, cfg);
    parse_config_name(root, cfg);
    validate_allowed_keys(root);
    validate_required_keys(root);
    parse_model_node(root, cfg);
    parse_io_nodes(root, cfg);
    parse_network_and_delay(root, cfg);
    parse_message_and_batching(root, cfg);
    parse_congestion(root, cfg);
    parse_generation_nodes(root, cfg);
    parse_device_nodes(root, cfg);
    parse_seed_tolerances_and_flags(root, cfg);
    parse_starpu_env(root, cfg);
// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
    if (const auto& hook = config_loader_post_parse_hook(); hook) {
      hook(cfg);
    }
#endif  // SONAR_IGNORE_END
    // GCOVR_EXCL_STOP
  }
  catch (const YAML::Exception& exception) {
    mark_config_invalid(cfg, exception.what());
  }
  catch (const std::invalid_argument& exception) {
    mark_config_invalid(cfg, exception.what());
  }
  catch (const std::filesystem::filesystem_error& exception) {
    mark_config_invalid(cfg, exception.what());
  }
}

void
finalize_config(RuntimeConfig& cfg, bool max_message_bytes_configured)
{
  if (!cfg.valid) {
    return;
  }

  try {
    validate_cross_field_invariants(cfg);

    if (max_message_bytes_configured) {
      if (cfg.model.has_value()) {
        const auto required_bytes = compute_model_message_bytes(
            cfg.batching.max_batch_size, cfg.model->inputs, cfg.model->outputs,
            0);
        if (required_bytes > cfg.batching.max_message_bytes) {
          mark_config_invalid(
              cfg,
              std::format(
                  "max_message_bytes ({}) is too small for configured model "
                  "(requires at least {} bytes)",
                  cfg.batching.max_message_bytes, required_bytes));
        }
      }
    } else {
      cfg.batching.max_message_bytes = compute_max_message_bytes(
          cfg.batching.max_batch_size, cfg.model,
          cfg.batching.max_message_bytes);
    }
  }
  catch (const InvalidDimensionException& invalid_dimension) {
    mark_config_invalid(cfg, invalid_dimension.what());
  }
  catch (const MessageSizeOverflowException& message_size_overflow) {
    mark_config_invalid(cfg, message_size_overflow.what());
  }
  catch (const UnsupportedDtypeException& unsupported_dtype) {
    mark_config_invalid(cfg, unsupported_dtype.what());
  }
  catch (const std::invalid_argument& invalid_argument) {
    mark_config_invalid(cfg, invalid_argument.what());
  }

  if (!cfg.valid) {
    return;
  }

  const auto grpc_limit =
      static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (cfg.batching.max_message_bytes > grpc_limit) {
    log_warning(std::format(
        "max_message_bytes ({}) exceeds gRPC limit ({}); gRPC will clamp "
        "to {}. Consider reducing max_message_bytes.",
        cfg.batching.max_message_bytes, grpc_limit, grpc_limit));
  }
}

}  // namespace config_loader_detail

auto
load_config(const std::string& path) -> RuntimeConfig
{
  RuntimeConfig cfg;
  bool max_message_bytes_configured = false;
  parse_config_file(path, cfg, max_message_bytes_configured);
  finalize_config(cfg, max_message_bytes_configured);
  return cfg;
}

}  // namespace starpu_server
