#include "config_loader.hpp"

#include <yaml-cpp/yaml.h>

#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "datatype_utils.hpp"
#include "logger.hpp"

namespace starpu_server {

namespace {

constexpr int kMinPort = 1;
constexpr int kMaxPort = 65535;

struct TransparentStringHash {
  using hash_type = std::hash<std::string_view>;
  using is_transparent = void;

  auto operator()(std::string_view value) const noexcept -> std::size_t
  {
    return hash_type{}(value);
  }

  auto operator()(const std::string& value) const noexcept -> std::size_t
  {
    return (*this)(std::string_view{value});
  }
};

auto parse_tensor_nodes(
    const YAML::Node& nodes, std::size_t max_inputs,
    std::size_t max_dims) -> std::vector<TensorConfig>;

void
parse_verbosity(const YAML::Node& root, RuntimeConfig& cfg)
{
  if (root["verbose"]) {
    cfg.verbosity = parse_verbosity_level(root["verbose"].as<std::string>());
  } else if (root["verbosity"]) {
    cfg.verbosity = parse_verbosity_level(root["verbosity"].as<std::string>());
  }
}

auto
validate_required_keys(const YAML::Node& root, RuntimeConfig& cfg) -> bool
{
  const std::vector<std::string> required_keys{
      "model",     "inputs",         "outputs",
      "pool_size", "max_batch_size", "batch_coalesce_timeout_ms"};
  for (const auto& key : required_keys) {
    if (!root[key]) {
      log_error(std::string("Missing required key: ") + key);
      cfg.valid = false;
    }
  }
  return cfg.valid;
}

auto
validate_allowed_keys(const YAML::Node& root, RuntimeConfig& cfg) -> bool
{
  static const std::unordered_set<
      std::string, TransparentStringHash, std::equal_to<>>
      kAllowedKeys{
          "verbose",
          "verbosity",
          "scheduler",
          "model",
          "request_nb",
          "device_ids",
          "inputs",
          "outputs",
          "delay_us",
          "batch_coalesce_timeout_ms",
          "address",
          "metrics_port",
          "max_message_bytes",
          "max_batch_size",
          "dynamic_batching",
          "pool_size",
          "pregen_inputs",
          "warmup_pregen_inputs",
          "warmup_request_nb",
          "seed",
          "rtol",
          "atol",
          "validate_results",
          "sync",
          "use_cpu",
          "use_cuda"};

  for (const auto& kvalue : root) {
    if (!kvalue.first.IsScalar()) {
      log_error("Configuration keys must be scalar strings");
      cfg.valid = false;
      continue;
    }
    const auto key = kvalue.first.as<std::string>();
    if (!kAllowedKeys.contains(key)) {
      log_error(std::string("Unknown configuration option: ") + key);
      cfg.valid = false;
    }
  }
  return cfg.valid;
}

void
parse_scheduler_node(const YAML::Node& root, RuntimeConfig& cfg)
{
  if (root["scheduler"]) {
    cfg.scheduler = root["scheduler"].as<std::string>();
    if (!kAllowedSchedulers.contains(cfg.scheduler)) {
      log_error(std::string("Unknown scheduler: ") + cfg.scheduler);
      cfg.valid = false;
    }
  }
}

void
parse_model_node(const YAML::Node& root, RuntimeConfig& cfg)
{
  if (root["model"]) {
    cfg.models.resize(1);
    cfg.models[0].path = root["model"].as<std::string>();
    if (!std::filesystem::exists(cfg.models[0].path)) {
      log_error(
          std::string("Model path does not exist: ") + cfg.models[0].path);
      cfg.valid = false;
    }
  }
}

void
parse_request_nb_and_devices(const YAML::Node& root, RuntimeConfig& cfg)
{
  if (root["request_nb"]) {
    cfg.batching.request_nb = root["request_nb"].as<int>();
    if (cfg.batching.request_nb < 0) {
      log_error("request_nb must be >= 0");
      cfg.valid = false;
    }
  }
  if (root["device_ids"]) {
    log_error(
        "device_ids must be nested inside the use_cuda block (e.g. "
        "\"use_cuda: [{ device_ids: [0] }]\")");
    cfg.valid = false;
  }
}

void
parse_device_nodes(const YAML::Node& root, RuntimeConfig& cfg)
{
  if (root["use_cpu"]) {
    cfg.devices.use_cpu = root["use_cpu"].as<bool>();
  }

  const YAML::Node use_cuda_node = root["use_cuda"];
  if (!use_cuda_node) {
    return;
  }

  if (use_cuda_node.IsScalar()) {
    cfg.devices.use_cuda = use_cuda_node.as<bool>();
    return;
  }

  if (!use_cuda_node.IsSequence()) {
    log_error("use_cuda must be a boolean or a sequence of device mappings");
    cfg.valid = false;
    return;
  }

  cfg.devices.use_cuda = true;
  cfg.devices.ids.clear();

  for (const auto& entry : use_cuda_node) {
    if (!entry.IsMap()) {
      log_error("use_cuda entries must be mappings that define device_ids");
      cfg.valid = false;
      continue;
    }

    const YAML::Node device_ids_node = entry["device_ids"];

    if (!device_ids_node) {
      log_error("use_cuda entries require a device_ids sequence");
      cfg.valid = false;
      continue;
    }

    if (!device_ids_node.IsSequence()) {
      log_error("device_ids inside use_cuda must be a sequence");
      cfg.valid = false;
      continue;
    }

    const auto device_ids = device_ids_node.as<std::vector<int>>();
    cfg.devices.ids.insert(
        cfg.devices.ids.end(), device_ids.begin(), device_ids.end());
  }

  if (cfg.devices.ids.empty()) {
    log_error("use_cuda requires at least one device_ids entry");
    cfg.valid = false;
    cfg.devices.use_cuda = false;
  }
}

void
parse_io_nodes(const YAML::Node& root, RuntimeConfig& cfg)
{
  if (root["inputs"]) {
    cfg.models.resize(1);
    cfg.models[0].inputs = parse_tensor_nodes(
        root["inputs"], cfg.limits.max_inputs, cfg.limits.max_dims);
  }
  if (root["outputs"]) {
    cfg.models.resize(1);
    cfg.models[0].outputs = parse_tensor_nodes(
        root["outputs"], cfg.limits.max_inputs, cfg.limits.max_dims);
  }
}

void
parse_network_and_delay(const YAML::Node& root, RuntimeConfig& cfg)
{
  if (root["delay_us"]) {
    cfg.batching.delay_us = root["delay_us"].as<int>();
    if (cfg.batching.delay_us < 0) {
      cfg.valid = false;
      throw std::invalid_argument("delay_us must be >= 0");
    }
  }
  if (root["batch_coalesce_timeout_ms"]) {
    cfg.batching.batch_coalesce_timeout_ms =
        root["batch_coalesce_timeout_ms"].as<int>();
    if (cfg.batching.batch_coalesce_timeout_ms < 0) {
      cfg.valid = false;
      throw std::invalid_argument("batch_coalesce_timeout_ms must be >= 0");
    }
  }
  if (root["address"]) {
    cfg.server_address = root["address"].as<std::string>();
  }
  if (root["metrics_port"]) {
    cfg.metrics_port = root["metrics_port"].as<int>();
    if (cfg.metrics_port < kMinPort || cfg.metrics_port > kMaxPort) {
      log_error("metrics_port must be between 1 and 65535");
      cfg.valid = false;
    }
  }
}

void
parse_message_and_batching(const YAML::Node& root, RuntimeConfig& cfg)
{
  if (root["max_message_bytes"]) {
    const auto tmp = root["max_message_bytes"].as<long long>();
    if (tmp < 0 || static_cast<unsigned long long>(tmp) >
                       std::numeric_limits<std::size_t>::max()) {
      throw std::invalid_argument(
          "max_message_bytes must be >= 0 and fit in size_t");
    }
    cfg.batching.max_message_bytes = static_cast<std::size_t>(tmp);
  }
  if (root["max_batch_size"]) {
    cfg.batching.max_batch_size = root["max_batch_size"].as<int>();
    if (cfg.batching.max_batch_size <= 0) {
      throw std::invalid_argument("max_batch_size must be > 0");
    }
  }
  if (root["dynamic_batching"]) {
    cfg.batching.dynamic_batching = root["dynamic_batching"].as<bool>();
  }
  if (root["pool_size"]) {
    cfg.batching.pool_size = root["pool_size"].as<int>();
    if (cfg.batching.pool_size <= 0) {
      throw std::invalid_argument("pool_size must be > 0");
    }
  }
}

void
parse_generation_nodes(const YAML::Node& root, RuntimeConfig& cfg)
{
  if (root["pregen_inputs"]) {
    const int tmp = root["pregen_inputs"].as<int>();
    if (tmp <= 0) {
      throw std::invalid_argument("pregen_inputs must be > 0");
    }
    cfg.batching.pregen_inputs = static_cast<size_t>(tmp);
  }
  if (root["warmup_pregen_inputs"]) {
    const int tmp = root["warmup_pregen_inputs"].as<int>();
    if (tmp <= 0) {
      throw std::invalid_argument("warmup_pregen_inputs must be > 0");
    }
    cfg.batching.warmup_pregen_inputs = static_cast<size_t>(tmp);
  }
  if (root["warmup_request_nb"]) {
    const int tmp = root["warmup_request_nb"].as<int>();
    if (tmp < 0) {
      throw std::invalid_argument("warmup_request_nb must be >= 0");
    }
    cfg.batching.warmup_request_nb = tmp;
  }
}

void
parse_seed_tolerances_and_flags(const YAML::Node& root, RuntimeConfig& cfg)
{
  if (root["seed"]) {
    const auto tmp = root["seed"].as<long long>();
    if (tmp < 0) {
      throw std::invalid_argument("seed must be >= 0");
    }
    cfg.seed = static_cast<uint64_t>(tmp);
  }
  if (root["rtol"]) {
    cfg.validation.rtol = root["rtol"].as<double>();
    if (cfg.validation.rtol < 0) {
      throw std::invalid_argument("rtol must be >= 0");
    }
  }
  if (root["atol"]) {
    cfg.validation.atol = root["atol"].as<double>();
    if (cfg.validation.atol < 0) {
      throw std::invalid_argument("atol must be >= 0");
    }
  }
  if (root["validate_results"]) {
    cfg.validation.validate_results = root["validate_results"].as<bool>();
  }
  if (root["sync"]) {
    cfg.batching.synchronous = root["sync"].as<bool>();
  }
}

auto
parse_tensor_nodes(
    const YAML::Node& nodes, std::size_t max_inputs,
    std::size_t max_dims) -> std::vector<TensorConfig>
{
  std::vector<TensorConfig> tensors;
  if (!nodes || !nodes.IsSequence()) {
    return tensors;
  }
  for (const auto& node : nodes) {
    if (tensors.size() >= max_inputs) {
      std::ostringstream oss;
      oss << "number of tensors must be <= " << max_inputs;
      throw std::invalid_argument(oss.str());
    }

    TensorConfig tensor_config{};
    if (node["name"]) {
      tensor_config.name = node["name"].as<std::string>();
    }
    if (!node["dims"]) {
      throw std::invalid_argument("tensor node missing dims");
    }
    if (!node["data_type"]) {
      throw std::invalid_argument("tensor node missing data_type");
    }

    tensor_config.dims = node["dims"].as<std::vector<int64_t>>();
    if (tensor_config.dims.size() > max_dims) {
      std::ostringstream oss;
      oss << "tensor dims must be <= " << max_dims;
      throw std::invalid_argument(oss.str());
    }

    for (size_t i = 0; i < tensor_config.dims.size(); ++i) {
      const auto dim_value = tensor_config.dims[i];
      if (dim_value <= 0) {
        std::ostringstream oss;
        oss << "dims[" << i << "] must be positive";
        throw std::invalid_argument(oss.str());
      }
      if (dim_value > std::numeric_limits<int>::max()) {
        std::ostringstream oss;
        oss << "dims[" << i
            << "] must be <= " << std::numeric_limits<int>::max();
        throw std::invalid_argument(oss.str());
      }
    }
    tensor_config.type =
        string_to_scalar_type(node["data_type"].as<std::string>());
    tensors.push_back(std::move(tensor_config));
  }
  return tensors;
}

}  // namespace

auto
load_config(const std::string& path) -> RuntimeConfig
{
  RuntimeConfig cfg;
  const auto mark_invalid = [&cfg](const std::string& message) {
    log_error(std::string("Failed to load config: ") + message);
    cfg.valid = false;
  };
  try {
    YAML::Node root = YAML::LoadFile(path);
    if (!root || !root.IsMap()) {
      log_error("Config root must be a mapping");
      cfg.valid = false;
      return cfg;
    }

    parse_verbosity(root, cfg);
    if (!validate_allowed_keys(root, cfg)) {
      return cfg;
    }
    if (!validate_required_keys(root, cfg)) {
      return cfg;
    }
    parse_scheduler_node(root, cfg);
    parse_model_node(root, cfg);
    parse_request_nb_and_devices(root, cfg);
    parse_io_nodes(root, cfg);
    parse_network_and_delay(root, cfg);
    parse_message_and_batching(root, cfg);
    parse_generation_nodes(root, cfg);
    parse_device_nodes(root, cfg);
    parse_seed_tolerances_and_flags(root, cfg);
  }
  catch (const YAML::Exception& exception) {
    mark_invalid(exception.what());
  }
  catch (const std::invalid_argument& exception) {
    mark_invalid(exception.what());
  }
  catch (const std::filesystem::filesystem_error& exception) {
    mark_invalid(exception.what());
  }

  if (cfg.valid) {
    try {
      cfg.batching.max_message_bytes = compute_max_message_bytes(
          cfg.batching.max_batch_size, cfg.models,
          cfg.batching.max_message_bytes);
    }
    catch (const InvalidDimensionException& invalid_dimension) {
      log_error(
          std::string("Failed to load config: ") + invalid_dimension.what());
      cfg.valid = false;
    }
    catch (const MessageSizeOverflowException& message_size_overflow) {
      log_error(
          std::string("Failed to load config: ") +
          message_size_overflow.what());
      cfg.valid = false;
    }
    catch (const UnsupportedDtypeException& unsupported_dtype) {
      log_error(
          std::string("Failed to load config: ") + unsupported_dtype.what());
      cfg.valid = false;
    }
  }
  return cfg;
}

}  // namespace starpu_server
