#include "config_loader.hpp"

#include <yaml-cpp/yaml.h>

#include <cstdint>
#include <filesystem>
#include <format>
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

#include "datatype_utils.hpp"
#include "logger.hpp"

namespace starpu_server {

namespace {

constexpr int kMinPort = 1;
constexpr int kMaxPort = 65535;
const std::filesystem::path kDefaultTraceOutputFile{
    RuntimeConfig::BatchingSettings{}.trace_output_path};

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
auto
config_loader_post_parse_hook() -> ConfigLoaderPostParseHook&
{
  static ConfigLoaderPostParseHook hook;
  return hook;
}
#endif  // SONAR_IGNORE_END
// GCOVR_EXCL_STOP

auto parse_tensor_nodes(
    const YAML::Node& nodes, std::size_t max_inputs, std::string_view label,
    std::size_t max_dims) -> std::vector<TensorConfig>;

auto
make_indexed_path(std::string_view base, std::size_t index) -> std::string
{
  return std::format("{}[{}]", base, index);
}

auto
make_indexed_field_path(
    std::string_view base, std::size_t index,
    std::string_view field) -> std::string
{
  return std::format("{}[{}].{}", base, index, field);
}

template <typename T>
auto
parse_scalar(
    const YAML::Node& node, std::string_view key,
    std::string_view type_desc) -> T
{
  if (!node.IsScalar()) {
    throw std::invalid_argument(std::format("{} must be {}", key, type_desc));
  }
  try {
    return node.as<T>();
  }
  catch (const YAML::BadConversion&) {
    throw std::invalid_argument(std::format("{} must be {}", key, type_desc));
  }
}

auto
ensure_model(RuntimeConfig& cfg) -> ModelConfig&
{
  if (!cfg.model.has_value()) {
    cfg.model = ModelConfig{};
  }
  return *cfg.model;
}

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

auto
resolve_trace_output_directory(std::string directory) -> std::string
{
  if (directory.empty()) {
    return directory;
  }

  std::filesystem::path directory_path(directory);
  std::error_code status_ec;
  const bool exists = std::filesystem::exists(directory_path, status_ec);
  if (status_ec) {
    throw std::filesystem::filesystem_error(
        "trace_output", directory_path, status_ec);
  }

  if (exists) {
    std::error_code type_ec;
    if (!std::filesystem::is_directory(directory_path, type_ec)) {
      throw std::invalid_argument("trace_output must be a directory path");
    }
  } else {
    const auto extension = directory_path.extension();
    if (!extension.empty() &&
        extension == kDefaultTraceOutputFile.extension()) {
      throw std::invalid_argument(
          "trace_output must be a directory path (omit the filename)");
    }
  }

  return (directory_path / kDefaultTraceOutputFile).string();
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
    if (!std::filesystem::exists(model.path)) {
      throw std::invalid_argument(
          std::string("Model path does not exist: ") + model.path);
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
parse_device_nodes(const YAML::Node& root, RuntimeConfig& cfg)
{
  if (root["device_ids"]) {
    throw std::invalid_argument(
        "device_ids must be nested inside the use_cuda block (e.g. "
        "\"use_cuda: [{ device_ids: [0] }]\")");
  }
  if (root["use_cpu"]) {
    cfg.devices.use_cpu =
        parse_scalar<bool>(root["use_cpu"], "use_cpu", "a boolean");
  }
  if (root["group_cpu_by_numa"]) {
    cfg.devices.group_cpu_by_numa = parse_scalar<bool>(
        root["group_cpu_by_numa"], "group_cpu_by_numa", "a boolean");
  }

  const YAML::Node use_cuda_node = root["use_cuda"];
  if (!use_cuda_node) {
    return;
  }

  if (use_cuda_node.IsScalar()) {
    if (const bool enabled =
            parse_scalar<bool>(use_cuda_node, "use_cuda", "a boolean");
        !enabled) {
      cfg.devices.use_cuda = false;
      cfg.devices.ids.clear();
      return;
    }
    throw std::invalid_argument(
        "use_cuda must be a sequence of device mappings when enabled (e.g. "
        "\"use_cuda: [{ device_ids: [0] }]\")");
  }

  if (!use_cuda_node.IsSequence()) {
    throw std::invalid_argument(
        "use_cuda must be a boolean or a sequence of device mappings");
  }

  cfg.devices.use_cuda = true;
  cfg.devices.ids.clear();
  if (use_cuda_node.size() == 0U) {
    throw std::invalid_argument(
        "use_cuda requires at least one device_ids "
        "entry");
  }

  for (std::size_t i = 0; i < use_cuda_node.size(); ++i) {
    const auto& entry = use_cuda_node[i];
    if (!entry.IsMap()) {
      throw std::invalid_argument(std::format(
          "use_cuda[{}] must be a mapping that defines device_ids", i));
    }

    const YAML::Node device_ids_node = entry["device_ids"];

    if (!device_ids_node) {
      throw std::invalid_argument(
          std::format("use_cuda[{}].device_ids is required", i));
    }

    if (!device_ids_node.IsSequence()) {
      throw std::invalid_argument(std::format(
          "use_cuda[{}].device_ids must be a sequence of integers", i));
    }

    for (std::size_t j = 0; j < device_ids_node.size(); ++j) {
      const auto& id_node = device_ids_node[j];
      const auto id_path = std::format("use_cuda[{}].device_ids[{}]", i, j);
      const int device_id = parse_scalar<int>(id_node, id_path, "an integer");
      if (device_id < 0) {
        throw std::invalid_argument(std::format("{} must be >= 0", id_path));
      }
      cfg.devices.ids.push_back(device_id);
    }
  }

  if (cfg.devices.ids.empty()) {
    throw std::invalid_argument(
        "use_cuda requires at least one device_ids "
        "entry");
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
parse_io_nodes(const YAML::Node& root, RuntimeConfig& cfg)
{
  if (root["inputs"]) {
    auto& model = ensure_model(cfg);
    model.inputs = parse_tensor_nodes(
        root["inputs"], cfg.limits.max_inputs, "inputs", cfg.limits.max_dims);
  }
  if (root["outputs"]) {
    auto& model = ensure_model(cfg);
    model.outputs = parse_tensor_nodes(
        root["outputs"], cfg.limits.max_inputs, "outputs", cfg.limits.max_dims);
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

void
parse_tensor_name(
    const YAML::Node& node, std::size_t index, std::string_view label,
    TensorConfig& tensor_config)
{
  if (const YAML::Node name_node = node["name"]) {
    const auto name_path = make_indexed_field_path(label, index, "name");
    if (!name_node.IsScalar()) {
      throw std::invalid_argument(
          std::format("{} must be a scalar string", name_path));
    }
    tensor_config.name = name_node.as<std::string>();
  }
}

auto
parse_tensor_dims(
    const YAML::Node& node, std::size_t index, std::string_view label,
    std::size_t max_dims) -> std::vector<int64_t>
{
  const YAML::Node dims_node = node["dims"];
  const auto dims_path = make_indexed_field_path(label, index, "dims");
  if (!dims_node) {
    throw std::invalid_argument(std::format("{} is required", dims_path));
  }
  if (!dims_node.IsSequence()) {
    throw std::invalid_argument(
        std::format("{} must be a sequence of integers", dims_path));
  }
  if (dims_node.size() > max_dims) {
    std::ostringstream oss;
    oss << dims_path << " must have at most " << max_dims << " entries";
    throw std::invalid_argument(oss.str());
  }

  std::vector<int64_t> dims;
  dims.reserve(dims_node.size());
  for (std::size_t j = 0; j < dims_node.size(); ++j) {
    const auto& dim_node = dims_node[j];
    const auto dim_path = std::format("{}[{}]", dims_path, j);
    const auto dim_value =
        parse_scalar<long long>(dim_node, dim_path, "an integer");

    if (dim_value <= 0) {
      throw std::invalid_argument(std::format("{} must be > 0", dim_path));
    }
    if (dim_value > std::numeric_limits<int>::max()) {
      throw std::invalid_argument(std::format(
          "{} must be <= {}", dim_path, std::numeric_limits<int>::max()));
    }
    dims.push_back(static_cast<int64_t>(dim_value));
  }
  return dims;
}

auto
parse_tensor_type(
    const YAML::Node& node, std::size_t index,
    std::string_view label) -> at::ScalarType
{
  const YAML::Node type_node = node["data_type"];
  const auto type_path = make_indexed_field_path(label, index, "data_type");
  if (!type_node) {
    throw std::invalid_argument(std::format("{} is required", type_path));
  }
  if (!type_node.IsScalar()) {
    throw std::invalid_argument(
        std::format("{} must be a scalar string", type_path));
  }
  try {
    return string_to_scalar_type(type_node.as<std::string>());
  }
  catch (const std::invalid_argument& exception) {
    throw std::invalid_argument(
        std::format("{}: {}", type_path, exception.what()));
  }
}

auto
parse_tensor_entry(
    const YAML::Node& node, std::size_t index, std::string_view label,
    std::size_t max_dims) -> TensorConfig
{
  const auto entry_path = make_indexed_path(label, index);
  if (!node.IsMap()) {
    throw std::invalid_argument(
        std::format("{} must be a mapping", entry_path));
  }

  TensorConfig tensor_config{};
  parse_tensor_name(node, index, label, tensor_config);
  tensor_config.dims = parse_tensor_dims(node, index, label, max_dims);
  tensor_config.type = parse_tensor_type(node, index, label);
  return tensor_config;
}

auto
parse_tensor_nodes(
    const YAML::Node& nodes, std::size_t max_inputs, std::string_view label,
    std::size_t max_dims) -> std::vector<TensorConfig>
{
  std::vector<TensorConfig> tensors;
  if (!nodes) {
    return tensors;
  }
  if (!nodes.IsSequence()) {
    throw std::invalid_argument(
        std::format("{} must be a sequence of tensor definitions", label));
  }
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    const auto& node = nodes[i];
    if (tensors.size() >= max_inputs) {
      std::ostringstream oss;
      oss << label << " must have at most " << max_inputs << " entries";
      throw std::invalid_argument(oss.str());
    }
    tensors.push_back(parse_tensor_entry(node, i, label, max_dims));
  }
  return tensors;
}

}  // namespace

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
#endif  // SONAR_IGNORE_END
// GCOVR_EXCL_STOP

namespace {

void
mark_config_invalid(RuntimeConfig& cfg, const std::string& message)
{
  log_error(std::string("Failed to load config: ") + message);
  cfg.valid = false;
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

}  // namespace

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
