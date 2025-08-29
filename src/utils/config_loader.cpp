#include "config_loader.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "datatype_utils.hpp"
#include "logger.hpp"

namespace starpu_server {

namespace {

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

    TensorConfig t{};
    if (node["name"]) {
      t.name = node["name"].as<std::string>();
    }
    if (!node["dims"]) {
      throw std::invalid_argument("tensor node missing dims");
    }
    if (!node["data_type"]) {
      throw std::invalid_argument("tensor node missing data_type");
    }

    t.dims = node["dims"].as<std::vector<int64_t>>();
    if (t.dims.size() > max_dims) {
      std::ostringstream oss;
      oss << "tensor dims must be <= " << max_dims;
      throw std::invalid_argument(oss.str());
    }

    for (size_t i = 0; i < t.dims.size(); ++i) {
      const auto d = t.dims[i];
      if (d <= 0) {
        std::ostringstream oss;
        oss << "dims[" << i << "] must be positive";
        throw std::invalid_argument(oss.str());
      }
      if (d > std::numeric_limits<int>::max()) {
        std::ostringstream oss;
        oss << "dims[" << i
            << "] must be <= " << std::numeric_limits<int>::max();
        throw std::invalid_argument(oss.str());
      }
    }
    t.type = string_to_scalar_type(node["data_type"].as<std::string>());
    tensors.push_back(std::move(t));
  }
  return tensors;
}

}  // namespace

auto
load_config(const std::string& path) -> RuntimeConfig
{
  RuntimeConfig cfg;
  try {
    YAML::Node root = YAML::LoadFile(path);

    if (root["verbose"]) {
      cfg.verbosity = parse_verbosity_level(root["verbose"].as<std::string>());
    } else if (root["verbosity"]) {
      cfg.verbosity =
          parse_verbosity_level(root["verbosity"].as<std::string>());
    }

    const std::vector<std::string> required_keys{"model", "input", "output"};
    for (const auto& key : required_keys) {
      if (!root[key]) {
        log_error(std::string("Missing required key: ") + key);
        cfg.valid = false;
      }
    }

    if (!cfg.valid) {
      return cfg;
    }
    if (root["scheduler"]) {
      cfg.scheduler = root["scheduler"].as<std::string>();
      if (!kAllowedSchedulers.contains(cfg.scheduler)) {
        log_error(std::string("Unknown scheduler: ") + cfg.scheduler);
        cfg.valid = false;
      }
    }
    if (root["model"]) {
      cfg.model_path = root["model"].as<std::string>();
      if (!std::filesystem::exists(cfg.model_path)) {
        log_error(std::string("Model path does not exist: ") + cfg.model_path);
        cfg.valid = false;
      }
    }
    if (root["iterations"]) {
      cfg.iterations = root["iterations"].as<int>();
      if (cfg.iterations < 0) {
        log_error("iterations must be >= 0");
        cfg.valid = false;
      }
    }
    if (root["device_ids"]) {
      cfg.device_ids = root["device_ids"].as<std::vector<int>>();
      if (!cfg.device_ids.empty()) {
        cfg.use_cuda = true;
      }
    }
    if (root["input"]) {
      cfg.inputs =
          parse_tensor_nodes(root["input"], cfg.max_inputs, cfg.max_dims);
    }
    if (root["output"]) {
      cfg.outputs =
          parse_tensor_nodes(root["output"], cfg.max_inputs, cfg.max_dims);
    }
    if (root["delay"]) {
      cfg.delay_ms = root["delay"].as<int>();
      if (cfg.delay_ms < 0) {
        cfg.valid = false;
        throw std::invalid_argument("delay must be >= 0");
      }
    }
    if (root["address"]) {
      cfg.server_address = root["address"].as<std::string>();
    }
    if (root["metrics_port"]) {
      cfg.metrics_port = root["metrics_port"].as<int>();
      if (cfg.metrics_port < 1 || cfg.metrics_port > 65535) {
        log_error("metrics_port must be between 1 and 65535");
        cfg.valid = false;
      }
    }
    if (root["max_message_bytes"]) {
      const auto tmp = root["max_message_bytes"].as<long long>();
      if (tmp < 0 || static_cast<unsigned long long>(tmp) >
                         std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(
            "max_message_bytes must be >= 0 and fit in size_t");
      }
      cfg.max_message_bytes = static_cast<std::size_t>(tmp);
    }
    if (root["max_batch_size"]) {
      cfg.max_batch_size = root["max_batch_size"].as<int>();
      if (cfg.max_batch_size <= 0) {
        throw std::invalid_argument("max_batch_size must be > 0");
      }
    }
    if (root["pregen_inputs"]) {
      const int tmp = root["pregen_inputs"].as<int>();
      if (tmp <= 0) {
        throw std::invalid_argument("pregen_inputs must be > 0");
      }
      cfg.pregen_inputs = static_cast<size_t>(tmp);
    }
    if (root["warmup_pregen_inputs"]) {
      const int tmp = root["warmup_pregen_inputs"].as<int>();
      if (tmp <= 0) {
        throw std::invalid_argument("warmup_pregen_inputs must be > 0");
      }
      cfg.warmup_pregen_inputs = static_cast<size_t>(tmp);
    }
    if (root["warmup_iterations"]) {
      const int tmp = root["warmup_iterations"].as<int>();
      if (tmp < 0) {
        throw std::invalid_argument("warmup_iterations must be >= 0");
      }
      cfg.warmup_iterations = tmp;
    }
    if (root["seed"]) {
      const auto tmp = root["seed"].as<long long>();
      if (tmp < 0) {
        throw std::invalid_argument("seed must be >= 0");
      }
      cfg.seed = static_cast<uint64_t>(tmp);
    }
    if (root["rtol"]) {
      cfg.rtol = root["rtol"].as<double>();
      if (cfg.rtol < 0) {
        throw std::invalid_argument("rtol must be >= 0");
      }
    }
    if (root["atol"]) {
      cfg.atol = root["atol"].as<double>();
      if (cfg.atol < 0) {
        throw std::invalid_argument("atol must be >= 0");
      }
    }
    if (root["sync"]) {
      cfg.synchronous = root["sync"].as<bool>();
    }
    if (root["use_cpu"]) {
      cfg.use_cpu = root["use_cpu"].as<bool>();
    }
    if (root["use_cuda"]) {
      cfg.use_cuda = root["use_cuda"].as<bool>();
    }
  }
  catch (const std::exception& e) {
    log_error(std::string("Failed to load config: ") + e.what());
    cfg.valid = false;
  }

  if (cfg.valid) {
    try {
      cfg.max_message_bytes = compute_max_message_bytes(
          cfg.max_batch_size, cfg.inputs, cfg.outputs, cfg.max_message_bytes);
    }
    catch (const InvalidDimensionException& e) {
      log_error(std::string("Failed to load config: ") + e.what());
      cfg.valid = false;
    }
    catch (const MessageSizeOverflowException& e) {
      log_error(std::string("Failed to load config: ") + e.what());
      cfg.valid = false;
    }
    catch (const UnsupportedDtypeException& e) {
      log_error(std::string("Failed to load config: ") + e.what());
      cfg.valid = false;
    }
  }
  return cfg;
}

}  // namespace starpu_server
