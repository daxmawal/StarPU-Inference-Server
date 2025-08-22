#include "config_loader.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "logger.hpp"
#include "transparent_hash.hpp"

namespace starpu_server {
namespace {

static auto
parse_type_string(const std::string& type_str) -> at::ScalarType
{
  std::string key = type_str;
  if (key.rfind("TYPE_", 0) == 0) {
    key = key.substr(5);
  }
  std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });

  static const std::unordered_map<
      std::string, at::ScalarType, TransparentHash, std::equal_to<>>
      type_map = {
          {"float", at::kFloat},
          {"float32", at::kFloat},
          {"fp32", at::kFloat},
          {"double", at::kDouble},
          {"float64", at::kDouble},
          {"fp64", at::kDouble},
          {"half", at::kHalf},
          {"float16", at::kHalf},
          {"fp16", at::kHalf},
          {"bfloat16", at::kBFloat16},
          {"bf16", at::kBFloat16},
          {"int", at::kInt},
          {"int32", at::kInt},
          {"int64", at::kLong},
          {"long", at::kLong},
          {"int16", at::kShort},
          {"short", at::kShort},
          {"int8", at::kChar},
          {"char", at::kChar},
          {"uint8", at::kByte},
          {"byte", at::kByte},
          {"bool", at::kBool},
          {"complex64", at::kComplexFloat},
          {"complex128", at::kComplexDouble},
      };

  auto iter = type_map.find(key);
  if (iter == type_map.end()) {
    throw std::invalid_argument("Unsupported type: " + type_str);
  }
  return iter->second;
}

static auto
parse_verbosity_level(const std::string& val) -> VerbosityLevel
{
  using enum VerbosityLevel;
  const int level = std::stoi(val);
  switch (level) {
    case 0:
      return Silent;
    case 1:
      return Info;
    case 2:
      return Stats;
    case 3:
      return Debug;
    case 4:
      return Trace;
    default:
      throw std::invalid_argument("Invalid verbosity level: " + val);
  }
}

}  // namespace

auto
load_config(const std::string& path) -> RuntimeConfig
{
  RuntimeConfig cfg;
  try {
    YAML::Node root = YAML::LoadFile(path);

    if (root["scheduler"]) {
      cfg.scheduler = root["scheduler"].as<std::string>();
    }
    if (root["model"]) {
      cfg.model_path = root["model"].as<std::string>();
    }
    if (root["iterations"]) {
      cfg.iterations = root["iterations"].as<int>();
    }
    if (root["device_ids"]) {
      cfg.device_ids = root["device_ids"].as<std::vector<int>>();
      if (!cfg.device_ids.empty()) {
        cfg.use_cuda = true;
      }
    }
    if (root["input"]) {
      for (const auto& node : root["input"]) {
        TensorConfig t{};
        if (node["name"]) {
          t.name = node["name"].as<std::string>();
        }
        if (node["dims"]) {
          t.dims = node["dims"].as<std::vector<int64_t>>();
        }
        if (node["data_type"]) {
          t.type = parse_type_string(node["data_type"].as<std::string>());
        }
        cfg.inputs.push_back(std::move(t));
      }
    }
    if (root["output"]) {
      for (const auto& node : root["output"]) {
        TensorConfig t{};
        if (node["name"]) {
          t.name = node["name"].as<std::string>();
        }
        if (node["dims"]) {
          t.dims = node["dims"].as<std::vector<int64_t>>();
        }
        if (node["data_type"]) {
          t.type = parse_type_string(node["data_type"].as<std::string>());
        }
        cfg.outputs.push_back(std::move(t));
      }
    }
    if (root["verbose"]) {
      cfg.verbosity = parse_verbosity_level(root["verbose"].as<std::string>());
    } else if (root["verbosity"]) {
      cfg.verbosity =
          parse_verbosity_level(root["verbosity"].as<std::string>());
    }
    if (root["delay"]) {
      cfg.delay_ms = root["delay"].as<int>();
    }
    if (root["address"]) {
      cfg.server_address = root["address"].as<std::string>();
    }
    if (root["metrics_port"]) {
      cfg.metrics_port = root["metrics_port"].as<int>();
    }
    if (root["max_batch_size"]) {
      cfg.max_batch_size = root["max_batch_size"].as<int>();
      if (cfg.max_batch_size <= 0) {
        throw std::invalid_argument("max_batch_size must be > 0");
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
    cfg.max_message_bytes =
        compute_max_message_bytes(cfg.max_batch_size, cfg.inputs);
  }
  catch (const std::exception& e) {
    log_error(std::string("Failed to load config: ") + e.what());
    cfg.valid = false;
  }
  return cfg;
}

}  // namespace starpu_server
