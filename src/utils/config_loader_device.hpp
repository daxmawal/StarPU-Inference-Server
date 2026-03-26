#pragma once

#include <yaml-cpp/yaml.h>

#include <format>
#include <stdexcept>
#include <unordered_set>

#include "config_loader_helpers.hpp"

namespace starpu_server::inline config_loader_detail {

void
ensure_no_root_device_ids(const YAML::Node& root)
{
  if (root["device_ids"]) {
    throw std::invalid_argument(
        "device_ids must be nested inside the use_cuda block (e.g. "
        "\"use_cuda: [{ device_ids: [0] }]\")");
  }
}

void
parse_device_flags(const YAML::Node& root, RuntimeConfig& cfg)
{
  if (root["use_cpu"]) {
    cfg.devices.use_cpu =
        parse_scalar<bool>(root["use_cpu"], "use_cpu", "a boolean");
  }
  if (root["group_cpu_by_numa"]) {
    cfg.devices.group_cpu_by_numa = parse_scalar<bool>(
        root["group_cpu_by_numa"], "group_cpu_by_numa", "a boolean");
  }
  if (root["gpu_model_replication"]) {
    const auto policy = parse_scalar<std::string>(
        root["gpu_model_replication"], "gpu_model_replication",
        "a scalar string");
    cfg.devices.gpu_model_replication =
        parse_gpu_model_replication_policy(policy);
  }
}

void
handle_scalar_use_cuda(const YAML::Node& use_cuda_node, RuntimeConfig& cfg)
{
  const bool enabled =
      parse_scalar<bool>(use_cuda_node, "use_cuda", "a boolean");
  if (!enabled) {
    cfg.devices.use_cuda = false;
    cfg.devices.ids.clear();
    return;
  }
  throw std::invalid_argument(
      "use_cuda must be a sequence of device mappings when enabled (e.g. "
      "\"use_cuda: [{ device_ids: [0] }]\")");
}

void
append_device_ids_from_entry(
    const YAML::Node& device_ids_node, std::size_t entry_index,
    std::unordered_set<int>& seen_device_ids, RuntimeConfig& cfg)
{
  for (std::size_t device_index = 0; device_index < device_ids_node.size();
       ++device_index) {
    const auto& id_node = device_ids_node[device_index];
    const auto id_path =
        std::format("use_cuda[{}].device_ids[{}]", entry_index, device_index);
    const int device_id = parse_scalar<int>(id_node, id_path, "an integer");
    if (device_id < 0) {
      throw std::invalid_argument(std::format("{} must be >= 0", id_path));
    }
    if (!seen_device_ids.insert(device_id).second) {
      throw std::invalid_argument(std::format("{} is duplicated", id_path));
    }
    cfg.devices.ids.push_back(device_id);
  }
}

void
parse_use_cuda_entry(
    const YAML::Node& entry, std::size_t entry_index,
    std::unordered_set<int>& seen_device_ids, RuntimeConfig& cfg)
{
  if (!entry.IsMap()) {
    throw std::invalid_argument(std::format(
        "use_cuda[{}] must be a mapping that defines device_ids", entry_index));
  }

  const YAML::Node device_ids_node = entry["device_ids"];
  if (!device_ids_node) {
    throw std::invalid_argument(
        std::format("use_cuda[{}].device_ids is required", entry_index));
  }
  if (!device_ids_node.IsSequence()) {
    throw std::invalid_argument(std::format(
        "use_cuda[{}].device_ids must be a sequence of integers", entry_index));
  }

  append_device_ids_from_entry(
      device_ids_node, entry_index, seen_device_ids, cfg);
}

void
parse_use_cuda_sequence(const YAML::Node& use_cuda_node, RuntimeConfig& cfg)
{
  if (!use_cuda_node.IsSequence()) {
    throw std::invalid_argument(
        "use_cuda must be a boolean or a sequence of device mappings");
  }

  cfg.devices.use_cuda = true;
  cfg.devices.ids.clear();
  std::unordered_set<int> seen_device_ids;
  if (use_cuda_node.size() == 0U) {
    throw std::invalid_argument(
        "use_cuda requires at least one device_ids "
        "entry");
  }

  for (std::size_t entry_index = 0; entry_index < use_cuda_node.size();
       ++entry_index) {
    parse_use_cuda_entry(
        use_cuda_node[entry_index], entry_index, seen_device_ids, cfg);
  }

  if (cfg.devices.ids.empty()) {
    throw std::invalid_argument(
        "use_cuda requires at least one device_ids "
        "entry");
  }
}

void
parse_device_nodes(const YAML::Node& root, RuntimeConfig& cfg)
{
  ensure_no_root_device_ids(root);
  parse_device_flags(root, cfg);

  const YAML::Node use_cuda_node = root["use_cuda"];
  if (!use_cuda_node) {
    return;
  }
  if (use_cuda_node.IsScalar()) {
    handle_scalar_use_cuda(use_cuda_node, cfg);
    return;
  }
  parse_use_cuda_sequence(use_cuda_node, cfg);
}

}  // namespace starpu_server::inline config_loader_detail
