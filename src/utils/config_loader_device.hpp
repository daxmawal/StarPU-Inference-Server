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
  std::unordered_set<int> seen_device_ids;
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
      if (!seen_device_ids.insert(device_id).second) {
        throw std::invalid_argument(std::format("{} is duplicated", id_path));
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
