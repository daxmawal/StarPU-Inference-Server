#pragma once

#include <yaml-cpp/yaml.h>

#include <format>
#include <limits>
#include <stdexcept>
#include <string_view>

#include "config_loader_helpers.hpp"

namespace starpu_server::inline config_loader_detail {

auto
parse_positive_libtorch_thread_count(
    const YAML::Node& libtorch_node, std::string_view key) -> int
{
  const auto value =
      parse_scalar<long long>(libtorch_node[key], key, "an integer");
  if (value <= 0 || value > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(
        std::format("libtorch.{} must be > 0 and fit in int", key));
  }
  return static_cast<int>(value);
}

void
parse_libtorch(const YAML::Node& root, RuntimeConfig& cfg)
{
  const YAML::Node libtorch_node = root["libtorch"];
  if (!libtorch_node) {
    return;
  }
  if (!libtorch_node.IsMap()) {
    throw std::invalid_argument("libtorch must be a mapping");
  }

  if (libtorch_node["intraop_threads"]) {
    cfg.libtorch.intraop_threads =
        parse_positive_libtorch_thread_count(libtorch_node, "intraop_threads");
  }
  if (libtorch_node["interop_threads"]) {
    cfg.libtorch.interop_threads =
        parse_positive_libtorch_thread_count(libtorch_node, "interop_threads");
  }
}

}  // namespace starpu_server::inline config_loader_detail
