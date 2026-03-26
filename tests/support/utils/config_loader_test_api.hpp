#pragma once

#if !defined(STARPU_TESTING)
#error "config_loader_test_api.hpp is test-only and requires STARPU_TESTING"
#endif

#include <yaml-cpp/yaml.h>

#include <functional>
#include <string_view>

#include "utils/config_loader.hpp"

namespace starpu_server {

using ConfigLoaderPostParseHook = std::function<void(RuntimeConfig&)>;

void set_config_loader_post_parse_hook(ConfigLoaderPostParseHook hook);
void reset_config_loader_post_parse_hook();
auto parse_tensor_nodes_for_test(
    const YAML::Node& nodes, std::size_t max_inputs, std::string_view label,
    std::size_t max_dims) -> std::vector<TensorConfig>;
void parse_congestion_horizons_for_test(
    const YAML::Node& congestion_node, RuntimeConfig& cfg);

}  // namespace starpu_server
