#pragma once
#include <string>

// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
#include <yaml-cpp/yaml.h>

#include <functional>
#endif  // SONAR_IGNORE_END
// GCOVR_EXCL_STOP

#include "runtime_config.hpp"

namespace starpu_server {

auto load_config(const std::string& path) -> RuntimeConfig;

// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
using ConfigLoaderPostParseHook = std::function<void(RuntimeConfig&)>;
void set_config_loader_post_parse_hook(ConfigLoaderPostParseHook hook);
void reset_config_loader_post_parse_hook();
auto parse_tensor_nodes_for_test(
    const YAML::Node& nodes, std::size_t max_inputs, std::string_view label,
    std::size_t max_dims) -> std::vector<TensorConfig>;
void parse_congestion_horizons_for_test(
    const YAML::Node& congestion_node, RuntimeConfig& cfg);
#endif  // SONAR_IGNORE_END
// GCOVR_EXCL_STOP

}  // namespace starpu_server
