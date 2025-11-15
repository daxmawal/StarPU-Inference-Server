#pragma once
#include <functional>
#include <string>

#include "runtime_config.hpp"

namespace starpu_server {

auto load_config(const std::string& path) -> RuntimeConfig;

using ConfigLoaderPostParseHook = std::function<void(RuntimeConfig&)>;
void set_config_loader_post_parse_hook(ConfigLoaderPostParseHook hook);
void reset_config_loader_post_parse_hook();

}  // namespace starpu_server
