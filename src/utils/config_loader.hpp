#pragma once
#include <string>

#if defined(STARPU_TESTING)
#include <functional>
#endif

#include "runtime_config.hpp"

namespace starpu_server {

auto load_config(const std::string& path) -> RuntimeConfig;

#if defined(STARPU_TESTING)
using ConfigLoaderPostParseHook = std::function<void(RuntimeConfig&)>;
void set_config_loader_post_parse_hook(ConfigLoaderPostParseHook hook);
void reset_config_loader_post_parse_hook();
#endif

}  // namespace starpu_server
