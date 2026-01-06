#pragma once
#include <string>

// GCOVR_EXCL_START
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
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
#endif  // SONAR_IGNORE_END
// GCOVR_EXCL_STOP

}  // namespace starpu_server
