#pragma once
#include <string>

#include "runtime_config.hpp"

namespace starpu_server {

auto load_config(const std::string& path) -> RuntimeConfig;

}  // namespace starpu_server
