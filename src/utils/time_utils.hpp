#pragma once

#include <chrono>
#include <string>

namespace time_utils {

auto format_timestamp(const std::chrono::high_resolution_clock::time_point& tp)
    -> std::string;

}  // namespace time_utils