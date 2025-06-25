#pragma once

#include <chrono>
#include <string>

<<<<<<< pipeline_ci
namespace starpu_server::time_utils {
=======
namespace starpu_server { namespace time_utils {
>>>>>>> main

auto format_timestamp(const std::chrono::high_resolution_clock::time_point&
                          time_point) -> std::string;

<<<<<<< pipeline_ci
}  // namespace starpu_server::time_utils
=======
}}  // namespace starpu_server::time_utils
>>>>>>> main
