#pragma once

#include <filesystem>
#include <optional>
#include <string>

namespace starpu_server {

struct ThroughputMeasurement {
  std::string config_signature;
  double throughput_gpu = -1.0;
  double throughput_cpu = -1.0;
};

[[nodiscard]] auto load_cached_throughput_measurements(
    const std::filesystem::path& file_path)
    -> std::optional<ThroughputMeasurement>;

auto save_throughput_measurements(
    const std::filesystem::path& file_path,
    const ThroughputMeasurement& measurements) -> bool;

}  // namespace starpu_server
