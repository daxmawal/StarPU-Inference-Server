#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <optional>
#include <vector>

namespace starpu_server {

struct LatencyStatistics {
  double p100;
  double p95;
  double p85;
  double p50;
  double mean;
};

[[nodiscard]] inline auto
compute_latency_statistics(std::vector<double> latencies)
    -> std::optional<LatencyStatistics>
{
  if (latencies.empty()) {
    return std::nullopt;
  }

  std::ranges::sort(latencies);
  const auto size = latencies.size();

  const auto percentile_value = [&latencies, size](const double percentile) {
    if (percentile <= 0.0) {
      return latencies.front();
    }
    if (percentile >= 100.0) {
      return latencies.back();
    }

    const double position =
        (percentile / 100.0) * static_cast<double>(size - 1U);
    const auto lower_index = static_cast<std::size_t>(std::floor(position));
    const auto upper_index = static_cast<std::size_t>(std::ceil(position));
    if (lower_index == upper_index) {
      return latencies[lower_index];
    }

    const double fraction = position - static_cast<double>(lower_index);
    return std::lerp(latencies[lower_index], latencies[upper_index], fraction);
  };

  const double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);

  return LatencyStatistics{
      .p100 = latencies.back(),
      .p95 = percentile_value(95.0),
      .p85 = percentile_value(85.0),
      .p50 = percentile_value(50.0),
      .mean = sum / static_cast<double>(size),
  };
}

}  // namespace starpu_server
