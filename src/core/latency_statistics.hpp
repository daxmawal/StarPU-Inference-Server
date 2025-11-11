#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numeric>
#include <optional>
#include <span>
#include <vector>

namespace starpu_server {

inline constexpr double kPercentileP95 = 95.0;
inline constexpr double kPercentileP85 = 85.0;
inline constexpr double kPercentileP50 = 50.0;

struct LatencyStatistics {
  double p0;
  double p100;
  double p95;
  double p85;
  double p50;
  double mean;
};

template <typename Sample, typename Projection = std::identity>
[[nodiscard]] inline auto
compute_latency_statistics(
    std::span<const Sample> samples,
    Projection projection = Projection{}) -> std::optional<LatencyStatistics>
{
  if (samples.empty()) {
    return std::nullopt;
  }

  const std::size_t size = samples.size();
  std::vector<std::size_t> order(size);
  std::iota(order.begin(), order.end(), std::size_t{0});

  const auto projected_value =
      [&](std::size_t index) noexcept(noexcept(static_cast<double>(
          std::invoke(projection, samples[index])))) -> double {
    return static_cast<double>(std::invoke(projection, samples[index]));
  };

  const auto comparator =
      [&](std::size_t lhs, std::size_t rhs) noexcept(
          noexcept(projected_value(lhs) < projected_value(rhs))) {
        return projected_value(lhs) < projected_value(rhs);
      };
  std::ranges::sort(order, comparator);

  const auto sorted_value = [&](std::size_t rank) -> double {
    return projected_value(order[rank]);
  };

  const auto percentile_value = [&](double percentile) -> double {
    if (percentile <= 0.0) {
      return sorted_value(0);
    }
    if (percentile >= 100.0) {
      return sorted_value(size - 1U);
    }

    const double position =
        (percentile / 100.0) * static_cast<double>(size - 1U);
    const auto lower_index = static_cast<std::size_t>(std::floor(position));
    const auto upper_index = static_cast<std::size_t>(std::ceil(position));
    if (lower_index == upper_index) {
      return sorted_value(lower_index);
    }

    const double fraction = position - static_cast<double>(lower_index);
    return std::lerp(
        sorted_value(lower_index), sorted_value(upper_index), fraction);
  };

  const double sum = std::accumulate(
      samples.begin(), samples.end(), 0.0,
      [&](double acc, const Sample& sample) {
        return acc + static_cast<double>(std::invoke(projection, sample));
      });

  const double min_value = percentile_value(0.0);
  const double max_value = percentile_value(100.0);

  return LatencyStatistics{
      .p0 = min_value,
      .p100 = max_value,
      .p95 = percentile_value(kPercentileP95),
      .p85 = percentile_value(kPercentileP85),
      .p50 = percentile_value(kPercentileP50),
      .mean = sum / static_cast<double>(size),
  };
}

[[nodiscard]] inline auto
compute_latency_statistics(const std::vector<double>& latencies)
    -> std::optional<LatencyStatistics>
{
  return compute_latency_statistics(
      std::span<const double>(latencies), std::identity{});
}

}  // namespace starpu_server
