#pragma once

#include <algorithm>
#include <optional>

#include "utils/runtime_config.hpp"

namespace starpu_server::testing {

inline auto
normalize_batch_capacity_for_tests(int batch_size) -> int
{
  return std::max(1, batch_size);
}

inline void
set_effective_batch_capacity_for_tests(RuntimeConfig& opts, int batch_size)
{
  const int normalized_batch_size =
      normalize_batch_capacity_for_tests(batch_size);
  opts.batching.resolved_max_batch_size = normalized_batch_size;
  opts.batching.adaptive.max_batch_size = normalized_batch_size;
  opts.batching.fixed.batch_size = normalized_batch_size;
  opts.batching.adaptive.min_batch_size = std::clamp(
      opts.batching.adaptive.min_batch_size, 1, normalized_batch_size);
}

inline void
configure_adaptive_batching_for_tests(
    RuntimeConfig& opts, int max_batch_size,
    std::optional<int> min_batch_size = std::nullopt)
{
  using enum BatchingStrategyKind;

  opts.batching.strategy = Adaptive;
  set_effective_batch_capacity_for_tests(opts, max_batch_size);
  if (min_batch_size.has_value()) {
    opts.batching.adaptive.min_batch_size =
        std::clamp(*min_batch_size, 1, opts.batching.adaptive.max_batch_size);
  }
}

inline void
configure_fixed_batching_for_tests(RuntimeConfig& opts, int batch_size)
{
  using enum BatchingStrategyKind;

  opts.batching.strategy = Fixed;
  set_effective_batch_capacity_for_tests(opts, batch_size);
}

inline void
configure_disabled_batching_for_tests(RuntimeConfig& opts)
{
  using enum BatchingStrategyKind;

  opts.batching.strategy = Disabled;
  set_effective_batch_capacity_for_tests(opts, 1);
}

}  // namespace starpu_server::testing
