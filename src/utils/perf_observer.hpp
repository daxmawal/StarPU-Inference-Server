#pragma once

#include <chrono>
#include <cstddef>
#include <optional>

#include "utils/monotonic_clock.hpp"

namespace starpu_server::perf_observer {

struct Snapshot {
  std::size_t total_inferences;
  double duration_seconds;
  double throughput;
};

void reset();

void record_job(
    MonotonicClock::time_point enqueue_time,
    MonotonicClock::time_point completion_time, std::size_t batch_size,
    bool is_warmup_job);

auto snapshot() -> std::optional<Snapshot>;

}  // namespace starpu_server::perf_observer
