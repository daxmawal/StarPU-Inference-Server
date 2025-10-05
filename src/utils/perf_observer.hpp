#pragma once

#include <chrono>
#include <cstddef>
#include <optional>

namespace starpu_server::perf_observer {

struct Snapshot {
  std::size_t total_inferences;
  double duration_seconds;
  double throughput;
};

void reset();

void record_job(
    std::chrono::high_resolution_clock::time_point enqueue_time,
    std::chrono::high_resolution_clock::time_point completion_time,
    std::size_t batch_size, bool is_warmup_job);

auto snapshot() -> std::optional<Snapshot>;

}  // namespace starpu_server::perf_observer
