#include "perf_observer.hpp"

#include <mutex>

namespace starpu_server::perf_observer {
namespace {
std::mutex g_mutex;
std::optional<std::chrono::high_resolution_clock::time_point> g_first_enqueue;
std::optional<std::chrono::high_resolution_clock::time_point> g_last_completion;
std::size_t g_total_inferences = 0;
}

void
reset()
{
  const std::scoped_lock lock(g_mutex);
  g_first_enqueue.reset();
  g_last_completion.reset();
  g_total_inferences = 0;
}

void
record_job(
    const std::chrono::high_resolution_clock::time_point enqueue_time,
    const std::chrono::high_resolution_clock::time_point completion_time,
    const std::size_t batch_size,
    const bool is_warmup_job)
{
  if (is_warmup_job) {
    return;
  }

  if (batch_size == 0) {
    return;
  }

  const std::scoped_lock lock(g_mutex);
  if (!g_first_enqueue || enqueue_time < *g_first_enqueue) {
    g_first_enqueue = enqueue_time;
  }
  if (!g_last_completion || completion_time > *g_last_completion) {
    g_last_completion = completion_time;
  }
  g_total_inferences += batch_size;
}

auto
snapshot() -> std::optional<Snapshot>
{
  const std::scoped_lock lock(g_mutex);
  if (!g_first_enqueue || !g_last_completion || g_total_inferences == 0) {
    return std::nullopt;
  }

  if (*g_last_completion <= *g_first_enqueue) {
    return std::nullopt;
  }

  const double duration_seconds = std::chrono::duration<double>(
                                     *g_last_completion - *g_first_enqueue)
                                     .count();
  if (duration_seconds <= 0.0) {
    return std::nullopt;
  }

  const double throughput =
      static_cast<double>(g_total_inferences) / duration_seconds;
  return Snapshot{g_total_inferences, duration_seconds, throughput};
}

}  // namespace starpu_server::perf_observer

