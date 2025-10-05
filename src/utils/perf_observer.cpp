#include "perf_observer.hpp"

#include <mutex>

namespace starpu_server::perf_observer {
namespace {
struct State {
  std::mutex mutex;
  std::optional<std::chrono::high_resolution_clock::time_point> first_enqueue;
  std::optional<std::chrono::high_resolution_clock::time_point> last_completion;
  std::size_t total_inferences = 0;
};

auto
state() -> State&
{
  static State instance;
  return instance;
}
}  // namespace

void
reset()
{
  auto& state_data = state();
  const std::scoped_lock lock(state_data.mutex);
  state_data.first_enqueue.reset();
  state_data.last_completion.reset();
  state_data.total_inferences = 0;
}

void
record_job(
    const std::chrono::high_resolution_clock::time_point enqueue_time,
    const std::chrono::high_resolution_clock::time_point completion_time,
    const std::size_t batch_size, const bool is_warmup_job)
{
  if (is_warmup_job) {
    return;
  }

  if (batch_size == 0) {
    return;
  }

  auto& state_data = state();
  const std::scoped_lock lock(state_data.mutex);
  if (!state_data.first_enqueue || enqueue_time < *state_data.first_enqueue) {
    state_data.first_enqueue = enqueue_time;
  }
  if (!state_data.last_completion ||
      completion_time > *state_data.last_completion) {
    state_data.last_completion = completion_time;
  }
  state_data.total_inferences += batch_size;
}

auto
snapshot() -> std::optional<Snapshot>
{
  auto& state_data = state();
  const std::scoped_lock lock(state_data.mutex);
  if (!state_data.first_enqueue || !state_data.last_completion ||
      state_data.total_inferences == 0) {
    return std::nullopt;
  }

  if (*state_data.last_completion <= *state_data.first_enqueue) {
    return std::nullopt;
  }

  const double duration_seconds =
      std::chrono::duration<double>(
          *state_data.last_completion - *state_data.first_enqueue)
          .count();

  const double throughput =
      static_cast<double>(state_data.total_inferences) / duration_seconds;
  return Snapshot{state_data.total_inferences, duration_seconds, throughput};
}

}  // namespace starpu_server::perf_observer
