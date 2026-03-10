#include "warmup.hpp"

#include <torch/torch.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <format>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
#include <optional>
#endif  // SONAR_IGNORE_END

#include "batching_trace_logger.hpp"
#include "client_utils.hpp"
#include "inference_queue.hpp"
#include "logger.hpp"
#include "runtime_config.hpp"
#include "starpu_setup.hpp"
#include "starpu_task_worker.hpp"
#include "utils/monotonic_clock.hpp"

namespace starpu_server {
inline namespace warmup_detail {
constexpr int kCpuWarmupDeviceId = std::numeric_limits<int>::min();
constexpr auto kWarmupDrainTimeout = std::chrono::seconds(30);
constexpr auto kWarmupDrainWaitStep = std::chrono::milliseconds(250);

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
struct WarmupHookState {
  std::mutex mutex;
  std::function<void()> server_thread_hook;
  std::function<void()> client_thread_hook;
  std::optional<std::chrono::milliseconds> drain_timeout_override;
  std::optional<std::chrono::milliseconds> drain_wait_step_override;
};

auto
warmup_hook_state() -> WarmupHookState&
{
  static WarmupHookState state;
  return state;
}

auto
resolve_warmup_drain_timeout_for_test() -> std::chrono::steady_clock::duration
{
  auto& state = warmup_hook_state();
  std::lock_guard lock(state.mutex);
  if (state.drain_timeout_override.has_value()) {
    return std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        *state.drain_timeout_override);
  }
  return std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      kWarmupDrainTimeout);
}

auto
resolve_warmup_drain_wait_step_for_test() -> std::chrono::steady_clock::duration
{
  auto& state = warmup_hook_state();
  std::lock_guard lock(state.mutex);
  if (state.drain_wait_step_override.has_value()) {
    return std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        *state.drain_wait_step_override);
  }
  return std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      kWarmupDrainWaitStep);
}
#endif  // SONAR_IGNORE_END

auto
collect_device_workers(const RuntimeConfig& opts)
    -> std::map<int, std::vector<int>>
{
  std::map<int, std::vector<int>> workers;

  if (opts.devices.use_cuda) {
    const auto device_workers =
        StarPUSetup::get_cuda_workers_by_device(opts.devices.ids);
    for (const auto& [device_id, worker_ids] : device_workers) {
      if (worker_ids.empty()) {
        continue;
      }
      auto& destination = workers[device_id];
      destination.reserve(destination.size() + worker_ids.size());
      destination.insert(
          destination.end(), worker_ids.begin(), worker_ids.end());
    }
  }

  if (opts.devices.use_cpu) {
    const auto cpu_workers =
        StarPUSetup::get_worker_ids_by_type(STARPU_CPU_WORKER);
    if (!cpu_workers.empty()) {
      auto& destination = workers[kCpuWarmupDeviceId];
      destination.reserve(destination.size() + cpu_workers.size());
      destination.insert(
          destination.end(), cpu_workers.begin(), cpu_workers.end());
    }
  }

  return workers;
}

}  // namespace warmup_detail
// =============================================================================
// Constructor
// =============================================================================

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
namespace testing {
auto
collect_device_workers_for_test(const RuntimeConfig& opts)
    -> std::map<int, std::vector<int>>
{
  return collect_device_workers(opts);
}

auto
set_warmup_server_thread_hook(std::function<void()> hook)
    -> std::function<void()>
{
  auto& state = warmup_hook_state();
  std::lock_guard lock(state.mutex);
  auto previous = std::move(state.server_thread_hook);
  state.server_thread_hook = std::move(hook);
  return previous;
}

auto
set_warmup_client_thread_hook(std::function<void()> hook)
    -> std::function<void()>
{
  auto& state = warmup_hook_state();
  std::lock_guard lock(state.mutex);
  auto previous = std::move(state.client_thread_hook);
  state.client_thread_hook = std::move(hook);
  return previous;
}

auto
set_warmup_drain_timeout_for_test(
    std::optional<std::chrono::milliseconds> timeout)
    -> std::optional<std::chrono::milliseconds>
{
  auto& state = warmup_hook_state();
  std::lock_guard lock(state.mutex);
  auto previous = state.drain_timeout_override;
  state.drain_timeout_override = timeout;
  return previous;
}

auto
set_warmup_drain_wait_step_for_test(
    std::optional<std::chrono::milliseconds> wait_step)
    -> std::optional<std::chrono::milliseconds>
{
  auto& state = warmup_hook_state();
  std::lock_guard lock(state.mutex);
  auto previous = state.drain_wait_step_override;
  state.drain_wait_step_override = wait_step;
  return previous;
}

auto
take_warmup_server_thread_hook() -> std::function<void()>
{
  auto& state = warmup_hook_state();
  std::lock_guard lock(state.mutex);
  return std::exchange(state.server_thread_hook, {});
}

auto
take_warmup_client_thread_hook() -> std::function<void()>
{
  auto& state = warmup_hook_state();
  std::lock_guard lock(state.mutex);
  return std::exchange(state.client_thread_hook, {});
}
}  // namespace testing
#endif  // SONAR_IGNORE_END

WarmupRunner::WarmupRunner(
    const RuntimeConfig& opts, StarPUSetup& starpu,
    torch::jit::script::Module& model_cpu,
    std::vector<torch::jit::script::Module>& models_gpu,
    const std::vector<torch::Tensor>& outputs_ref,
    CompletionObserver completion_observer)
    : opts_(opts), starpu_(starpu), model_cpu_(model_cpu),
      models_gpu_(models_gpu), outputs_ref_(outputs_ref),
      completion_observer_(std::move(completion_observer))
{
}

// =============================================================================
// Client thread: generate and enqueue jobs for warmup
// =============================================================================

auto
WarmupRunner::client_worker(
    const std::map<int, std::vector<int>>& device_workers,
    InferenceQueue& queue, int request_nb_per_worker) const -> std::size_t
{
  if (request_nb_per_worker < 0) {
    throw std::invalid_argument("request_nb_per_worker must be non-negative");
  }

  thread_local std::mt19937 rng;
  if (opts_.seed.has_value()) {
    rng.seed(*opts_.seed);
    torch::manual_seed(*opts_.seed);
  } else {
    rng.seed(std::random_device{}());
  }

  auto pregen_inputs = client_utils::pre_generate_inputs(
      opts_, opts_.batching.warmup_pregen_inputs);

  const size_t worker_count = std::accumulate(
      device_workers.begin(), device_workers.end(), std::size_t{0},
      [](std::size_t sum, const auto& pair) {
        return sum + pair.second.size();
      });
  const size_t total_size_t =
      worker_count * static_cast<std::size_t>(request_nb_per_worker);

  if (total_size_t > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("Total exceeds int capacity");
  }

  const auto total = static_cast<int>(total_size_t);
  int request_id = 0;
  std::size_t enqueued_jobs = 0;
  const auto& model_name = opts_.model ? opts_.model->name : opts_.name;

  std::vector<int> worker_ids_flat;
  worker_ids_flat.reserve(worker_count);
  for (const auto& [device_id, device_worker_ids] : device_workers) {
    (void)device_id;
    worker_ids_flat.insert(
        worker_ids_flat.end(), device_worker_ids.begin(),
        device_worker_ids.end());
  }

  for (const int worker_id : worker_ids_flat) {
    for (auto request_nb = 0; request_nb < request_nb_per_worker;
         ++request_nb) {
      const auto& inputs = client_utils::pick_random_input(pregen_inputs, rng);
      auto job = client_utils::create_job(
          inputs, outputs_ref_, request_id, {}, {}, model_name);
      const int job_request_id = request_id;
      job->set_fixed_worker_id(worker_id);

      const auto enqueued_now = MonotonicClock::now();
      job->update_timing_info([enqueued_now](detail::TimingInfo& timing) {
        timing.enqueued_time = enqueued_now;
        timing.last_enqueued_time = enqueued_now;
      });

      if (bool queue_full = false; !queue.push(std::move(job), &queue_full)) {
        const auto* const reason =
            queue_full ? "queue is full" : "queue shutting down";
        log_warning(std::format(
            "[Warmup] Failed to enqueue job {}: {}", job_request_id, reason));
        queue.shutdown();
        return enqueued_jobs;
      }
      const auto log_now = std::chrono::system_clock::now();
      client_utils::log_job_enqueued(opts_, job_request_id, total, log_now);
      request_id++;
      ++enqueued_jobs;
    }
  }

  queue.shutdown();
  return enqueued_jobs;
}

inline namespace warmup_detail {

struct WarmupSyncState {
  std::atomic<std::size_t> completed_jobs{0};
  std::atomic<std::size_t> enqueued_jobs{0};
  std::atomic<bool> client_done{false};
  std::mutex completed_mutex;
  std::condition_variable completed_cv;
  std::exception_ptr thread_exception;
  std::mutex thread_exception_mutex;

  void store_exception(const std::exception_ptr& exception)
  {
    std::lock_guard lock(thread_exception_mutex);
    if (!thread_exception) {
      thread_exception = exception;
    }
  }

  auto load_exception() -> std::exception_ptr
  {
    std::lock_guard lock(thread_exception_mutex);
    return thread_exception;
  }

  void notify_exception(
      const std::exception_ptr& exception, InferenceQueue& queue)
  {
    store_exception(exception);
    queue.shutdown();
    completed_cv.notify_all();
  }

  void set_client_enqueued_jobs(std::size_t value)
  {
    enqueued_jobs.store(value, std::memory_order_release);
    client_done.store(true, std::memory_order_release);
    completed_cv.notify_all();
  }
};

template <typename Fn, typename Notify>
void
run_warmup_server_thread(Fn&& task_fn, Notify&& notify_exception)
{
  try {
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
    if (auto hook = testing::take_warmup_server_thread_hook()) {
      hook();
    }
#endif  // SONAR_IGNORE_END
    std::forward<Fn>(task_fn)();
  }
  catch (...) {
    notify_exception(std::current_exception());
  }
}

template <typename Fn, typename Notify>
void
run_warmup_client_thread(Fn&& task_fn, Notify&& notify_exception)
{
  try {
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
    if (auto hook = testing::take_warmup_client_thread_hook()) {
      hook();
    }
#endif  // SONAR_IGNORE_END
    std::forward<Fn>(task_fn)();
  }
  catch (...) {
    notify_exception(std::current_exception());
  }
}

struct WarmupProgress {
  std::size_t completed = 0;
  std::size_t enqueued = 0;
};

auto
read_warmup_progress(const WarmupSyncState& state) -> WarmupProgress
{
  return WarmupProgress{
      .completed = state.completed_jobs.load(std::memory_order_acquire),
      .enqueued = state.enqueued_jobs.load(std::memory_order_acquire)};
}

auto
remaining_warmup_jobs(const WarmupProgress& progress) -> std::size_t
{
  return progress.enqueued > progress.completed
             ? progress.enqueued - progress.completed
             : 0;
}

auto
make_warmup_timeout_exception(
    const WarmupProgress& progress, bool client_done,
    long long timeout_ms) -> std::exception_ptr
{
  const auto remaining = remaining_warmup_jobs(progress);
  log_error(std::format(
      "Warmup drain timeout after {} ms: completed={} enqueued={} "
      "remaining={} client_done={}",
      timeout_ms, progress.completed, progress.enqueued, remaining,
      client_done));
  return std::make_exception_ptr(std::runtime_error(std::format(
      "Warmup drain timeout: completed={} enqueued={} remaining={} "
      "client_done={}",
      progress.completed, progress.enqueued, remaining, client_done)));
}

auto
advance_warmup_wait_iteration(
    WarmupSyncState& state, std::unique_lock<std::mutex>& lock,
    std::chrono::steady_clock::time_point deadline,
    std::chrono::steady_clock::duration drain_wait_step, long long timeout_ms,
    std::exception_ptr& wait_exception) -> bool
{
  if (state.load_exception()) {
    return true;
  }

  const bool client_done = state.client_done.load(std::memory_order_acquire);
  const auto progress = read_warmup_progress(state);
  if (client_done && progress.completed >= progress.enqueued) {
    return true;
  }

  const auto now = std::chrono::steady_clock::now();
  if (now >= deadline) {
    wait_exception =
        make_warmup_timeout_exception(progress, client_done, timeout_ms);
    return true;
  }

  const auto until_deadline = deadline - now;
  const auto wait_budget =
      until_deadline < drain_wait_step ? until_deadline : drain_wait_step;
  static_cast<void>(state.completed_cv.wait_for(lock, wait_budget));
  return false;
}

auto
wait_for_warmup_completion(
    WarmupSyncState& state,
    const WarmupRunner::CompletionObserver& completion_observer)
    -> std::exception_ptr
{
  std::exception_ptr wait_exception;
  std::unique_lock lock(state.completed_mutex);
  const auto drain_timeout =
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
      resolve_warmup_drain_timeout_for_test();
#else
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          kWarmupDrainTimeout);
#endif  // SONAR_IGNORE_END
  const auto drain_wait_step =
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
      resolve_warmup_drain_wait_step_for_test();
#else
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          kWarmupDrainWaitStep);
#endif  // SONAR_IGNORE_END
  const auto deadline = std::chrono::steady_clock::now() + drain_timeout;
  const auto timeout_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(drain_timeout)
          .count();
  try {
    bool done = false;
    while (!done) {
      if (completion_observer) {
        completion_observer(state.completed_jobs);
      }
      done = advance_warmup_wait_iteration(
          state, lock, deadline, drain_wait_step, timeout_ms, wait_exception);
    }
  }
  catch (...) {
    wait_exception = std::current_exception();
  }
  return wait_exception;
}

}  // namespace warmup_detail

// =============================================================================
// Warmup execution: launch server and client threads and wait for completion
// =============================================================================

void
WarmupRunner::run(int request_nb_per_worker)
{
  if (request_nb_per_worker < 0) {
    throw std::invalid_argument("request_nb_per_worker must be non-negative");
  }

  if (opts_.batching.warmup_pregen_inputs == 0) {
    log_info(
        opts_.verbosity,
        "Warmup skipped because warmup_pregen_inputs is set to 0.");
    return;
  }

  auto& tracer = BatchingTraceLogger::instance();
  auto suppression_guard = tracer.scoped_warmup_suppression(true);

  auto device_workers = collect_device_workers(opts_);
  if (device_workers.empty()) {
    log_info(
        opts_.verbosity,
        "Warmup skipped because no eligible workers were detected.");
    return;
  }

  const std::size_t warmup_queue_limit = opts_.batching.max_queue_size;
  std::size_t warmup_inflight_limit = warmup_queue_limit;
  if (opts_.batching.max_inflight_tasks > 0 &&
      opts_.batching.max_inflight_tasks < warmup_queue_limit) {
    warmup_inflight_limit = opts_.batching.max_inflight_tasks;
  }

  if (opts_.batching.max_inflight_tasks == 0 ||
      opts_.batching.max_inflight_tasks > warmup_queue_limit) {
    log_info(
        opts_.verbosity,
        std::format(
            "Warmup guardrail enabled: max_inflight_tasks={} (queue cap={}).",
            warmup_inflight_limit, warmup_queue_limit));
  }

  InferenceQueue queue(warmup_queue_limit);
  WarmupSyncState sync_state;
  const auto notify_thread_exception =
      [&sync_state, &queue](const std::exception_ptr& exception) {
        sync_state.notify_exception(exception, queue);
      };

  StarPUTaskRunnerConfig config{};
  config.queue = &queue;
  config.model_cpu = &model_cpu_;
  config.models_gpu = &models_gpu_;
  config.starpu = &starpu_;
  RuntimeConfig warmup_opts = opts_;
  warmup_opts.batching.trace_enabled = false;
  warmup_opts.batching.max_inflight_tasks = warmup_inflight_limit;
  warmup_opts.batching.max_queue_size = warmup_queue_limit;
  config.opts = &warmup_opts;
  config.completed_jobs = &sync_state.completed_jobs;
  config.all_done_cv = &sync_state.completed_cv;
  StarPUTaskRunner worker(config);

  std::jthread server([&]() {
    run_warmup_server_thread(
        [&worker]() { worker.run(); }, notify_thread_exception);
  });

  std::jthread client([this, &device_workers, &queue, &notify_thread_exception,
                       &sync_state, request_nb_per_worker]() {
    run_warmup_client_thread(
        [this, &device_workers, &queue, &sync_state, request_nb_per_worker]() {
          const auto enqueued_jobs =
              client_worker(device_workers, queue, request_nb_per_worker);
          sync_state.set_client_enqueued_jobs(enqueued_jobs);
        },
        notify_thread_exception);
  });

  auto wait_exception =
      wait_for_warmup_completion(sync_state, completion_observer_);
  if (wait_exception || sync_state.load_exception()) {
    queue.shutdown();
  }

  if (client.joinable()) {
    client.join();
  }
  if (server.joinable()) {
    server.join();
  }

  starpu_task_wait_for_all();

  auto thread_exception_copy = sync_state.load_exception();
  if (wait_exception || thread_exception_copy) {
    std::rethrow_exception(
        wait_exception ? wait_exception : thread_exception_copy);
  }
}
}  // namespace starpu_server
