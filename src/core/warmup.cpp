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
#include <thread>
#include <utility>
#include <vector>

#include "batching_trace_logger.hpp"
#include "client_utils.hpp"
#include "exceptions.hpp"
#include "inference_queue.hpp"
#include "logger.hpp"
#include "runtime_config.hpp"
#include "starpu_setup.hpp"
#include "starpu_task_worker.hpp"
#include "utils/monotonic_clock.hpp"

namespace starpu_server {
namespace {
constexpr int kCpuWarmupDeviceId = std::numeric_limits<int>::min();

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
std::mutex warmup_hook_mutex;
std::function<void()> warmup_server_thread_hook;
std::function<void()> warmup_client_thread_hook;
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

}  // namespace
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
  std::lock_guard lock(warmup_hook_mutex);
  auto previous = std::move(warmup_server_thread_hook);
  warmup_server_thread_hook = std::move(hook);
  return previous;
}

auto
set_warmup_client_thread_hook(std::function<void()> hook)
    -> std::function<void()>
{
  std::lock_guard lock(warmup_hook_mutex);
  auto previous = std::move(warmup_client_thread_hook);
  warmup_client_thread_hook = std::move(hook);
  return previous;
}

auto
take_warmup_server_thread_hook() -> std::function<void()>
{
  std::lock_guard lock(warmup_hook_mutex);
  return std::exchange(warmup_server_thread_hook, {});
}

auto
take_warmup_client_thread_hook() -> std::function<void()>
{
  std::lock_guard lock(warmup_hook_mutex);
  return std::exchange(warmup_client_thread_hook, {});
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

void
WarmupRunner::client_worker(
    const std::map<int, std::vector<int>>& device_workers,
    InferenceQueue& queue, int request_nb_per_worker) const
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
      job->timing_info().enqueued_time = enqueued_now;
      job->timing_info().last_enqueued_time = enqueued_now;

      if (bool queue_full = false; !queue.push(std::move(job), &queue_full)) {
        const auto* const reason =
            queue_full ? "queue is full" : "queue shutting down";
        log_warning(std::format(
            "[Warmup] Failed to enqueue job {}: {}", job_request_id, reason));
        queue.shutdown();
        return;
      }
      const auto log_now = std::chrono::system_clock::now();
      client_utils::log_job_enqueued(opts_, job_request_id, total, log_now);
      request_id++;
    }
  }

  queue.shutdown();
}

namespace {

struct WarmupSyncState {
  std::atomic<int> completed_jobs{0};
  std::mutex completed_mutex;
  std::condition_variable completed_cv;
  std::exception_ptr thread_exception;
  std::mutex thread_exception_mutex;

  void store_exception(std::exception_ptr exception)
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

  void notify_exception(std::exception_ptr exception, InferenceQueue& queue)
  {
    store_exception(exception);
    queue.shutdown();
    completed_cv.notify_all();
  }
};

auto
count_worker_total(const std::map<int, std::vector<int>>& device_workers)
    -> std::size_t
{
  return std::accumulate(
      device_workers.begin(), device_workers.end(), std::size_t{0},
      [](std::size_t sum, const auto& pair) {
        return sum + pair.second.size();
      });
}

template <typename Fn, typename Notify>
void
run_warmup_server_thread(Fn&& fn, Notify&& notify_exception)
{
  try {
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
    if (auto hook = testing::take_warmup_server_thread_hook()) {
      hook();
    }
#endif  // SONAR_IGNORE_END
    std::forward<Fn>(fn)();
  }
  catch (...) {
    notify_exception(std::current_exception());
  }
}

template <typename Fn, typename Notify>
void
run_warmup_client_thread(Fn&& fn, Notify&& notify_exception)
{
  try {
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
    if (auto hook = testing::take_warmup_client_thread_hook()) {
      hook();
    }
#endif  // SONAR_IGNORE_END
    std::forward<Fn>(fn)();
  }
  catch (...) {
    notify_exception(std::current_exception());
  }
}

auto
wait_for_warmup_completion(
    WarmupSyncState& state, std::size_t total_jobs,
    const WarmupRunner::CompletionObserver& completion_observer)
    -> std::exception_ptr
{
  std::exception_ptr wait_exception;
  std::unique_lock lock(state.completed_mutex);
  try {
    state.completed_cv.wait(lock, [&]() {
      if (completion_observer) {
        completion_observer(state.completed_jobs);
      }
      if (state.load_exception()) {
        return true;
      }
      int count = state.completed_jobs.load();
      if (count < 0) {
        throw InferenceExecutionException(
            "dummy_completed_jobs became negative, which should not happen.");
      }
      return static_cast<std::size_t>(count) >= total_jobs;
    });
  }
  catch (...) {
    wait_exception = std::current_exception();
  }
  return wait_exception;
}

}  // namespace

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

  InferenceQueue queue(std::numeric_limits<std::size_t>::max());
  WarmupSyncState sync_state;
  const auto notify_thread_exception = [&sync_state,
                                        &queue](std::exception_ptr exception) {
    sync_state.notify_exception(exception, queue);
  };

  StarPUTaskRunnerConfig config{};
  config.queue = &queue;
  config.model_cpu = &model_cpu_;
  config.models_gpu = &models_gpu_;
  config.starpu = &starpu_;
  RuntimeConfig warmup_opts = opts_;
  warmup_opts.batching.trace_enabled = false;
  warmup_opts.batching.max_inflight_tasks = 0;
  warmup_opts.batching.max_queue_size = std::numeric_limits<std::size_t>::max();
  config.opts = &warmup_opts;
  config.completed_jobs = &sync_state.completed_jobs;
  config.all_done_cv = &sync_state.completed_cv;
  StarPUTaskRunner worker(config);

  std::jthread server([&]() {
    run_warmup_server_thread(
        [&worker]() { worker.run(); }, notify_thread_exception);
  });

  std::jthread client([this, &device_workers, &queue, &notify_thread_exception,
                       request_nb_per_worker]() {
    run_warmup_client_thread(
        [this, &device_workers, &queue, request_nb_per_worker]() {
          client_worker(device_workers, queue, request_nb_per_worker);
        },
        notify_thread_exception);
  });

  const size_t total_worker_count = count_worker_total(device_workers);
  const size_t total_jobs =
      static_cast<size_t>(request_nb_per_worker) * total_worker_count;

  auto wait_exception =
      wait_for_warmup_completion(sync_state, total_jobs, completion_observer_);
  auto thread_exception_copy = sync_state.load_exception();

  if (wait_exception || thread_exception_copy) {
    queue.shutdown();
    if (client.joinable()) {
      client.join();
    }
    if (server.joinable()) {
      server.join();
    }
    starpu_task_wait_for_all();
    std::rethrow_exception(
        wait_exception ? wait_exception : thread_exception_copy);
  }
}
}  // namespace starpu_server
