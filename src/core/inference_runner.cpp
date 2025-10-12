#include "inference_runner.hpp"

#include <ATen/Context.h>
#include <ATen/core/ScalarType.h>
#include <c10/core/InferenceMode.h>
#include <c10/util/Exception.h>
#include <torch/torch.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <format>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "client_utils.hpp"
#include "exceptions.hpp"
#include "inference_queue.hpp"
#include "inference_validator.hpp"
#include "input_generator.hpp"
#include "latency_statistics.hpp"
#include "logger.hpp"
#include "runtime_config.hpp"
#include "starpu_setup.hpp"
#include "starpu_task_worker.hpp"
#include "utils/nvtx.hpp"
#include "warmup.hpp"

namespace starpu_server {

namespace {

class CuDnnBenchmarkGuard {
 public:
  explicit CuDnnBenchmarkGuard(bool enable)
      : active_(enable && torch::cuda::is_available())
  {
    if (active_) {
      previous_ = at::globalContext().benchmarkCuDNN();
      at::globalContext().setBenchmarkCuDNN(true);
    }
  }

  CuDnnBenchmarkGuard(const CuDnnBenchmarkGuard&) = delete;
  CuDnnBenchmarkGuard(CuDnnBenchmarkGuard&&) = default;
  auto operator=(const CuDnnBenchmarkGuard&) -> CuDnnBenchmarkGuard& = delete;
  auto operator=(CuDnnBenchmarkGuard&&) -> CuDnnBenchmarkGuard& = default;

  ~CuDnnBenchmarkGuard()
  {
    if (active_) {
      at::globalContext().setBenchmarkCuDNN(previous_);
    }
  }

 private:
  bool previous_ = false;
  bool active_ = false;
};

}  // namespace

static auto
default_worker_thread_launcher(StarPUTaskRunner& worker) -> std::jthread
{
  return std::jthread(&StarPUTaskRunner::run, &worker);
}

namespace {
inline auto
launcher_storage() -> WorkerThreadLauncher&
{
  static WorkerThreadLauncher launcher = default_worker_thread_launcher;
  return launcher;
}
}  // namespace

auto
get_worker_thread_launcher() -> WorkerThreadLauncher
{
  return launcher_storage();
}

void
set_worker_thread_launcher(WorkerThreadLauncher launcher)
{
  launcher_storage() = launcher;
}

// =============================================================================
// InferenceJob: Encapsulates a single inference task, including input data,
// types, ID, and completion callback
// =============================================================================

InferenceJob::InferenceJob(
    std::vector<torch::Tensor> inputs, std::vector<at::ScalarType> types,
    int request_identifier,
    std::function<void(const std::vector<torch::Tensor>&, double)> callback)
    : input_tensors_(std::move(inputs)), input_types_(std::move(types)),
      request_id_(request_identifier), on_complete_(std::move(callback)),
      start_time_(std::chrono::high_resolution_clock::now())
{
}

auto
InferenceJob::make_shutdown_job() -> std::shared_ptr<InferenceJob>
{
  auto job = std::make_shared<InferenceJob>();
  job->is_shutdown_signal_ = true;
  job->logical_job_count_ = 0;
  job->aggregated_sub_jobs_.clear();
  return job;
}

// =============================================================================
// Client Logic: Generates and enqueues inference jobs into the shared queue
// =============================================================================

namespace detail {
void
client_worker(
    InferenceQueue& queue, const RuntimeConfig& opts,
    const std::vector<torch::Tensor>& outputs_ref, const int request_nb)
{
  thread_local std::mt19937 rng;
  if (opts.seed) {
    rng.seed(static_cast<std::mt19937::result_type>(*opts.seed));
    torch::manual_seed(static_cast<uint64_t>(*opts.seed));
  } else {
    rng.seed(std::random_device{}());
  }

  auto pregen_inputs =
      client_utils::pre_generate_inputs(opts, opts.pregen_inputs);

  auto next_time = std::chrono::steady_clock::now();
  const auto delay = std::chrono::microseconds(opts.delay_us);
  for (auto request_id = 0; request_id < request_nb; ++request_id) {
    std::this_thread::sleep_until(next_time);
    next_time += delay;
    const auto& inputs = client_utils::pick_random_input(pregen_inputs, rng);
    auto job = client_utils::create_job(inputs, outputs_ref, request_id);
    client_utils::log_job_enqueued(
        opts, request_id, request_nb, job->timing_info().enqueued_time);
    if (!queue.push(job)) {
      log_warning(std::format(
          "[Client] Failed to enqueue job {}: queue shutting down",
          request_id));
      break;
    }
  }

  queue.shutdown();
}
}  // namespace detail

// =============================================================================
// Model Loading and Cloning to GPU
// =============================================================================

static auto
load_model(const std::string& model_path) -> torch::jit::script::Module
{
  try {
    auto model = torch::jit::load(model_path);
    model.eval();
    return model;
  }
  catch (const c10::Error& e) {
    log_error("Failed to load model: " + std::string(e.what()));
    throw;
  }
}

static auto
clone_model_to_gpus(
    const torch::jit::script::Module& model_cpu,
    const std::vector<int>& device_ids)
    -> std::vector<torch::jit::script::Module>
{
  const int device_count =
      static_cast<int>(static_cast<unsigned char>(torch::cuda::device_count()));
  for (const int device_id : device_ids) {
    if (device_id < 0 || device_id >= device_count) {
      log_error(std::format(
          "GPU ID {} out of range. Only {} device(s) available.", device_id,
          device_count));
      throw std::runtime_error("Invalid GPU device ID");
    }
  }

  std::vector<torch::jit::script::Module> models_gpu;
  models_gpu.reserve(device_ids.size());

  for (const auto& device_id : device_ids) {
    torch::jit::script::Module model_gpu = model_cpu.clone();
    model_gpu.to(
        torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(device_id)));
    models_gpu.emplace_back(std::move(model_gpu));
  }

  return models_gpu;
}

// =============================================================================
// Input Generation and Reference Inference Execution (CPU only)
// =============================================================================

static auto
generate_inputs(const std::vector<TensorConfig>& tensors)
    -> std::vector<torch::Tensor>
{
  return input_generator::generate_random_inputs(tensors);
}

static auto
run_reference_inference(
    torch::jit::script::Module& model,
    const std::vector<torch::Tensor>& inputs) -> std::vector<torch::Tensor>
{
  c10::InferenceMode guard;
  const std::vector<torch::IValue> input_ivalues(inputs.begin(), inputs.end());
  const c10::IValue output = model.forward(input_ivalues);
  try {
    return extract_tensors_from_output(output);
  }
  catch (const UnsupportedModelOutputTypeException&) {
    log_error("Unsupported output type from model.");
    throw;
  }
}

// =============================================================================
// Model and Reference Output Loader: returns CPU model, GPU clones, and ref
// outputs
// =============================================================================

auto
load_model_and_reference_output(const RuntimeConfig& opts)
    -> std::optional<std::tuple<
        torch::jit::script::Module, std::vector<torch::jit::script::Module>,
        std::vector<torch::Tensor>>>
{
  try {
    auto model_cpu =
        load_model(opts.models.empty() ? std::string{} : opts.models[0].path);
    auto models_gpu = opts.use_cuda
                          ? clone_model_to_gpus(model_cpu, opts.device_ids)
                          : std::vector<torch::jit::script::Module>{};
    auto inputs = generate_inputs(
        opts.models.empty() ? std::vector<TensorConfig>{}
                            : opts.models[0].inputs);
    auto output_refs = run_reference_inference(model_cpu, inputs);

    return std::tuple{model_cpu, models_gpu, output_refs};
  }
  catch (const c10::Error& e) {
    log_error(
        "Failed to load model or run reference inference: " +
        std::string(e.what()));
    return std::nullopt;
  }
}

// =============================================================================
// Warmup Phase: Run small number of inference tasks to warm up StarPU & Torch
// =============================================================================

void
run_warmup(
    const RuntimeConfig& opts, StarPUSetup& starpu,
    torch::jit::script::Module& model_cpu,
    std::vector<torch::jit::script::Module>& models_gpu,
    const std::vector<torch::Tensor>& outputs_ref)
{
  NvtxRange nvtx_scope("warmup");
  if (!opts.use_cuda || opts.warmup_request_nb <= 0) {
    return;
  }

  const int warmup_request_nb =
      std::max(opts.warmup_request_nb, opts.max_batch_size);
  log_info(
      opts.verbosity, std::format(
                          "Starting warmup with {} request_nb per CUDA device "
                          "(enforcing max_batch_size)...",
                          warmup_request_nb));

  WarmupRunner warmup_runner(opts, starpu, model_cpu, models_gpu, outputs_ref);
  warmup_runner.run(warmup_request_nb);

  log_info(opts.verbosity, "Warmup complete. Proceeding to real inference.\n");
}

// =============================================================================
// Result Processing: Print latency breakdowns and validate results
// =============================================================================

namespace detail {

auto
build_gpu_model_lookup(
    std::vector<torch::jit::script::Module>& models_gpu,
    const std::vector<int>& device_ids)
    -> std::vector<torch::jit::script::Module*>
{
  std::vector<torch::jit::script::Module*> lookup;
  if (models_gpu.empty() || device_ids.empty()) {
    return lookup;
  }

  const auto max_it = std::max_element(device_ids.begin(), device_ids.end());
  if (max_it == device_ids.end() || *max_it < 0) {
    return lookup;
  }

  lookup.resize(static_cast<size_t>(*max_it) + 1, nullptr);
  const size_t replicas = std::min(models_gpu.size(), device_ids.size());
  for (size_t idx = 0; idx < replicas; ++idx) {
    const int device_id = device_ids[idx];
    if (device_id < 0) {
      continue;
    }
    lookup[static_cast<size_t>(device_id)] = &models_gpu[idx];
  }

  return lookup;
}

auto
resolve_validation_model(
    const InferenceResult& result, torch::jit::script::Module& cpu_model,
    const std::vector<torch::jit::script::Module*>& gpu_lookup,
    bool validate_results) -> std::optional<torch::jit::script::Module*>
{
  if (result.executed_on != DeviceType::CUDA) {
    return &cpu_model;
  }

  if (result.device_id < 0) {
    if (validate_results) {
      log_warning(std::format(
          "[Client] Skipping validation for job {}: invalid device id {}",
          result.request_id, result.device_id));
    }
    return std::nullopt;
  }

  const auto device_id = static_cast<size_t>(result.device_id);
  if (device_id >= gpu_lookup.size() || gpu_lookup[device_id] == nullptr) {
    if (validate_results) {
      log_warning(std::format(
          "[Client] Skipping validation for job {}: no GPU replica for device "
          "{}",
          result.request_id, result.device_id));
    }
    return std::nullopt;
  }

  return gpu_lookup[device_id];
}

void
process_results(
    const std::vector<InferenceResult>& results,
    torch::jit::script::Module& model_cpu,
    std::vector<torch::jit::script::Module>& models_gpu,
    const std::vector<int>& device_ids, bool validate_results,
    VerbosityLevel verbosity, double rtol, double atol)
{
  if (!validate_results) {
    log_info(verbosity, "Result validation disabled; skipping checks.");
  }

  auto gpu_model_lookup = build_gpu_model_lookup(models_gpu, device_ids);
  for (const auto& result : results) {
    const bool has_results =
        !result.results.empty() && result.results[0].defined();
    if (!has_results) {
      if (validate_results) {
        log_error(std::format("[Client] Job {} failed.", result.request_id));
      }
      continue;
    }

    const auto validation_model = resolve_validation_model(
        result, model_cpu, gpu_model_lookup, validate_results);
    if (!validation_model.has_value()) {
      continue;
    }

    if (validate_results) {
      validate_inference_result(
          result, **validation_model, verbosity, rtol, atol);
    }
  }
}

}  // namespace detail

// =============================================================================
// Main Inference Loop: Initializes models, runs warmup, starts client/server,
// waits for all jobs to complete, processes results
// =============================================================================

void
run_inference_loop(const RuntimeConfig& opts, StarPUSetup& starpu)
{
  NvtxRange nvtx_scope("inference_loop");
  const c10::InferenceMode inference_guard;
  CuDnnBenchmarkGuard cudnn_benchmark_guard(
      opts.use_cuda && !opts.dynamic_batching);
  torch::jit::script::Module model_cpu;
  std::vector<torch::jit::script::Module> models_gpu;
  std::vector<torch::Tensor> outputs_ref;

  try {
    auto result = load_model_and_reference_output(opts);
    if (!result) {
      log_error("Failed to load model or reference outputs");
      return;
    }
    std::tie(model_cpu, models_gpu, outputs_ref) = std::move(*result);
  }
  catch (const std::exception& e) {
    log_error(
        std::format("Failed to load model or reference outputs: {}", e.what()));
    return;
  }

  run_warmup(opts, starpu, model_cpu, models_gpu, outputs_ref);

  InferenceQueue queue;
  std::vector<InferenceResult> results;
  std::mutex results_mutex;

  if (opts.request_nb > 0) {
    results.reserve(static_cast<size_t>(opts.request_nb));
  }

  std::atomic completed_jobs = 0;
  std::condition_variable all_done_cv;
  std::mutex all_done_mutex;

  StarPUTaskRunnerConfig config{};
  config.queue = &queue;
  config.model_cpu = &model_cpu;
  config.models_gpu = &models_gpu;
  config.starpu = &starpu;
  config.opts = &opts;
  config.results = &results;
  config.results_mutex = &results_mutex;
  config.completed_jobs = &completed_jobs;
  config.all_done_cv = &all_done_cv;
  StarPUTaskRunner worker(config);

  std::jthread server;
  std::jthread client;
  try {
    server = get_worker_thread_launcher()(worker);
    client = std::jthread([&queue, &opts, &outputs_ref]() {
      detail::client_worker(queue, opts, outputs_ref, opts.request_nb);
    });
  }
  catch (const std::exception& e) {
    log_error(std::format("Failed to start worker thread: {}", e.what()));
    queue.shutdown();
    if (client.joinable()) {
      client.join();
    }
    if (server.joinable()) {
      server.join();
    }
    throw;
  }

  {
    std::unique_lock lock(all_done_mutex);
    all_done_cv.wait(lock, [&completed_jobs, &opts]() {
      return completed_jobs.load(std::memory_order_acquire) >= opts.request_nb;
    });
  }

  if (client.joinable()) {
    client.join();
  }
  if (server.joinable()) {
    server.join();
  }

  {
    std::vector<double> latencies;
    latencies.reserve(results.size());
    for (const auto& result : results) {
      latencies.push_back(result.latency_ms);
    }

    if (auto stats = compute_latency_statistics(std::move(latencies))) {
      if (should_log(VerbosityLevel::Stats, opts.verbosity)) {
        log_stats(
            opts.verbosity,
            std::format(
                "Latency stats (ms): p50={:.3f}, p85={:.3f}, p95={:.3f}, "
                "p100={:.3f}, mean={:.3f}",
                stats->p50, stats->p85, stats->p95, stats->p100, stats->mean));
      }
    }
  }

  detail::process_results(
      results, model_cpu, models_gpu, opts.device_ids, opts.validate_results,
      opts.verbosity, opts.rtol, opts.atol);
}
}  // namespace starpu_server
