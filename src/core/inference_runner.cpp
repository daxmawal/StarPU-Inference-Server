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
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "client_utils.hpp"
#include "exceptions.hpp"
#include "inference_queue.hpp"
#include "inference_session.hpp"
#include "inference_validator.hpp"
#include "input_generator.hpp"
#include "latency_statistics.hpp"
#include "logger.hpp"
#include "runtime_config.hpp"
#include "starpu_setup.hpp"
#include "starpu_task_worker.hpp"
#include "utils/batching_trace_logger.hpp"
#include "utils/nvtx.hpp"
#include "warmup.hpp"

namespace starpu_server {

namespace {

auto
cuda_device_count_override_storage() -> std::optional<int>&
{
  static std::optional<int> override;
  return override;
}

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
  CuDnnBenchmarkGuard(CuDnnBenchmarkGuard&& other) noexcept
      : previous_(other.previous_), active_(other.active_)
  {
    other.active_ = false;
  }
  auto operator=(const CuDnnBenchmarkGuard&) -> CuDnnBenchmarkGuard& = delete;
  auto operator=(CuDnnBenchmarkGuard&& other) noexcept -> CuDnnBenchmarkGuard&
  {
    if (this != &other) {
      if (active_) {
        at::globalContext().setBenchmarkCuDNN(previous_);
      }
      previous_ = other.previous_;
      active_ = other.active_;
      if (active_) {
        at::globalContext().setBenchmarkCuDNN(true);
      }
      other.active_ = false;
    }
    return *this;
  }

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

namespace detail {

void
set_cuda_device_count_override(std::optional<int> override)
{
  if (override.has_value() && *override < 0) {
    throw std::invalid_argument(
        "CUDA device count override must be non-negative");
  }
  cuda_device_count_override_storage() = override;
}

auto
get_cuda_device_count() -> int
{
  if (const auto& override = cuda_device_count_override_storage();
      override.has_value()) {
    return *override;
  }

  const auto raw_count = torch::cuda::device_count();
  if (raw_count < 0) {
    throw InvalidGpuDeviceException(
        "torch::cuda::device_count returned a negative value.");
  }
  if (raw_count > std::numeric_limits<int>::max()) {
    throw InvalidGpuDeviceException(std::format(
        "torch::cuda::device_count returned {}, which exceeds int range.",
        raw_count));
  }

  return static_cast<int>(raw_count);
}

void
validate_device_ids(std::span<const int> device_ids, int available_device_count)
{
  if (available_device_count < 0) {
    throw InvalidGpuDeviceException("CUDA device count cannot be negative.");
  }

  for (const int device_id : device_ids) {
    if (device_id < 0 || device_id >= available_device_count) {
      log_error(std::format(
          "GPU ID {} out of range. Only {} device(s) available.", device_id,
          available_device_count));
      throw InvalidGpuDeviceException(std::format(
          "Invalid GPU device ID {} ({} device(s) available).", device_id,
          available_device_count));
    }
  }
}

}  // namespace detail

namespace {
inline auto
current_worker_thread_launcher_storage() -> WorkerThreadLauncher&
{
  static WorkerThreadLauncher launcher = default_worker_thread_launcher;
  return launcher;
}
}  // namespace

auto
get_worker_thread_launcher() -> WorkerThreadLauncher
{
  return current_worker_thread_launcher_storage();
}

void
set_worker_thread_launcher(WorkerThreadLauncher launcher)
{
  current_worker_thread_launcher_storage() = std::move(launcher);
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
  if (opts.seed.has_value()) {
    rng.seed(*opts.seed);
    torch::manual_seed(*opts.seed);
  } else {
    rng.seed(std::random_device{}());
  }

  auto pregen_inputs =
      client_utils::pre_generate_inputs(opts, opts.batching.pregen_inputs);

  auto next_time = std::chrono::steady_clock::now();
  const auto delay = std::chrono::microseconds(opts.batching.delay_us);
  for (auto request_id = 0; request_id < request_nb; ++request_id) {
    std::this_thread::sleep_until(next_time);
    next_time += delay;
    const auto& inputs = client_utils::pick_random_input(pregen_inputs, rng);
    auto job = client_utils::create_job(
        inputs, outputs_ref, request_id, {}, {}, opts.name);
    if (!queue.push(job)) {
      log_warning(std::format(
          "[Client] Failed to enqueue job {}: queue shutting down",
          request_id));
      break;
    }
    const auto enqueued_now = std::chrono::high_resolution_clock::now();
    job->timing_info().enqueued_time = enqueued_now;
    job->timing_info().last_enqueued_time = enqueued_now;
    client_utils::log_job_enqueued(
        opts, request_id, request_nb, job->timing_info().enqueued_time);
    auto& tracer = BatchingTraceLogger::instance();
    if (tracer.enabled()) {
      tracer.log_request_enqueued(
          job->get_request_id(), job->model_name(), /*is_warmup=*/false,
          enqueued_now);
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
    log_error(std::format("Failed to load model: {}", e.what()));
    throw;
  }
}

static auto
clone_model_to_gpus(
    const torch::jit::script::Module& model_cpu,
    const std::vector<int>& device_ids)
    -> std::vector<torch::jit::script::Module>
{
  if (device_ids.empty()) {
    return {};
  }

  const auto device_count = detail::get_cuda_device_count();
  detail::validate_device_ids(device_ids, device_count);

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

static auto
synthesize_outputs_from_config(const RuntimeConfig& opts)
    -> std::optional<std::vector<torch::Tensor>>
{
  if (opts.models.empty()) {
    return std::nullopt;
  }

  const auto& model_cfg = opts.models.front();
  if (model_cfg.outputs.empty()) {
    return std::nullopt;
  }

  std::vector<torch::Tensor> outputs;
  outputs.reserve(model_cfg.outputs.size());

  for (const auto& tensor_cfg : model_cfg.outputs) {
    const auto tensor_name =
        tensor_cfg.name.empty() ? "<unnamed output>" : tensor_cfg.name;
    if (tensor_cfg.type == at::ScalarType::Undefined) {
      log_warning(std::format(
          "Output tensor '{}' is missing a valid data_type; falling back to "
          "reference inference.",
          tensor_name));
      return std::nullopt;
    }
    if (tensor_cfg.dims.empty()) {
      log_warning(std::format(
          "Output tensor '{}' is missing dims; falling back to reference "
          "inference.",
          tensor_name));
      return std::nullopt;
    }

    for (const auto dim : tensor_cfg.dims) {
      if (dim <= 0) {
        log_warning(std::format(
            "Output tensor '{}' has non-positive dimension {}; falling back to "
            "reference inference.",
            tensor_name, dim));
        return std::nullopt;
      }
    }

    const auto options =
        torch::TensorOptions().device(torch::kCPU).dtype(tensor_cfg.type);
    outputs.emplace_back(torch::empty(tensor_cfg.dims, options));
  }

  return outputs;
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
    auto models_gpu = opts.devices.use_cuda
                          ? clone_model_to_gpus(model_cpu, opts.devices.ids)
                          : std::vector<torch::jit::script::Module>{};
    std::optional<std::vector<torch::Tensor>> synthetic_outputs;
    if (!opts.validation.validate_results) {
      synthetic_outputs = synthesize_outputs_from_config(opts);
    }

    std::vector<torch::Tensor> output_refs;
    if (opts.validation.validate_results || !synthetic_outputs.has_value()) {
      if (!opts.validation.validate_results) {
        log_debug(
            opts.verbosity,
            "Validation disabled but missing usable output schema; running "
            "reference inference once to infer output sizes.");
      }
      auto inputs = generate_inputs(
          opts.models.empty() ? std::vector<TensorConfig>{}
                              : opts.models[0].inputs);
      output_refs = run_reference_inference(model_cpu, inputs);
    } else {
      log_debug(
          opts.verbosity,
          "Validation disabled; using configured output schema instead of "
          "running CPU reference inference.");
      output_refs = std::move(*synthetic_outputs);
    }

    return std::tuple{model_cpu, models_gpu, output_refs};
  }
  catch (const c10::Error& e) {
    log_error(std::format(
        "Failed to load model or run reference inference: {}", e.what()));
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
  if (!opts.devices.use_cuda || opts.batching.warmup_request_nb <= 0) {
    return;
  }

  const int warmup_request_nb =
      std::max(opts.batching.warmup_request_nb, opts.batching.max_batch_size);
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

inline auto
result_job_id(const InferenceResult& result) -> int
{
  return (result.submission_id >= 0) ? result.submission_id : result.request_id;
}

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

  const auto max_it = std::ranges::max_element(device_ids);
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
    std::span<torch::jit::script::Module*> gpu_lookup,
    bool validate_results) -> std::optional<torch::jit::script::Module*>
{
  if (result.executed_on != DeviceType::CUDA) {
    return &cpu_model;
  }

  if (result.device_id < 0) {
    if (validate_results) {
      log_warning(std::format(
          "[Client] Skipping validation for job {}: invalid device id {}",
          result_job_id(result), result.device_id));
    }
    return std::nullopt;
  }

  const auto device_id = static_cast<size_t>(result.device_id);
  if (device_id >= gpu_lookup.size() || gpu_lookup[device_id] == nullptr) {
    if (validate_results) {
      log_warning(std::format(
          "[Client] Skipping validation for job {}: no GPU replica for device "
          "{}",
          result_job_id(result), result.device_id));
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
    const RuntimeConfig& opts)
{
  if (!opts.validation.validate_results) {
    log_info(opts.verbosity, "Result validation disabled; skipping checks.");
  }

  auto gpu_model_lookup = build_gpu_model_lookup(models_gpu, opts.devices.ids);
  for (const auto& result : results) {
    if (const bool has_results =
            !result.results.empty() && result.results[0].defined();
        !has_results) {
      if (opts.validation.validate_results) {
        log_error(
            std::format("[Client] Job {} failed.", result_job_id(result)));
      }
      continue;
    }

    const auto validation_model = resolve_validation_model(
        result, model_cpu, gpu_model_lookup, opts.validation.validate_results);
    if (!validation_model.has_value()) {
      continue;
    }

    if (opts.validation.validate_results) {
      validate_inference_result(
          result, **validation_model, opts.verbosity, opts.validation.rtol,
          opts.validation.atol);
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
  BatchingTraceLogger::instance().configure_from_runtime(opts);
  const c10::InferenceMode inference_guard;
  CuDnnBenchmarkGuard cudnn_benchmark_guard(
      opts.devices.use_cuda && !opts.batching.dynamic_batching);
  InferenceSession session(opts, starpu, detail::client_worker);
  session.run();
}
}  // namespace starpu_server
