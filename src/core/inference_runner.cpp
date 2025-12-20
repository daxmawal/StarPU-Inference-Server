#include "inference_runner.hpp"

#include <ATen/core/ScalarType.h>
#include <c10/core/InferenceMode.h>
#include <c10/util/Exception.h>
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <format>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include "exceptions.hpp"
#include "input_generator.hpp"
#include "logger.hpp"
#include "runtime_config.hpp"
#include "starpu_setup.hpp"
#include "utils/nvtx.hpp"
#include "warmup.hpp"

namespace starpu_server {

namespace {

auto
cuda_device_count_override_storage() -> std::optional<int>&
{
  static std::optional<int> override_storage;
  return override_storage;
}

}  // namespace

namespace detail {

#if defined(STARPU_TESTING)
void
set_cuda_device_count_override(std::optional<int> override_count)
{
  if (override_count.has_value() && *override_count < 0) {
    throw std::invalid_argument(
        "CUDA device count override must be non-negative");
  }
  cuda_device_count_override_storage() = override_count;
}
#endif

auto
sanitize_cuda_device_count(long long raw_count) -> int
{
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

auto
get_cuda_device_count() -> int
{
  if (const auto& override_storage = cuda_device_count_override_storage();
      override_storage.has_value()) {
    return *override_storage;
  }

  using DeviceCountSigned = long long;
  const auto device_count_signed =
      static_cast<DeviceCountSigned>(torch::cuda::device_count());

  return sanitize_cuda_device_count(device_count_signed);
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

auto
compute_latency_breakdown(const TimingInfo& timing, double total_latency_ms)
    -> BaseLatencyBreakdown
{
  using duration_f = std::chrono::duration<double, std::milli>;

  BaseLatencyBreakdown breakdown{};
  breakdown.queue_ms =
      duration_f(timing.dequeued_time - timing.enqueued_time).count();
  breakdown.batch_ms = std::max(
      0.0, duration_f(
               timing.batch_collect_end_time - timing.batch_collect_start_time)
               .count());

  auto submit_start = timing.batch_collect_start_time;
  if (submit_start == std::chrono::high_resolution_clock::time_point{}) {
    submit_start = timing.dequeued_time;
  }
  breakdown.submit_ms = std::max(
      0.0,
      duration_f(timing.before_starpu_submitted_time - submit_start).count());

  breakdown.scheduling_ms =
      duration_f(
          timing.codelet_start_time - timing.before_starpu_submitted_time)
          .count();
  breakdown.codelet_ms =
      duration_f(timing.codelet_end_time - timing.codelet_start_time).count();
  breakdown.inference_ms =
      duration_f(timing.callback_start_time - timing.inference_start_time)
          .count();
  breakdown.callback_ms =
      duration_f(timing.callback_end_time - timing.callback_start_time).count();
  breakdown.total_ms = total_latency_ms;
  return breakdown;
}

}  // namespace detail

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
  job->set_logical_job_count(0);
  job->set_aggregated_sub_jobs({});
  return job;
}

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
  if (!opts.model.has_value()) {
    return std::nullopt;
  }

  const auto& model_cfg = *opts.model;
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
  const auto load_start = std::chrono::high_resolution_clock::now();
  const auto model_label = [&opts]() -> std::string {
    if (!opts.model.has_value()) {
      return "default";
    }
    if (!opts.model->name.empty()) {
      return opts.model->name;
    }
    return opts.model->path;
  }();
  auto mark_failure = [&]() {
    increment_model_load_failure(model_label);
    set_model_loaded(model_label, "cpu", false);
    if (opts.devices.use_cuda) {
      for (const auto device_id : opts.devices.ids) {
        set_model_loaded(model_label, std::format("cuda:{}", device_id), false);
      }
    }
  };

  try {
    auto model_cpu =
        load_model(opts.model.has_value() ? opts.model->path : std::string{});
    auto models_gpu = opts.devices.use_cuda
                          ? clone_model_to_gpus(model_cpu, opts.devices.ids)
                          : std::vector<torch::jit::script::Module>{};
    auto synthetic_outputs = synthesize_outputs_from_config(opts);

    std::vector<torch::Tensor> output_refs;
    if (synthetic_outputs.has_value()) {
      log_debug(
          opts.verbosity,
          "Using configured output schema instead of running CPU reference "
          "inference.");
      output_refs = std::move(*synthetic_outputs);
    } else {
      log_debug(
          opts.verbosity,
          "No usable output schema provided; running reference inference once "
          "to infer output sizes.");
      auto inputs = generate_inputs(
          opts.model.has_value() ? opts.model->inputs
                                 : std::vector<TensorConfig>{});
      output_refs = run_reference_inference(model_cpu, inputs);
    }

    const double duration_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - load_start)
            .count();
    observe_model_load_duration(duration_ms);
    set_model_loaded(model_label, "cpu", true);
    if (opts.devices.use_cuda) {
      for (const auto device_id : opts.devices.ids) {
        set_model_loaded(model_label, std::format("cuda:{}", device_id), true);
      }
    }

    return std::tuple{model_cpu, models_gpu, output_refs};
  }
  catch (const c10::Error& e) {
    mark_failure();
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
  if (!opts.devices.use_cpu && !opts.devices.use_cuda) {
    return;
  }

  const int configured_batches =
      std::max(0, opts.batching.warmup_batches_per_worker);
  if (opts.batching.warmup_request_nb <= 0 && configured_batches <= 0) {
    return;
  }

  const int max_batch_size = std::max(1, opts.batching.max_batch_size);
  const int min_requests_for_batches = configured_batches * max_batch_size;
  const int warmup_request_nb =
      std::max(opts.batching.warmup_request_nb, min_requests_for_batches);
  if (warmup_request_nb <= 0) {
    return;
  }
  const auto target_desc = [&opts]() -> std::string_view {
    if (opts.devices.use_cpu && opts.devices.use_cuda) {
      return "CPU and CUDA workers";
    }
    if (opts.devices.use_cuda) {
      return "CUDA workers";
    }
    return "CPU workers";
  }();
  log_info(
      opts.verbosity, std::format(
                          "Starting warmup with {} request(s) per {} "
                          "(configured batch runs per worker: {})",
                          warmup_request_nb, target_desc, configured_batches));

  WarmupRunner warmup_runner(opts, starpu, model_cpu, models_gpu, outputs_ref);
  warmup_runner.run(warmup_request_nb);

  log_info(opts.verbosity, "Warmup complete.");
}

}  // namespace starpu_server
