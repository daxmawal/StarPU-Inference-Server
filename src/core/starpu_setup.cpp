#include "starpu_setup.hpp"

#include <ATen/core/TensorBody.h>
#include <c10/core/Device.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <starpu.h>
#include <starpu_config.h>
#include <starpu_cuda.h>
#include <torch/types.h>

#include <array>
#include <bit>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <format>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "device_type.hpp"
#include "exceptions.hpp"
#include "inference_params.hpp"
#include "logger.hpp"
#include "runtime_config.hpp"
#include "tensor_builder.hpp"
#include "utils/nvtx.hpp"

namespace starpu_server {
namespace {
void append_ivalue(const c10::IValue& value, std::vector<at::Tensor>& outputs);

auto
starpu_init_fn_ref() -> StarPUSetup::StarpuInitFn&
{
  static auto instance = StarPUSetup::StarpuInitFn(starpu_init);
  return instance;
}

auto
worker_stream_query_fn_ref() -> StarPUSetup::WorkerStreamQueryFn&
{
  static auto instance =
      StarPUSetup::WorkerStreamQueryFn(starpu_worker_get_stream_workerids);
  return instance;
}

void
apply_starpu_env(const RuntimeConfig& opts)
{
  for (const auto& [name, value] : opts.starpu_env) {
    if (name.empty()) {
      throw StarPUInitializationException(
          "Environment variable name cannot be empty");
    }
    if (setenv(name.c_str(), value.c_str(), 1) != 0) {
      throw StarPUInitializationException(std::format(
          "Failed to set environment variable {}: {}", name,
          std::strerror(errno)));
    }
  }
}

void
configure_cpu(starpu_conf& conf, bool use_cpu)
{
  if (!use_cpu) {
    conf.ncpus = 0;
  }
}

void
configure_gpu(starpu_conf& conf, const RuntimeConfig& opts)
{
  if (!opts.devices.use_cuda) {
    conf.ncuda = 0;
    return;
  }

  if (opts.devices.ids.size() > STARPU_NMAXWORKERS) {
    throw std::invalid_argument(std::format(
        "[ERROR] Number of CUDA device IDs exceeds maximum of {}",
        STARPU_NMAXWORKERS));
  }

  std::unordered_set<int> unique_ids;
  std::vector<int> valid_device_ids;
  valid_device_ids.reserve(opts.devices.ids.size());

  for (const int device_id : opts.devices.ids) {
    if (device_id < 0) {
      log_error(std::format(
          "Invalid CUDA device ID {}: must be non-negative", device_id));
      continue;
    }
    if (!unique_ids.insert(device_id).second) {
      throw std::invalid_argument(
          std::format("[ERROR] Duplicate CUDA device ID: {}", device_id));
    }
    valid_device_ids.push_back(device_id);
  }

  conf.use_explicit_workers_cuda_gpuid = valid_device_ids.empty() ? 0U : 1U;
  conf.ncuda = static_cast<int>(valid_device_ids.size());

  std::span<unsigned int, STARPU_NMAXWORKERS> workers_cuda_gpuid(
      conf.workers_cuda_gpuid);
  for (size_t idx = 0; idx < valid_device_ids.size(); ++idx) {
    workers_cuda_gpuid[idx] = static_cast<unsigned int>(valid_device_ids[idx]);
  }
}

auto
initialize_input_pool(const RuntimeConfig& opts)
    -> std::unique_ptr<InputSlotPool>
{
  if (opts.models.empty() || opts.models[0].inputs.empty()) {
    return nullptr;
  }

  try {
    return std::make_unique<InputSlotPool>(opts, opts.batching.pool_size);
  }
  catch (const std::exception& e) {
    log_error(std::format("Failed to initialize InputSlotPool: {}", e.what()));
    throw;
  }
}

auto
initialize_output_pool(const RuntimeConfig& opts)
    -> std::unique_ptr<OutputSlotPool>
{
  if (opts.models.empty() || opts.models[0].outputs.empty()) {
    return nullptr;
  }

  try {
    return std::make_unique<OutputSlotPool>(opts, opts.batching.pool_size);
  }
  catch (const std::exception& e) {
    log_error(
        std::format("Failed to initialize OutputSlotPool: {}", +e.what()));
    throw;
  }
}
}  // namespace
namespace {
void
append_from_tuple(
    const c10::IValue& tuple_value, std::vector<at::Tensor>& outputs)
{
  for (const auto& element : tuple_value.toTuple()->elements()) {
    append_ivalue(element, outputs);
  }
}

void
append_from_list(
    const c10::IValue& list_value, std::vector<at::Tensor>& outputs)
{
  for (const auto& element : list_value.toList()) {
    append_ivalue(element, outputs);
  }
}

void
append_from_dict(
    const c10::IValue& dict_value, std::vector<at::Tensor>& outputs)
{
  for (const auto& item : dict_value.toGenericDict()) {
    append_ivalue(item.value(), outputs);
  }
}

void
append_ivalue(const c10::IValue& value, std::vector<at::Tensor>& outputs)
{
  if (value.isTensor()) {
    outputs.emplace_back(value.toTensor());
  } else if (value.isTensorList()) {
    const auto tensors = value.toTensorList();
    outputs.insert(outputs.end(), tensors.begin(), tensors.end());
  } else if (value.isTuple()) {
    append_from_tuple(value, outputs);
  } else if (value.isList()) {
    append_from_list(value, outputs);
  } else if (value.isGenericDict()) {
    append_from_dict(value, outputs);
  } else {
    throw UnsupportedModelOutputTypeException("Unsupported model output type");
  }
}
}  // namespace

void run_inference(
    InferenceParams* params, const std::vector<StarpuBufferPtr>& buffers,
    torch::Device device, torch::jit::script::Module* model,
    const std::function<void(const at::Tensor&, std::span<std::byte>)>&
        copy_output_fn);
// =============================================================================
// InferenceCodelet: constructor and access to codelet
// =============================================================================

InferenceCodelet::InferenceCodelet() : codelet_{}
{
  starpu_codelet_init(&codelet_);
  codelet_.nbuffers = STARPU_VARIABLE_NBUFFERS;
  codelet_.type = STARPU_FORKJOIN;
  codelet_.max_parallelism = INT_MAX;
  codelet_.cpu_funcs[0] = &InferenceCodelet::cpu_inference_func;
  codelet_.cuda_funcs[0] = &InferenceCodelet::cuda_inference_func;
  codelet_.cuda_flags[0] = 1U;
}

auto
InferenceCodelet::get_codelet() -> struct starpu_codelet*
{
  return &codelet_;
}

// =============================================================================
// Utility: extract list of tensors from torch::IValue output
// =============================================================================

auto
extract_tensors_from_output(const c10::IValue& result)
    -> std::vector<at::Tensor>
{
  std::vector<at::Tensor> outputs;
  append_ivalue(result, outputs);

  return outputs;
}

// =============================================================================
// Inference execution (common to CPU and GPU codelets)
// =============================================================================

inline void
run_inference(
    InferenceParams* params, StarpuBufferSpan buffers, torch::Device device,
    torch::jit::script::Module* model,
    const std::function<void(const at::Tensor&, std::span<std::byte>)>&
        copy_output_fn)
{
  const auto inputs =
      TensorBuilder::from_starpu_buffers(params, buffers, device);
  const std::vector<c10::IValue> ivalue_inputs(inputs.begin(), inputs.end());

  if (params->timing.inference_start_time != nullptr) {
    *params->timing.inference_start_time =
        std::chrono::high_resolution_clock::now();
  }

  auto result = model->forward(ivalue_inputs);
  std::vector<at::Tensor> outputs = extract_tensors_from_output(result);

  TORCH_CHECK(
      outputs.size() == static_cast<size_t>(params->num_outputs),
      "Mismatch between model outputs and StarPU buffers");

  for (size_t i = 0; i < params->num_outputs; ++i) {
    auto* var_iface = buffers[params->num_inputs + i];
    auto* buffer_ptr = std::bit_cast<std::byte*>(var_iface->ptr);
    const auto byte_size = outputs[i].nbytes();
    std::span<std::byte> buffer(buffer_ptr, byte_size);
    copy_output_fn(outputs[i], buffer);
  }
}

void
run_inference(
    InferenceParams* params, const std::vector<StarpuBufferPtr>& buffers,
    torch::Device device, torch::jit::script::Module* model,
    const std::function<void(const at::Tensor&, std::span<std::byte>)>&
        copy_output_fn)
{
  run_inference(
      params, StarpuBufferSpan(buffers.data(), buffers.size()), device, model,
      copy_output_fn);
}

template <typename CopyOutputFn>
void
run_codelet_inference(
    InferenceParams* params, StarpuBufferSpan buffers,
    const torch::Device device, torch::jit::script::Module* model,
    CopyOutputFn copy_output_fn, const DeviceType executed_on_type)
{
  if (params->timing.codelet_start_time) {
    *params->timing.codelet_start_time =
        std::chrono::high_resolution_clock::now();
  }

  const int worker_id = starpu_worker_get_id();
  const int device_id = starpu_worker_get_devid(worker_id);

  if (should_log(VerbosityLevel::Trace, params->verbosity)) {
    log_trace(
        params->verbosity,
        std::format(
            "{} device id {}, worker id {}, job id {}",
            (executed_on_type == DeviceType::CPU ? "CPU" : "GPU"), device_id,
            worker_id, params->request_id));
  }

  if (params->device.executed_on) {
    *params->device.executed_on = executed_on_type;
  }
  if (params->device.worker_id) {
    *params->device.worker_id = worker_id;
  }
  if (params->device.device_id) {
    *params->device.device_id = device_id;
  }

  try {
    run_inference(params, buffers, device, model, copy_output_fn);
  }
  catch (const std::exception& e) {
    throw StarPUCodeletException(
        std::format("[ERROR] Codelet failure: {}", e.what()));
  }

  if (params->timing.codelet_end_time) {
    *params->timing.codelet_end_time =
        std::chrono::high_resolution_clock::now();
  }
}

auto
select_gpu_module(const InferenceParams& params, const int device_id)
    -> torch::jit::script::Module*
{
  if (device_id >= 0) {
    const auto module_index = static_cast<size_t>(device_id);
    if (module_index < params.models.models_gpu.size()) {
      if (auto* model_instance = params.models.models_gpu[module_index]) {
        return model_instance;
      }
    }
  }

  throw StarPUCodeletException(std::format(
      "[ERROR] No GPU model replica available for device {}", device_id));
}

// =============================================================================
// StarPU CPU codelet implementation
// =============================================================================

inline void
InferenceCodelet::cpu_inference_func(void** buffers, void* cl_arg)
{
  auto* params = static_cast<InferenceParams*>(cl_arg);
  const auto total_buffers = params->num_inputs + params->num_outputs;
  const auto* typed_buffers = std::bit_cast<StarpuBufferPtr const*>(buffers);
  const StarpuBufferSpan buffers_span(typed_buffers, total_buffers);

  const c10::InferenceMode no_autograd;

  run_codelet_inference(
      params, buffers_span, torch::kCPU, params->models.model_cpu,
      [](const at::Tensor& out, std::span<std::byte> buffer) {
        TensorBuilder::copy_output_to_buffer(
            out, buffer, out.numel(), out.scalar_type());
      },
      DeviceType::CPU);
}

// =============================================================================
// StarPU CUDA codelet implementation
// =============================================================================

inline void
InferenceCodelet::cuda_inference_func(void** buffers, void* cl_arg)
{
  auto* params = static_cast<InferenceParams*>(cl_arg);
  const auto total_buffers = params->num_inputs + params->num_outputs;
  const auto* typed_buffers = std::bit_cast<StarpuBufferPtr const*>(buffers);
  const StarpuBufferSpan buffers_span(typed_buffers, total_buffers);
  const int worker_id = starpu_worker_get_id();
  const int device_id = starpu_worker_get_devid(worker_id);

  NvtxRange nvtx_scope(
      std::format("codelet.cuda job {} dev {}", params->request_id, device_id));

  cudaStream_t stream = starpu_cuda_get_local_stream();
  const at::cuda::CUDAStream torch_stream = at::cuda::getStreamFromExternal(
      stream, static_cast<c10::DeviceIndex>(device_id));

  const c10::InferenceMode no_autograd;
  const at::cuda::CUDAStreamGuard guard(torch_stream);

  torch::jit::script::Module* model_instance =
      select_gpu_module(*params, device_id);

  run_codelet_inference(
      params, buffers_span,
      torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(device_id)),
      model_instance,
      [device_id](const at::Tensor& out, std::span<std::byte> buffer) {
        const at::Tensor wrapper = torch::from_blob(
            static_cast<void*>(buffer.data()), out.sizes(),
            torch::TensorOptions()
                .dtype(out.scalar_type())
                .device(torch::kCUDA, device_id));
        TORCH_CHECK(
            buffer.size() == static_cast<size_t>(out.nbytes()),
            "Output buffer size mismatch in bytes");
        wrapper.copy_(out, true);
      },
      DeviceType::CUDA);
}

// =============================================================================
// StarPUSetup: constructor and destructor (handles StarPU global state)
// =============================================================================

StarPUSetup::StarPUSetup(const RuntimeConfig& opts)
    : scheduler_name_(opts.scheduler), conf_{}
{
  apply_starpu_env(opts);
  starpu_conf_init(&conf_);
  conf_.sched_policy_name = scheduler_name_.c_str();

  configure_cpu(conf_, opts.devices.use_cpu);
  configure_gpu(conf_, opts);

  if (starpu_init_fn_ref()(&conf_) != 0) {
    throw StarPUInitializationException("[ERROR] StarPU initialization error");
  }

  input_pool_ = initialize_input_pool(opts);
  output_pool_ = initialize_output_pool(opts);
}

StarPUSetup::~StarPUSetup()
{
  input_pool_.reset();
  output_pool_.reset();
  starpu_shutdown();
}

void
StarPUSetup::set_starpu_init_fn(StarpuInitFn hook_fn)
{
  auto& fn_ref = starpu_init_fn_ref();
  if (hook_fn) {
    fn_ref = std::move(hook_fn);
  } else {
    fn_ref = starpu_init;
  }
}

void
StarPUSetup::reset_starpu_init_fn()
{
  starpu_init_fn_ref() = starpu_init;
}

void
StarPUSetup::set_worker_stream_query_fn(WorkerStreamQueryFn hook_fn)
{
  auto& fn_ref = worker_stream_query_fn_ref();
  if (hook_fn) {
    fn_ref = std::move(hook_fn);
  } else {
    fn_ref = starpu_worker_get_stream_workerids;
  }
}

void
StarPUSetup::reset_worker_stream_query_fn()
{
  worker_stream_query_fn_ref() = starpu_worker_get_stream_workerids;
}

// =============================================================================
// StarPUSetup: access to codelet and CUDA worker mapping
// =============================================================================

auto
StarPUSetup::get_codelet() -> struct starpu_codelet*
{
  return codelet_.get_codelet();
}

auto
StarPUSetup::get_cuda_workers_by_device(const std::vector<int>& device_ids)
    -> std::map<int, std::vector<int>>
{
  std::map<int, std::vector<int>> device_to_workers;

  for (auto device_id : device_ids) {
    if (device_id < 0) {
      throw std::invalid_argument("device_id must be non-negative");
    }

    std::array<int, STARPU_NMAXWORKERS> workerids{};
    const int nworkers = worker_stream_query_fn_ref()(
        static_cast<unsigned int>(device_id), workerids.data(),
        STARPU_CUDA_WORKER);

    if (nworkers < 0) {
      throw StarPUWorkerQueryException(
          std::format("Failed to get CUDA workers for device {}", device_id));
    }

    device_to_workers[device_id] =
        std::vector<int>(workerids.begin(), workerids.begin() + nworkers);
  }

  return device_to_workers;
}
}  // namespace starpu_server
