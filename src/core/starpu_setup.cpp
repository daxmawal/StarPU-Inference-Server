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
#include <chrono>
#include <climits>
#include <cstddef>
#include <exception>
#include <format>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "device_type.hpp"
#include "exceptions.hpp"
#include "inference_params.hpp"
#include "logger.hpp"
#include "runtime_config.hpp"
#include "tensor_builder.hpp"

namespace starpu_server {
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

  std::function<void(const c10::IValue&)> extract;
  extract = [&](const c10::IValue& value) {
    if (value.isTensor()) {
      outputs.emplace_back(value.toTensor());
    } else if (value.isTuple()) {
      for (const auto& val : value.toTuple()->elements()) {
        extract(val);
      }
    } else if (value.isTensorList()) {
      outputs.insert(
          outputs.end(), value.toTensorList().begin(),
          value.toTensorList().end());
    } else if (value.isList()) {
      for (const auto& val : value.toList()) {
        extract(val);
      }
    } else {
      throw UnsupportedModelOutputTypeException(
          "Unsupported model output type");
    }
  };
  extract(result);

  return outputs;
}

// =============================================================================
// Inference execution (common to CPU and GPU codelets)
// =============================================================================

inline void
run_inference(
    InferenceParams* params, const std::vector<void*>& buffers,
    const torch::Device device, torch::jit::script::Module* model,
    const std::function<void(const at::Tensor&, void* buffer_ptr)>&
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
    auto* var_iface = static_cast<starpu_variable_interface*>(
        buffers[params->num_inputs + i]);
    auto* buffer = reinterpret_cast<void*>(var_iface->ptr);
    copy_output_fn(outputs[i], buffer);
  }
}

template <typename CopyOutputFn>
void
run_codelet_inference(
    InferenceParams* params, const std::vector<void*>& buffers,
    const torch::Device device, torch::jit::script::Module* model,
    CopyOutputFn copy_output_fn, const DeviceType executed_on_type)
{
  if (params->timing.codelet_start_time) {
    *params->timing.codelet_start_time =
        std::chrono::high_resolution_clock::now();
  }

  const int worker_id = starpu_worker_get_id();
  const int device_id = starpu_worker_get_devid(worker_id);

  log_trace(
      params->verbosity,
      std::format(
          "{}, device id, worker id, job id : {} {} {}",
          (executed_on_type == DeviceType::CPU ? "CPU" : "GPU"), device_id,
          worker_id, params->job_id));

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

// =============================================================================
// StarPU CPU codelet implementation
// =============================================================================

inline void
InferenceCodelet::cpu_inference_func(void* buffers[], void* cl_arg)
{
  auto* params = static_cast<InferenceParams*>(cl_arg);
  const std::vector<void*> buffers_vec(
      buffers, buffers + params->num_inputs + params->num_outputs);

  const c10::InferenceMode no_autograd;

  run_codelet_inference(
      params, buffers_vec, torch::kCPU, params->models.model_cpu,
      [](const at::Tensor& out, void* buffer_ptr) {
        TensorBuilder::copy_output_to_buffer(out, buffer_ptr, out.numel());
      },
      DeviceType::CPU);
}

// =============================================================================
// StarPU CUDA codelet implementation
// =============================================================================

inline void
InferenceCodelet::cuda_inference_func(void* buffers[], void* cl_arg)
{
  auto* params = static_cast<InferenceParams*>(cl_arg);
  const std::vector<void*> buffers_vec(
      buffers, buffers + params->num_inputs + params->num_outputs);
  const int worker_id = starpu_worker_get_id();
  const int device_id = starpu_worker_get_devid(worker_id);

  cudaStream_t stream = starpu_cuda_get_local_stream();
  const at::cuda::CUDAStream torch_stream = at::cuda::getStreamFromExternal(
      stream, static_cast<c10::DeviceIndex>(device_id));

  const c10::InferenceMode no_autograd;
  const at::cuda::CUDAStreamGuard guard(torch_stream);

  run_codelet_inference(
      params, buffers_vec,
      torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(device_id)),
      params->models.models_gpu.at(static_cast<size_t>(device_id)),
      [device_id](const at::Tensor& out, void* buffer_ptr) {
        const at::Tensor wrapper = torch::from_blob(
            buffer_ptr, out.sizes(),
            torch::TensorOptions()
                .dtype(out.scalar_type())
                .device(torch::kCUDA, device_id));
        wrapper.copy_(out, true);
      },
      DeviceType::CUDA);
}

// =============================================================================
// StarPUSetup: constructor and destructor (handles StarPU global state)
// =============================================================================

StarPUSetup::StarPUSetup(const RuntimeConfig& opts) : conf_{}
{
  starpu_conf_init(&conf_);
  scheduler_name_ = opts.scheduler;
  conf_.sched_policy_name = scheduler_name_.c_str();

  if (!opts.use_cpu) {
    conf_.ncpus = 0;
  }

  if (!opts.use_cuda) {
    conf_.ncuda = 0;
  } else {
    conf_.use_explicit_workers_cuda_gpuid = 1U;
    conf_.ncuda = static_cast<int>(opts.device_ids.size());
    for (size_t idx = 0; idx < opts.device_ids.size(); ++idx) {
      int device_id = opts.device_ids[idx];
      if (device_id < 0) {
        throw std::invalid_argument(
            "[ERROR] Invalid CUDA device ID: must be >= 0");
      }
      conf_.workers_cuda_gpuid[idx] = static_cast<unsigned int>(device_id);
    }
  }

  if (starpu_init(&conf_) != 0) {
    throw StarPUInitializationException("[ERROR] StarPU initialization error");
  }
}

StarPUSetup::~StarPUSetup()
{
  starpu_shutdown();
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
    const int nworkers = starpu_worker_get_stream_workerids(
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
