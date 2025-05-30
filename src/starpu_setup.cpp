#include "starpu_setup.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "exceptions.hpp"

// =============================================================================
// InferenceCodelet: Setup and execution functions for StarPU codelets
// =============================================================================
InferenceCodelet::InferenceCodelet()
{
  starpu_codelet_init(&codelet_);
  codelet_.nbuffers = STARPU_VARIABLE_NBUFFERS;
  codelet_.type = STARPU_FORKJOIN;
  codelet_.max_parallelism = INT_MAX,
  codelet_.cpu_funcs[0] = &InferenceCodelet::cpu_inference_func;
  codelet_.cuda_funcs[0] = &InferenceCodelet::cuda_inference_func;
  codelet_.cuda_flags[0] = STARPU_CUDA_ASYNC;
}

struct starpu_codelet*
InferenceCodelet::get_codelet()
{
  return &codelet_;
}

// Convert output IValue to vector of Tensors
std::vector<at::Tensor>
extract_tensors_from_output(const c10::IValue& result)
{
  std::vector<at::Tensor> outputs;

  if (result.isTensor()) {
    outputs.emplace_back(result.toTensor());
  } else if (result.isTuple()) {
    for (const auto& val : result.toTuple()->elements()) {
      TORCH_CHECK(val.isTensor(), "Expected tensor in tuple output");
      outputs.emplace_back(val.toTensor());
    }
  } else if (result.isTensorList()) {
    outputs.insert(
        outputs.end(), result.toTensorList().begin(),
        result.toTensorList().end());

  } else {
    throw std::runtime_error("Unsupported model output type");
  }

  return outputs;
}

// Unified inference execution
inline void
run_inference(
    InferenceParams* params, void* buffers[], torch::Device device,
    torch::jit::script::Module* model,
    std::function<void(const at::Tensor&, void* buffer_ptr)> copy_output_fn)
{
  auto inputs = TensorBuilder::from_starpu_buffers(params, buffers, device);
  std::vector<c10::IValue> ivalue_inputs(inputs.begin(), inputs.end());

  *params->timing.inference_start_time =
      std::chrono::high_resolution_clock::now();

  auto result = model->forward(ivalue_inputs);
  std::vector<at::Tensor> outputs = extract_tensors_from_output(result);

  TORCH_CHECK(
      outputs.size() == static_cast<size_t>(params->num_outputs),
      "Mismatch between model outputs and StarPU buffers");

  for (size_t i = 0; i < params->num_outputs; ++i) {
    void* buffer = reinterpret_cast<void*>(
        STARPU_VARIABLE_GET_PTR(buffers[params->num_inputs + i]));
    copy_output_fn(outputs[i], buffer);
  }
}

template <typename CopyOutputFn>
void
run_codelet_inference(
    InferenceParams* params, void* buffers[], torch::Device device,
    torch::jit::script::Module* model, CopyOutputFn copy_output_fn,
    DeviceType executed_on_type)
{
  *params->timing.codelet_start_time =
      std::chrono::high_resolution_clock::now();

  const int worker_id = starpu_worker_get_id();
  const int device_id = starpu_worker_get_devid(worker_id);

  log_trace(
      params->verbosity, (executed_on_type == DeviceType::CPU ? "CPU" : "GPU") +
                             std::string(", device id, worker id, job id : ") +
                             std::to_string(device_id) + " " +
                             std::to_string(worker_id) + " " +
                             std::to_string(params->job_id));

  if (params->device.executed_on)
    *params->device.executed_on = executed_on_type;
  *params->device.worker_id = worker_id;
  *params->device.device_id = device_id;

  try {
    run_inference(params, buffers, device, model, copy_output_fn);
  }
  catch (const std::exception& e) {
    throw std::runtime_error(
        "[ERROR] Codelet failure: " + std::string(e.what()));
  }

  *params->timing.codelet_end_time = std::chrono::high_resolution_clock::now();
}

// CPU codelet
void
InferenceCodelet::cpu_inference_func(void* buffers[], void* cl_arg)
{
  auto* params = static_cast<InferenceParams*>(cl_arg);
  run_codelet_inference(
      params, buffers, torch::kCPU, params->models.model_cpu,
      [](const at::Tensor& out, void* buffer_ptr) {
        TensorBuilder::copy_output_to_buffer(out, buffer_ptr, out.numel());
      },
      DeviceType::CPU);
}

// GPU codelet
void
InferenceCodelet::cuda_inference_func(void* buffers[], void* cl_arg)
{
  auto* params = static_cast<InferenceParams*>(cl_arg);
  const int worker_id = starpu_worker_get_id();
  const int device_id = starpu_worker_get_devid(worker_id);

  const cudaStream_t stream = starpu_cuda_get_local_stream();
  const at::cuda::CUDAStream torch_stream = at::cuda::getStreamFromExternal(
      stream, static_cast<c10::DeviceIndex>(device_id));

  c10::InferenceMode no_autograd;
  at::cuda::CUDAStreamGuard guard(torch_stream);

  run_codelet_inference(
      params, buffers,
      torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(device_id)),
      params->models.models_gpu[static_cast<size_t>(device_id)],
      [device_id](const at::Tensor& out, void* buffer_ptr) {
        at::Tensor wrapper = torch::from_blob(
            buffer_ptr, out.sizes(),
            torch::TensorOptions()
                .dtype(torch::kFloat32)
                .device(torch::kCUDA, device_id));
        wrapper.copy_(out, true);
      },
      DeviceType::CUDA);
}

// =============================================================================
// StarPUSetup: StarPU initialization and resource management
// =============================================================================
StarPUSetup::StarPUSetup(const ProgramOptions& opts)
{
  starpu_conf_init(&conf_);
  conf_.sched_policy_name = opts.scheduler.c_str();

  if (!opts.use_cpu)
    conf_.ncpus = 0;

  if (!opts.use_cuda) {
    conf_.ncuda = 0;
  } else {
    conf_.use_explicit_workers_cuda_gpuid = 1;
    conf_.ncuda = static_cast<int>(opts.device_ids.size());
    for (size_t i = 0; i < opts.device_ids.size(); ++i)
      conf_.workers_cuda_gpuid[i] = opts.device_ids[i];
  }

  if (starpu_init(&conf_) != 0)
    throw std::runtime_error("[ERROR] StarPU initialization error");
}

StarPUSetup::~StarPUSetup()
{
  starpu_shutdown();
}

struct starpu_codelet*
StarPUSetup::codelet()
{
  return codelet_.get_codelet();
}

const std::map<unsigned int, std::vector<int>>
StarPUSetup::get_cuda_workers_by_device(
    const std::vector<unsigned int>& device_ids)
{
  std::map<unsigned int, std::vector<int>> device_to_workers;

  for (auto device_id : device_ids) {
    int workerids[STARPU_NMAXWORKERS];
    int nworkers = starpu_worker_get_stream_workerids(
        device_id, workerids, STARPU_CUDA_WORKER);

    if (nworkers < 0) {
      throw std::runtime_error(
          "Failed to get CUDA workers for device " + std::to_string(device_id));
    }

    device_to_workers[device_id] =
        std::vector<int>(workerids, workerids + nworkers);
  }

  return device_to_workers;
}