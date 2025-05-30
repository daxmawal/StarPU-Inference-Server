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

void
InferenceCodelet::cpu_inference_func(void* buffers[], void* cl_arg)
{
  const int worker_id = starpu_worker_get_id();
  const int device_id = starpu_worker_get_devid(worker_id);

  auto* params = static_cast<InferenceParams*>(cl_arg);
  *params->timing.codelet_start_time =
      std::chrono::high_resolution_clock::now();

  log_trace(
      params->verbosity,
      "CPU, device id, worker id, job id : " + std::to_string(device_id) + " " +
          std::to_string(worker_id) + " " + std::to_string(params->job_id));

  if (params->device.executed_on)
    *params->device.executed_on = DeviceType::CPU;
  *params->device.worker_id = worker_id;
  *params->device.device_id = device_id;

  try {
    auto inputs =
        TensorBuilder::from_starpu_buffers(params, buffers, torch::kCPU);
    std::vector<c10::IValue> ivalue_inputs(inputs.begin(), inputs.end());

    float* output_data = reinterpret_cast<float*>(
        STARPU_VARIABLE_GET_PTR(buffers[params->num_inputs]));


    auto* model = params->models.model_cpu;
    TORCH_CHECK(model, "TorchScript module is null");

    *params->timing.inference_start_time =
        std::chrono::high_resolution_clock::now();

    at::Tensor output = model->forward(ivalue_inputs).toTensor();
    if (output.numel() != params->output_size)
      throw std::runtime_error("[ERROR] Output size mismatch");

    TensorBuilder::copy_output_to_buffer(output, output_data, output.numel());
  }
  catch (const std::exception& e) {
    throw std::runtime_error(
        "[ERROR] CPU codelet failure: " + std::string(e.what()));
  }

  *params->timing.codelet_end_time = std::chrono::high_resolution_clock::now();
}

void
InferenceCodelet::cuda_inference_func(void* buffers[], void* cl_arg)
{
  const int worker_id = starpu_worker_get_id();
  const int device_id = starpu_worker_get_devid(worker_id);

  auto* params = static_cast<InferenceParams*>(cl_arg);
  *params->timing.codelet_start_time =
      std::chrono::high_resolution_clock::now();

  log_trace(
      params->verbosity,
      "GPU, device id, worker id, job id : " + std::to_string(device_id) + " " +
          std::to_string(worker_id) + " " + std::to_string(params->job_id));

  if (params->device.executed_on)
    *params->device.executed_on = DeviceType::CUDA;
  *params->device.worker_id = worker_id;
  *params->device.device_id = device_id;

  try {
    auto inputs =
        TensorBuilder::from_starpu_buffers(params, buffers, torch::kCUDA);
    std::vector<c10::IValue> ivalue_inputs(inputs.begin(), inputs.end());

    float* output_data = reinterpret_cast<float*>(
        STARPU_VARIABLE_GET_PTR(buffers[params->num_inputs]));

    auto* model = params->models.models_gpu[static_cast<size_t>(device_id)];
    TORCH_CHECK(model, "TorchScript module is null");

    const cudaStream_t stream = starpu_cuda_get_local_stream();

    const at::cuda::CUDAStream torch_stream = at::cuda::getStreamFromExternal(
        stream, static_cast<c10::DeviceIndex>(device_id));

    c10::InferenceMode no_autograd;
    at::cuda::CUDAStreamGuard guard(torch_stream);

    *params->timing.inference_start_time =
        std::chrono::high_resolution_clock::now();

    at::Tensor output = model->forward(ivalue_inputs).toTensor();

    at::Tensor output_wrapper = torch::from_blob(
        output_data, output.sizes(),
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    output_wrapper.copy_(output, true);
  }
  catch (const c10::Error& e) {
    throw std::runtime_error(
        "[ERROR] GPU (Torch) execution failed: " + std::string(e.what()));
  }
  catch (const std::exception& e) {
    throw std::runtime_error(
        "[ERROR] GPU codelet failure: " + std::string(e.what()));
  }

  *params->timing.codelet_end_time = std::chrono::high_resolution_clock::now();
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

const std::map<int, std::vector<int>>
StarPUSetup::get_cuda_workers_by_device(
    const std::vector<unsigned int>& device_ids)
{
  std::map<int, std::vector<int>> device_to_workers;

  for (int device_id : device_ids) {
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