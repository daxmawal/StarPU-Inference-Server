#include "starpu_setup.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

InferenceCodelet::InferenceCodelet()
{
  starpu_codelet_init(&codelet_);
  codelet_.nbuffers = STARPU_VARIABLE_NBUFFERS;
  codelet_.type = STARPU_FORKJOIN;
  codelet_.cpu_funcs[0] = &InferenceCodelet::cpu_inference_func;
  codelet_.cuda_funcs[0] = &InferenceCodelet::cuda_inference_func;
}

struct starpu_codelet*
InferenceCodelet::get_codelet()
{
  return &codelet_;
}

void
InferenceCodelet::cpu_inference_func(void* buffers[], void* cl_arg)
{
  InferenceParams* params = static_cast<InferenceParams*>(cl_arg);
  *params->codelet_start_time = std::chrono::high_resolution_clock::now();
  std::cout << "CPU, worker id, job id : " << starpu_worker_get_id() << " "
            << params->job_id << std::endl;
  if (params->executed_on) {
    *(params->executed_on) = DeviceType::CPU;
  }
  *params->device_id = starpu_worker_get_id();
  try {
    auto inputs =
        TensorBuilder::from_starpu_buffers(params, buffers, torch::kCPU);
    std::vector<c10::IValue> ivalue_inputs(inputs.begin(), inputs.end());

    float* output_data = reinterpret_cast<float*>(
        STARPU_VARIABLE_GET_PTR(buffers[params->num_inputs]));

    if (!params->modele_cpu) {
      throw std::runtime_error("[ERROR] TorchScript module is null");
    }

    *params->inference_start_time = std::chrono::high_resolution_clock::now();
    at::Tensor output = params->modele_cpu->forward(ivalue_inputs).toTensor();
    if (output.numel() != params->output_size) {
      throw std::runtime_error("[ERROR] Output size mismatch");
    }

    TensorBuilder::copy_output_to_buffer(output, output_data, output.numel());
  }
  catch (const std::exception& e) {
    throw std::runtime_error(
        std::string("[ERROR] CPU codelet failure: ") + e.what());
  }

  *params->codelet_end_time = std::chrono::high_resolution_clock::now();
}


void
InferenceCodelet::cuda_inference_func(void* buffers[], void* cl_arg)
{
  InferenceParams* params = static_cast<InferenceParams*>(cl_arg);
  *params->codelet_start_time = std::chrono::high_resolution_clock::now();
  std::cout << "GPU, worker id, job id : " << starpu_worker_get_id() << " "
            << params->job_id << std::endl;
  if (params->executed_on) {
    *(params->executed_on) = DeviceType::CUDA;
  }
  *params->device_id = starpu_worker_get_id();
  try {
    auto inputs =
        TensorBuilder::from_starpu_buffers(params, buffers, torch::kCUDA);
    std::vector<c10::IValue> ivalue_inputs(inputs.begin(), inputs.end());

    float* output_data = reinterpret_cast<float*>(
        STARPU_VARIABLE_GET_PTR(buffers[params->num_inputs]));

    if (!params->modele_gpu) {
      throw std::runtime_error("[ERROR] TorchScript module is null");
    }

    const cudaStream_t stream = starpu_cuda_get_local_stream();
    const int worker_id = starpu_worker_get_id();
    TORCH_CHECK(
        worker_id <= std::numeric_limits<c10::DeviceIndex>::max(),
        "Worker ID too large");
    const at::cuda::CUDAStream torch_stream = at::cuda::getStreamFromExternal(
        stream, static_cast<c10::DeviceIndex>(worker_id));


    const at::cuda::CUDAStreamGuard guard(torch_stream);

    *params->inference_start_time = std::chrono::high_resolution_clock::now();
    at::Tensor output = params->modele_gpu->forward(ivalue_inputs).toTensor();

    at::Tensor output_wrapper = torch::from_blob(
        output_data, output.sizes(),
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    output_wrapper.copy_(output, true);

    cudaStreamSynchronize(stream);
  }
  catch (const c10::Error& e) {
    throw std::runtime_error(
        std::string("[ERROR] GPU (Torch) execution failed: ") + e.what());
  }
  catch (const std::exception& e) {
    throw std::runtime_error(
        std::string("[ERROR] GPU codelet failure: ") + e.what());
  }

  *params->codelet_end_time = std::chrono::high_resolution_clock::now();
}

StarPUSetup::StarPUSetup(const ProgramOptions& opts)
{
  starpu_conf_init(&conf_);
  conf_.sched_policy_name = opts.scheduler.c_str();
  if (!opts.use_cpu) {
    conf_.ncpus = 0;
  }

  if (!opts.use_cuda) {
    conf_.ncuda = 0;
  } else {
    conf_.use_explicit_workers_cuda_gpuid = 1;
    conf_.ncuda = static_cast<int>(opts.device_ids.size());
    for (size_t i = 0; i < opts.device_ids.size(); ++i) {
      conf_.workers_cuda_gpuid[i] = opts.device_ids[i];
    }
  }

  if (starpu_init(&conf_) != 0) {
    throw std::runtime_error("[ERROR] StarPU initialization error");
  }
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
