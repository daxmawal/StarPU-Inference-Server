#include "starpu_setup.hpp"

#include <ATen/cuda/CUDAContext.h>

InferenceCodelet::InferenceCodelet()
{
  starpu_codelet_init(&codelet_);
  codelet_.nbuffers = STARPU_VARIABLE_NBUFFERS;
  codelet_.max_parallelism = INT_MAX;
  codelet_.type = STARPU_FORKJOIN;
  codelet_.cpu_funcs[0] = &InferenceCodelet::cpu_inference_func;
  codelet_.cuda_funcs[0] = &InferenceCodelet::cuda_inference_func;
  codelet_.name = "inference";
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
  std::cout << "CPU, worker id, job id : " << starpu_worker_get_id() << " "
            << params->job_id << std::endl;

  try {
    auto inputs =
        TensorBuilder::from_starpu_buffers(params, buffers, torch::kCPU);
    std::vector<c10::IValue> ivalue_inputs(inputs.begin(), inputs.end());

    float* output_data = reinterpret_cast<float*>(
        STARPU_VARIABLE_GET_PTR(buffers[params->num_inputs]));

    if (!params->modele_cpu) {
      throw std::runtime_error("[ERROR] TorchScript module is null");
    }

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
}


void
InferenceCodelet::cuda_inference_func(void* buffers[], void* cl_arg)
{
  InferenceParams* params = static_cast<InferenceParams*>(cl_arg);
  std::cout << "GPU, worker id, job id : " << starpu_worker_get_id() << " "
            << params->job_id << std::endl;

  try {
    auto inputs =
        TensorBuilder::from_starpu_buffers(params, buffers, torch::kCUDA);
    std::vector<c10::IValue> ivalue_inputs(inputs.begin(), inputs.end());

    float* output_data = reinterpret_cast<float*>(
        STARPU_VARIABLE_GET_PTR(buffers[params->num_inputs]));

    if (!params->modele_gpu) {
      throw std::runtime_error("[ERROR] TorchScript module is null");
    }

    at::Tensor output = params->modele_gpu->forward(ivalue_inputs).toTensor();

    at::Tensor output_wrapper = torch::from_blob(
        output_data, output.sizes(),
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    cudaStream_t stream = starpu_cuda_get_local_stream();
    c10::cuda::CUDAStream torch_stream =
        c10::cuda::getStreamFromExternal(stream, starpu_worker_get_id());
    c10::cuda::CUDAStream guard(torch_stream);
    output_wrapper.copy_(output, true);
  }
  catch (const c10::Error& e) {
    throw std::runtime_error(
        std::string("[ERROR] GPU (Torch) execution failed: ") + e.what());
  }
  catch (const std::exception& e) {
    throw std::runtime_error(
        std::string("[ERROR] GPU codelet failure: ") + e.what());
  }
}

StarPUSetup::StarPUSetup(const char* sched_policy)
{
  starpu_conf_init(&conf_);
  conf_.sched_policy_name = sched_policy;
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
