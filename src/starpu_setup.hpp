#pragma once
#include <starpu.h>
#include <torch/script.h>
#include <iostream>
#include <cstring>
#include <vector>

struct InferenceParams
{
  char model_path[512];
  int64_t input_size;
  int64_t output_size;
  int64_t dims[8];
  int ndims;
};

class StarPUSetup
{
 public:
  StarPUSetup(const char* sched_policy)
  {
    starpu_conf_init(&conf_);
    conf_.sched_policy_name = sched_policy;
    if (starpu_init(&conf_) != 0)
    {
      throw std::runtime_error("StarPU initialization error");
    }

    starpu_codelet_init(&codelet_);
    codelet_.nbuffers = 2;
    codelet_.max_parallelism = INT_MAX;
    codelet_.type = STARPU_FORKJOIN;
    codelet_.cpu_funcs[0] = cpu_codelet_func;
    codelet_.cpu_funcs_name[0] = "cpu_codelet_func";
    codelet_.modes[0] = STARPU_R;
    codelet_.modes[1] = STARPU_W;
    codelet_.name = "cpu_inference";
  }

  ~StarPUSetup()
  {
    starpu_shutdown();
  }

  struct starpu_codelet* codelet() 
  { 
    return &codelet_; 
  }

 private:
  static void cpu_codelet_func(void *buffers[], void *cl_arg)
  {
    float* input_data = reinterpret_cast<float *>(STARPU_VARIABLE_GET_PTR(buffers[0]));
    float* output_data = reinterpret_cast<float *>(STARPU_VARIABLE_GET_PTR(buffers[1]));
    InferenceParams* params = static_cast<InferenceParams*>(cl_arg);

    try
    {
      torch::jit::script::Module module = torch::jit::load(params->model_path);

      std::vector<int64_t> shape(params->dims, params->dims + params->ndims);
      torch::Tensor input = torch::from_blob(input_data, shape, torch::kFloat32).clone();

      at::Tensor output = module.forward({input}).toTensor();

      std::memcpy(output_data, output.data_ptr<float>(), params->output_size * sizeof(float));
      std::cout << "Inference done." << std::endl;
    }
    catch (const c10::Error& e)
    {
      std::cerr << "Error loading or running the model: " << e.what() << std::endl;
    }
  }

  struct starpu_conf conf_;
  struct starpu_codelet codelet_;
};