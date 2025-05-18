#pragma once
#include <starpu.h>
#include <torch/script.h>

#include <cstring>
#include <iostream>
#include <vector>

#include "exceptions.hpp"
#include "inference_params.hpp"
#include "tensor_builder.hpp"

class InferenceCodelet {
 public:
  InferenceCodelet()
  {
    starpu_codelet_init(&codelet_);
    codelet_.nbuffers = STARPU_VARIABLE_NBUFFERS;
    codelet_.max_parallelism = INT_MAX;
    codelet_.type = STARPU_FORKJOIN;
    codelet_.cpu_funcs[0] = &InferenceCodelet::cpu_inference_func;
    codelet_.cpu_funcs_name[0] = "cpu_infer_func";
    codelet_.name = "cpu_inference";
  }

  struct starpu_codelet* get_codelet() { return &codelet_; }

 private:
  static inline void cpu_inference_func(void* buffers[], void* cl_arg)
  {
    InferenceParams* params = static_cast<InferenceParams*>(cl_arg);

    try {
      auto inputs = TensorBuilder::from_starpu_buffers(params, buffers);
      std::vector<c10::IValue> ivalue_inputs(inputs.begin(), inputs.end());

      float* output_data = reinterpret_cast<float*>(
          STARPU_VARIABLE_GET_PTR(buffers[params->num_inputs]));

      if (!params->module) {
        throw std::runtime_error("[ERROR] TorchScript module is null");
      }

      at::Tensor output = params->module->forward(ivalue_inputs).toTensor();
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

  struct starpu_codelet codelet_;
};

class StarPUSetup {
 public:
  explicit StarPUSetup(const char* sched_policy)
  {
    starpu_conf_init(&conf_);
    conf_.sched_policy_name = sched_policy;
    if (starpu_init(&conf_) != 0) {
      throw std::runtime_error("[ERROR] StarPU initialization error");
    }
  }

  ~StarPUSetup() { starpu_shutdown(); }

  struct starpu_codelet* codelet() { return codelet_.get_codelet(); }

 private:
  struct starpu_conf conf_;
  InferenceCodelet codelet_;
};