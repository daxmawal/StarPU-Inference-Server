#pragma once
#include <starpu.h>
#include <torch/script.h>

#include <cstring>
#include <iostream>
#include <vector>

#include "exceptions.hpp"

// Structure holding the parameters required for an inference task
struct InferenceParams {
  torch::jit::script::Module module;
  size_t num_inputs;
  size_t num_outputs;
  int64_t output_size;
  int64_t dims[8][8];
  size_t num_dims[16];
  at::ScalarType input_types[16];
};

// StarPUSetup encapsulates StarPU initialization and codelet configuration
class StarPUSetup {
 public:
  StarPUSetup(const char* sched_policy)
  {
    starpu_conf_init(&conf_);
    conf_.sched_policy_name = sched_policy;
    if (starpu_init(&conf_) != 0) {
      throw std::runtime_error("StarPU initialization error");
    }

    // Initialize the codelet that will perform inference on CPU
    starpu_codelet_init(&codelet_);
    codelet_.nbuffers = STARPU_VARIABLE_NBUFFERS;
    codelet_.max_parallelism = INT_MAX;
    codelet_.type = STARPU_FORKJOIN;
    codelet_.cpu_funcs[0] = cpu_codelet_func;
    codelet_.cpu_funcs_name[0] = "cpu_codelet_func";
    codelet_.modes[0] = STARPU_R;
    codelet_.modes[1] = STARPU_W;
    codelet_.name = "cpu_inference";
  }

  ~StarPUSetup() { starpu_shutdown(); }

  struct starpu_codelet* codelet() { return &codelet_; }

 private:
  // CPU function executed by StarPU when the task runs
  static void cpu_codelet_func(void* buffers[], void* cl_arg)
  {
    InferenceParams* params = static_cast<InferenceParams*>(cl_arg);

    try {
      std::vector<torch::Tensor> inputs;

      for (size_t i = 0; i < params->num_inputs; ++i) {
        void* raw_ptr = reinterpret_cast<void*>(
            static_cast<uintptr_t>(STARPU_VARIABLE_GET_PTR(buffers[i])));

        std::vector<int64_t> shape(
            params->dims[i], params->dims[i] + params->num_dims[i]);

        torch::Tensor input;

        switch (params->input_types[i]) {
          case at::kFloat:
            input = torch::from_blob(
                        reinterpret_cast<float*>(raw_ptr), shape, at::kFloat)
                        .clone();
            break;
          case at::kInt:
            input = torch::from_blob(
                        reinterpret_cast<int32_t*>(raw_ptr), shape, at::kInt)
                        .clone();
            break;
          case at::kLong:
            input = torch::from_blob(
                        reinterpret_cast<int64_t*>(raw_ptr), shape, at::kLong)
                        .clone();
            break;
          case at::kBool:
            input = torch::from_blob(
                        reinterpret_cast<bool*>(raw_ptr), shape, at::kBool)
                        .clone();
            break;
          default:
            throw std::runtime_error(
                "Unsupported input type in StarPU codelet");
        }

        inputs.push_back(input);
      }

      // Convert to IValue inputs
      std::vector<c10::IValue> ivalue_inputs;
      for (const auto& tensor : inputs) {
        ivalue_inputs.push_back(tensor);
      }

      // Output
      float* output_data = reinterpret_cast<float*>(
          STARPU_VARIABLE_GET_PTR(buffers[params->num_inputs]));

      at::Tensor output = params->module.forward(ivalue_inputs).toTensor();

      std::memcpy(
          output_data, output.data_ptr<float>(),
          static_cast<size_t>(params->output_size) * sizeof(float));
    }
    catch (const c10::Error& e) {
      throw InferenceExecutionException(
          std::string("Error during model inference in StarPU task: ") +
          e.what());
    }
  }


  struct starpu_conf conf_;
  struct starpu_codelet codelet_;
};