#pragma once
#include <starpu.h>
#include <torch/script.h>

#include <cstring>
#include <iostream>
#include <vector>

#include "exceptions.hpp"

namespace InferLimits {
constexpr size_t MaxInputs = 16;
constexpr size_t MaxDims = 8;
}  // namespace InferLimits

// Structure holding the parameters required for an inference task
struct InferenceParams {
  torch::jit::script::Module* module;
  size_t num_inputs;
  size_t num_outputs;
  int64_t output_size;
  int64_t dims[InferLimits::MaxInputs][InferLimits::MaxDims];
  int64_t num_dims[InferLimits::MaxInputs];
  at::ScalarType input_types[InferLimits::MaxInputs];
};

// StarPUSetup encapsulates StarPU initialization and codelet configuration
class StarPUSetup {
 public:
  static constexpr const char* kCodeletName = "cpu_inference";
  static constexpr const char* kCpuFuncsName = "cpu_codelet_func";

  StarPUSetup(const char* sched_policy)
  {
    starpu_conf_init(&conf_);
    conf_.sched_policy_name = sched_policy;
    if (starpu_init(&conf_) != 0) {
      throw std::runtime_error("[ERROR] StarPU initialization error");
    }

    // Initialize the codelet that will perform inference on CPU
    starpu_codelet_init(&codelet_);
    codelet_.nbuffers = STARPU_VARIABLE_NBUFFERS;
    codelet_.max_parallelism = INT_MAX;
    codelet_.type = STARPU_FORKJOIN;
    codelet_.cpu_funcs[0] = cpu_codelet_func;
    codelet_.cpu_funcs_name[0] = kCpuFuncsName;
    codelet_.name = kCodeletName;
  }

  ~StarPUSetup() { starpu_shutdown(); }

  struct starpu_codelet* codelet() { return &codelet_; }

 private:
  static torch::Tensor raw_ptr_to_tensor(
      void* raw_ptr, at::ScalarType type, const std::vector<int64_t>& shape)
  {
    switch (type) {
      case at::kFloat:
        return torch::from_blob(
                   reinterpret_cast<float*>(raw_ptr), shape, at::kFloat)
            .contiguous();
      case at::kInt:
        return torch::from_blob(
                   reinterpret_cast<int32_t*>(raw_ptr), shape, at::kInt)
            .contiguous();
      case at::kLong:
        return torch::from_blob(
                   reinterpret_cast<int64_t*>(raw_ptr), shape, at::kLong)
            .contiguous();
      case at::kBool:
        return torch::from_blob(
                   reinterpret_cast<bool*>(raw_ptr), shape, at::kBool)
            .contiguous();
      default:
        throw UnsupportedDtypeException(
            "[ERROR] Unsupported input type in raw_ptr_to_tensor(): " +
            std::to_string(static_cast<int>(type)));
    }
  }

  static std::vector<torch::Tensor> create_input_tensors(
      const InferenceParams* params, void* buffers[])
  {
    if (params->num_inputs > InferLimits::MaxInputs) {
      throw InferenceExecutionException(
          "[ERROR] Too many input tensors, the maximum is : " +
          std::to_string(InferLimits::MaxInputs));
    }

    std::vector<torch::Tensor> inputs;
    inputs.reserve(params->num_inputs);

    for (size_t i = 0; i < params->num_inputs; ++i) {
      void* raw_ptr = reinterpret_cast<void*>(
          static_cast<uintptr_t>(STARPU_VARIABLE_GET_PTR(buffers[i])));

      if (!raw_ptr) {
        throw InferenceExecutionException(
            "[ERROR] Received null pointer in StarPU buffer");
      }

      if (params->num_dims[i] > InferLimits::MaxDims) {
        throw InferenceExecutionException(
            "[ERROR] Too many dimensions for input " + std::to_string(i));
      }

      std::vector<int64_t> shape(
          params->dims[i], params->dims[i] + params->num_dims[i]);
      inputs.emplace_back(
          raw_ptr_to_tensor(raw_ptr, params->input_types[i], shape));
    }

    return inputs;
  }

  // CPU function executed by StarPU when the task runs
  static void cpu_codelet_func(void* buffers[], void* cl_arg)
  {
    InferenceParams* params = static_cast<InferenceParams*>(cl_arg);

    try {
      std::vector<torch::Tensor> inputs = create_input_tensors(params, buffers);

      std::vector<c10::IValue> ivalue_inputs(inputs.begin(), inputs.end());

      float* output_data = reinterpret_cast<float*>(
          STARPU_VARIABLE_GET_PTR(buffers[params->num_inputs]));

      at::Tensor output = params->module->forward(ivalue_inputs).toTensor();

      if (output.numel() != params->output_size) {
        throw InferenceExecutionException(
            "[ERROR] Mismatch between declared output size and model output "
            "size.");
      }

      std::memcpy(
          output_data, output.data_ptr<float>(),
          static_cast<size_t>(params->output_size) * sizeof(float));
    }
    catch (const c10::Error& e) {
      throw InferenceExecutionException(
          std::string("[ERROR] Error during model inference in StarPU task: ") +
          e.what());
    }
    catch (const std::exception& e) {
      throw InferenceExecutionException(
          std::string("[ERROR] Standard exception during StarPU task: ") +
          e.what());
    }
    catch (...) {
      throw InferenceExecutionException(
          "[ERROR] Unknown exception during StarPU task");
    }
  }

  struct starpu_conf conf_;
  struct starpu_codelet codelet_;
};