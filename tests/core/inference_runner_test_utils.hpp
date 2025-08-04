#pragma once

#include <torch/script.h>

#include <vector>

#include "inference_runner_test_utils.hpp"
#include "utils/exceptions.hpp"
#include "utils/logger.hpp"

namespace starpu_server {

inline auto
make_identity_model() -> torch::jit::script::Module
{
  torch::jit::script::Module m{"m"};
  m.define(R"JIT(
        def forward(self, x):
            return x
    )JIT");
  return m;
}

inline auto
run_reference_inference(
    torch::jit::script::Module& model,
    const std::vector<torch::Tensor>& inputs) -> std::vector<torch::Tensor>
{
  std::vector<torch::Tensor> output_refs;
  const std::vector<torch::IValue> input_ivalues(inputs.begin(), inputs.end());

  if (const auto output = model.forward(input_ivalues); output.isTensor()) {
    output_refs.push_back(output.toTensor());
  } else if (output.isTuple()) {
    for (const auto& val : output.toTuple()->elements()) {
      if (val.isTensor()) {
        output_refs.push_back(val.toTensor());
      }
    }
  } else if (output.isTensorList()) {
    output_refs.insert(
        output_refs.end(), output.toTensorList().begin(),
        output.toTensorList().end());
  } else {
    log_error("Unsupported output type from model.");
    throw UnsupportedModelOutputTypeException("Unsupported model output type");
  }

  return output_refs;
}

}  // namespace starpu_server
