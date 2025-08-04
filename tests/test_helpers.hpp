#pragma once

#include <starpu.h>
#include <torch/script.h>

#include "core/inference_params.hpp"

namespace starpu_server {

inline starpu_variable_interface
make_variable_interface(float* ptr)
{
  starpu_variable_interface iface;
  iface.ptr = reinterpret_cast<uintptr_t>(ptr);
  return iface;
}

inline InferenceParams
make_basic_params(int elements, at::ScalarType type = at::kFloat)
{
  InferenceParams params{};
  params.num_inputs = 1;
  params.num_outputs = 1;
  params.layout.num_dims[0] = 1;
  params.layout.dims[0][0] = elements;
  params.layout.input_types[0] = type;
  return params;
}

}  // namespace starpu_server
