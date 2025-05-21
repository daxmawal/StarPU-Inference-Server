#pragma once
#include <starpu.h>
#include <torch/script.h>

#include <cstring>
#include <stdexcept>
#include <vector>

#include "inference_params.hpp"

class TensorBuilder {
 public:
  static std::vector<torch::Tensor> from_starpu_buffers(
      const InferenceParams* params, void* buffers[], torch::Device device);

  static void copy_output_to_buffer(
      const at::Tensor& output, void* buffer_ptr, int64_t expected_numel);

 private:
  static torch::Tensor from_raw_ptr(
      void* ptr, at::ScalarType type, const std::vector<int64_t>& shape,
      torch::Device device);
};
