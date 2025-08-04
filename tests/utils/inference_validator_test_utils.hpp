#pragma once

#include <utility>
#include <vector>

#include "core/inference_runner.hpp"

namespace starpu_server {

inline auto
make_result(
    std::vector<torch::Tensor> inputs, std::vector<torch::Tensor> outputs,
    int job_id, DeviceType device, int device_id = 0,
    int worker_id = 0) -> InferenceResult
{
  InferenceResult result;
  result.inputs = std::move(inputs);
  result.results = std::move(outputs);
  result.job_id = job_id;
  result.executed_on = device;
  result.device_id = device_id;
  result.worker_id = worker_id;
  return result;
}

}  // namespace starpu_server
