#include "inference_validator.hpp"

#include <algorithm>
#include <iostream>

#include "device_type.hpp"

// =============================================================================
// Validate the result of an inference by comparing it to the reference output
// =============================================================================
bool
validate_inference_result(
    const InferenceResult& r, torch::jit::script::Module& module,
    const VerbosityLevel& verbosity)
{
  // Select the appropriate device based on where the job was executed
  torch::Device device(torch::kCPU);
  switch (r.executed_on) {
    case DeviceType::CUDA:
      device = torch::Device(torch::kCUDA);
      break;
    case DeviceType::CPU:
      device = torch::Device(torch::kCPU);
      break;
    default:
      log_error(
          "[Validator] Unknown device for job " + std::to_string(r.job_id));
      return false;
  }

  // Move model and inputs to the correct device
  module.to(device);

  std::vector<torch::IValue> input_ivalues;
  std::transform(
      r.inputs.begin(), r.inputs.end(), std::back_inserter(input_ivalues),
      [&](const torch::Tensor& t) { return t.to(device); });

  // Run reference inference
  torch::Tensor reference_output = module.forward(input_ivalues).toTensor();

  // Move the actual result to the same device for fair comparison
  torch::Tensor result_on_device = r.result.to(device);

  // Compare result and reference (element-wise)
  const bool is_valid = torch::allclose(
      reference_output, result_on_device, /*rtol=*/1e-3, /*atol=*/1e-5);

  // Logging
  if (!is_valid) {
    log_error(
        "[Validator] Mismatch detected for job " + std::to_string(r.job_id));
    log_error(
        "  Absolute diff max: " +
        std::to_string(
            (reference_output - result_on_device).abs().max().item<float>()));
    log_error("  Executed on: " + std::string(to_string(r.executed_on)));
    log_error(
        "  Reference: " +
        reference_output.flatten().slice(0, 0, 10).toString());
    log_error("  Obtained : " + r.result.flatten().slice(0, 0, 10).toString());
  } else {
    log_info(
        verbosity, "[Validator] Job " + std::to_string(r.job_id) +
                       " passed on " + to_string(r.executed_on) +
                       " (device id " + std::to_string(r.device_id) +
                       " worker id " + std::to_string(r.worker_id) + ")");
  }

  return is_valid;
}
