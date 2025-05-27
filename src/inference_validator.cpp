#include "inference_validator.hpp"

#include <iostream>

bool
validate_inference_result(
    const InferenceResult& r, torch::jit::script::Module& module,
    const VerbosityLevel& verbosity)
{
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

  module.to(device);

  std::vector<torch::IValue> input_ivalues;
  std::transform(
      r.inputs.begin(), r.inputs.end(), std::back_inserter(input_ivalues),
      [&](const auto& t) { return t.to(device); });

  torch::Tensor ref = module.forward(input_ivalues).toTensor();
  torch::Tensor result_on_device = r.result.to(device);

  const bool is_valid =
      torch::allclose(ref, result_on_device, /*rtol=*/1e-3, /*atol=*/1e-5);

  auto device_str = [](DeviceType dev) {
    switch (dev) {
      case DeviceType::CPU:
        return "CPU";
      case DeviceType::CUDA:
        return "CUDA";
      default:
        return "Unknown";
    }
  };

  if (!is_valid) {
    log_error(
        "[Validator] Mismatch detected for job " + std::to_string(r.job_id));
    log_error("  Executed on: " + std::string(device_str(r.executed_on)));
    log_error("  Reference: " + ref.flatten().slice(0, 0, 10).toString());
    log_error("  Obtained : " + r.result.flatten().slice(0, 0, 10).toString());
  } else {
    log_info(
        verbosity, "[Validator] Job " + std::to_string(r.job_id) +
                       " passed on " + device_str(r.executed_on) +
                       " on device id " + std::to_string(r.device_id));
  }

  return is_valid;
}
