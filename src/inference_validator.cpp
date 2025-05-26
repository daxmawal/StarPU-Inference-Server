#include "inference_validator.hpp"

#include <iostream>

bool
validate_inference_result(
    const InferenceResult& r, torch::jit::script::Module& module)
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
      std::cerr << "[Validator] Unknown device for job " << r.job_id << "\n";
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
    std::cerr << "[Validator] Mismatch detected for job " << r.job_id << "!\n";
    std::cerr << "  Executed on: " << device_str(r.executed_on) << "\n";
    std::cerr << "  Reference: " << ref.flatten().slice(0, 0, 10) << "\n";
    std::cerr << "  Obtained : " << r.result.flatten().slice(0, 0, 10) << "\n";
  } else {
    std::cout << "[Validator] Job " << r.job_id << " passed on "
              << device_str(r.executed_on) << "\n";
  }

  return is_valid;
}