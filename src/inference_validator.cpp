#include "inference_validator.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <string>
#include <vector>

#include "device_type.hpp"
#include "inference_runner.hpp"
#include "logger.hpp"

constexpr int kPreviewLimit = 10;

// =============================================================================
// Validate the result of an inference by comparing it to the reference output
// =============================================================================
auto
validate_inference_result(
    const InferenceResult& result, torch::jit::script::Module& module,
    const VerbosityLevel& verbosity) -> bool
{
  // Select the device used for inference
  torch::Device device(torch::kCPU);
  switch (result.executed_on) {
    case DeviceType::CUDA:
      device = torch::Device(torch::kCUDA);
      break;
    case DeviceType::CPU:
      device = torch::Device(torch::kCPU);
      break;
    default:
      log_error(
          "[Validator] Unknown device for job " +
          std::to_string(result.job_id));
      return false;
  }

  module.to(device);

  // Prepare input tensors
  std::vector<torch::IValue> input_ivalues;
  std::transform(
      result.inputs.begin(), result.inputs.end(),
      std::back_inserter(input_ivalues),
      [&](const torch::Tensor& tensor) { return tensor.to(device); });

  // Run reference inference
  const torch::IValue output = module.forward(input_ivalues);

  // Convert reference output to a vector of tensors
  std::vector<torch::Tensor> reference_outputs;
  if (output.isTensor()) {
    reference_outputs.push_back(output.toTensor());
  } else if (output.isTuple()) {
    auto elements = output.toTuple()->elements();
    for (const auto& val : elements) {
      if (val.isTensor()) {
        reference_outputs.push_back(val.toTensor());
      } else {
        log_error(
            "[Validator] Non-tensor output in tuple for job " +
            std::to_string(result.job_id));
        return false;
      }
    }
  } else {
    log_error(
        "[Validator] Unsupported output type for job " +
        std::to_string(result.job_id));
    return false;
  }

  // Check that the number of outputs matches
  if (reference_outputs.size() != result.results.size()) {
    log_error(
        "[Validator] Output count mismatch for job " +
        std::to_string(result.job_id));
    return false;
  }

  // Compare each output tensor
  bool all_valid = true;
  for (size_t i = 0; i < reference_outputs.size(); ++i) {
    const torch::Tensor ref = reference_outputs[i].to(device);
    const torch::Tensor res = result.results[i].to(device);

    const bool is_valid =
        torch::allclose(ref, res, /*rtol=*/1e-3, /*atol=*/1e-5);
    all_valid &= is_valid;

    if (!is_valid) {
      log_error(
          "[Validator] Mismatch on output #" + std::to_string(i) + " for job " +
          std::to_string(result.job_id));
      log_error(
          "  Absolute diff max: " +
          std::to_string((ref - res).abs().max().item<float>()));
      log_error(
          "  Reference: " +
          ref.flatten().slice(0, 0, kPreviewLimit).toString());
      log_error(
          "  Obtained : " +
          res.flatten().slice(0, 0, kPreviewLimit).toString());
    }
  }

  if (all_valid) {
    log_info(
        verbosity, "[Validator] Job " + std::to_string(result.job_id) +
                       " passed on " + to_string(result.executed_on) +
                       " (device id " + std::to_string(result.device_id) +
                       " worker id " + std::to_string(result.worker_id) + ")");
  }

  return all_valid;
}