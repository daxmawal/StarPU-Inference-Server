#include "inference_validator.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <string>
#include <vector>

#include "device_type.hpp"
#include "exceptions.hpp"
#include "inference_runner.hpp"
#include "logger.hpp"

namespace starpu_server {
constexpr int kPreviewLimit = 10;

// =============================================================================
// Device and Input Preparation Utilities
// =============================================================================

// Returns the device on which the inference ran (CPU or CUDA)
auto
get_inference_device(const InferenceResult& result) -> torch::Device
{
  switch (result.executed_on) {
    case DeviceType::CUDA:
      return {torch::kCUDA};
    case DeviceType::CPU:
      return {torch::kCPU};
    default:
      log_error(
          "[Validator] Unknown device for job " +
          std::to_string(result.job_id));
      throw InferenceExecutionException("Unknown device");
  }
}

// Converts input tensors to the appropriate device and wraps them in IValues
auto
prepare_inputs(
    const std::vector<torch::Tensor>& inputs,
    const torch::Device& device) -> std::vector<torch::IValue>
{
  std::vector<torch::IValue> input_ivalues;
  std::ranges::transform(
      inputs, std::back_inserter(input_ivalues),
      [&](const torch::Tensor& tensor) { return tensor.to(device); });
  return input_ivalues;
}

// =============================================================================
// Output Extraction and Comparison
// =============================================================================

// Converts model output (Tensor or Tuple) to a list of tensors
auto
extract_reference_outputs(
    const torch::IValue& output,
    const InferenceResult& result) -> std::vector<torch::Tensor>
{
  std::vector<torch::Tensor> tensors;

  if (output.isTensor()) {
    tensors.push_back(output.toTensor());
  } else if (output.isTuple()) {
    for (const auto& val : output.toTuple()->elements()) {
      if (!val.isTensor()) {
        log_error(
            "[Validator] Non-tensor output in tuple for job " +
            std::to_string(result.job_id));
        throw InferenceExecutionException("Invalid tuple element");
      }
      tensors.push_back(val.toTensor());
    }
  } else {
    log_error(
        "[Validator] Unsupported output type for job " +
        std::to_string(result.job_id));
    throw InferenceExecutionException("Unsupported output type");
  }

  return tensors;
}

// Compares two lists of tensors (reference vs actual), with logging on mismatch
auto
compare_outputs(
    const std::vector<torch::Tensor>& reference,
    const std::vector<torch::Tensor>& actual, const InferenceResult& result,
    const torch::Device& device) -> bool
{
  if (reference.size() != actual.size()) {
    log_error(
        "[Validator] Output count mismatch for job " +
        std::to_string(result.job_id));
    return false;
  }

  bool all_valid = true;
  for (size_t i = 0; i < reference.size(); ++i) {
    const torch::Tensor& ref = reference[i].to(device);
    const torch::Tensor& res = actual[i].to(device);

    const bool is_valid = torch::allclose(ref, res, 1e-3, 1e-5);
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

  return all_valid;
}

// =============================================================================
// Entry Point: Validate One Inference Result Against the Model
// =============================================================================

auto
validate_inference_result(
    const InferenceResult& result, torch::jit::script::Module& jit_model,
    const VerbosityLevel& verbosity) -> bool
{
  try {
    const torch::Device device = get_inference_device(result);

    auto input_ivalues = prepare_inputs(result.inputs, device);

    const torch::IValue output = jit_model.forward(input_ivalues);
    auto reference_outputs = extract_reference_outputs(output, result);

    if (reference_outputs.size() != result.results.size()) {
      log_error(
          "[Validator] Output count mismatch for job " +
          std::to_string(result.job_id));
      return false;
    }

    const bool all_valid =
        compare_outputs(reference_outputs, result.results, result, device);

    if (all_valid) {
      log_info(
          verbosity, "[Validator] Job " + std::to_string(result.job_id) +
                         " passed on " + to_string(result.executed_on) +
                         " (device id " + std::to_string(result.device_id) +
                         " worker id " + std::to_string(result.worker_id) +
                         ")");
    }

    return all_valid;
  }
  catch (const c10::Error& e) {
    log_error(
        "[Validator] C10 error in job " + std::to_string(result.job_id) + ": " +
        e.what());
  }
  catch (const std::exception& e) {
    log_error(
        "[Validator] Exception in job " + std::to_string(result.job_id) + ": " +
        e.what());
  }

  return false;
}
}  // namespace starpu_server