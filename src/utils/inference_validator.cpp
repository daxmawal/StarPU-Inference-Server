#include "inference_validator.hpp"

#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cstddef>
#include <format>
#include <iterator>
#include <vector>

#include "device_type.hpp"
#include "exceptions.hpp"
#include "logger.hpp"

namespace starpu_server {
constexpr int kPreviewLimit = 10;

// =============================================================================
// Device and Input Preparation Utilities
// =============================================================================

static auto
get_inference_device(const InferenceResult& result) -> torch::Device
{
  switch (result.executed_on) {
    case DeviceType::CUDA: {
      const int idx = (result.device_id >= 0) ? result.device_id : 0;
      return {torch::kCUDA, static_cast<c10::DeviceIndex>(idx)};
    }
    case DeviceType::CPU:
      return {torch::kCPU};
    default:
      log_error(std::format(
          "[Validator] Unknown device for job {}", result.request_id));
      throw InferenceExecutionException("Unknown device");
  }
}

static auto
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

static auto
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
        log_error(std::format(
            "[Validator] Non-tensor output in tuple for job {}",
            result.request_id));
        throw InferenceExecutionException("Non-tensor tuple element");
      }
      tensors.push_back(val.toTensor());
    }
  } else if (output.isTensorList()) {
    for (const auto& tensor : output.toTensorList()) {
      tensors.push_back(tensor);
    }
  } else {
    log_error(std::format(
        "[Validator] Unsupported output type for job {}", result.request_id));
    throw InferenceExecutionException("Unsupported output type");
  }

  return tensors;
}

static auto
compare_outputs(
    const std::vector<torch::Tensor>& reference,
    const std::vector<torch::Tensor>& actual, const InferenceResult& result,
    const torch::Device& device, double rtol, double atol) -> bool
{
  if (reference.size() != actual.size()) {
    log_error(std::format(
        "[Validator] Output count mismatch for job {}", result.request_id));
    return false;
  }

  bool all_valid = true;
  for (size_t i = 0; i < reference.size(); ++i) {
    const auto ref_dev = reference[i].to(device);
    const auto res_dev = actual[i].to(device);

    const auto ref_cmp = ref_dev.to(torch::kFloat);
    const auto res_cmp = res_dev.to(torch::kFloat);

    const bool is_valid = torch::allclose(ref_cmp, res_cmp, rtol, atol);
    all_valid &= is_valid;

    if (!is_valid) {
      log_error(std::format(
          "[Validator] Mismatch on output #{} for job {}", i,
          result.request_id));
      log_error(std::format(
          "  Absolute diff max: {}",
          (ref_cmp - res_cmp).abs().max().item<float>()));
      log_error(std::format(
          "  Reference: {}",
          ref_cmp.flatten().slice(0, 0, kPreviewLimit).toString()));
      log_error(std::format(
          "  Obtained : {}",
          res_cmp.flatten().slice(0, 0, kPreviewLimit).toString()));
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
    const VerbosityLevel& verbosity, double rtol, double atol) -> bool
{
  try {
    const torch::Device device = get_inference_device(result);

    c10::cuda::OptionalCUDAGuard device_guard;
    if (device.is_cuda()) {
      device_guard.reset_device(device);
    }

    auto input_ivalues = prepare_inputs(result.inputs, device);

    const torch::IValue output = jit_model.forward(input_ivalues);
    auto reference_outputs = extract_reference_outputs(output, result);

    const bool all_valid = compare_outputs(
        reference_outputs, result.results, result, device, rtol, atol);

    if (all_valid) {
      log_info(
          verbosity,
          std::format(
              "[Validator] Job {} passed on {} (device id {} worker id {})",
              result.request_id, to_string(result.executed_on),
              result.device_id, result.worker_id));
    }

    return all_valid;
  }
  catch (const c10::Error& e) {
    log_error(std::format(
        "[Validator] C10 error in job {}: {}", result.request_id, e.what()));
    throw InferenceExecutionException(e.what());
  }
  catch (const std::exception& e) {
    log_error(std::format(
        "[Validator] Exception in job {}: {}", result.request_id, e.what()));
    throw InferenceExecutionException(e.what());
  }
}
}  // namespace starpu_server
