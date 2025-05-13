#include "inference_validator.hpp"

#include <iostream>

bool
validate_inference_result(
    const InferenceResult& r, torch::jit::script::Module& module)
{
  torch::Tensor ref = module.forward({r.input}).toTensor();
  bool is_valid = torch::allclose(ref, r.result, /*rtol=*/1e-3, /*atol=*/1e-5);

  if (!is_valid) {
    std::cerr << "[Validator] Mismatch detected for job " << r.job_id << "!\n";
    std::cerr << "  Reference: " << ref.flatten().slice(0, 0, 10) << "\n";
    std::cerr << "  Obtained : " << r.result.flatten().slice(0, 0, 10) << "\n";
  }

  return is_valid;
}