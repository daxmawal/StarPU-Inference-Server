#pragma once

#include "inference_runner.hpp"

namespace starpu_server {
// =============================================================================
// Compares the result of an inference job to a reference output generated
// from the same model on the same device. Returns true if close enough.
// =============================================================================
auto validate_inference_result(
    const InferenceResult& result, torch::jit::script::Module& jit_model,
    const VerbosityLevel& verbosity) -> bool;
}  // namespace starpu_server
