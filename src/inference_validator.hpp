#pragma once

// ----------------------------------------------------------------------------
// inference_validator.hpp
// Provides a function to validate inference results against reference output
// ----------------------------------------------------------------------------
#include "inference_runner.hpp"


// =============================================================================
// Compares the result of an inference job to a reference output generated
// from the same model on the same device. Returns true if close enough.
// =============================================================================
bool validate_inference_result(
    const InferenceResult& r, torch::jit::script::Module& module,
    const VerbosityLevel& verbosity);