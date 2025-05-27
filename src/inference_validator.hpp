#pragma once

#include "inference_runner.hpp"

bool validate_inference_result(
    const InferenceResult& r, torch::jit::script::Module& module,
    const VerbosityLevel& verbosity);