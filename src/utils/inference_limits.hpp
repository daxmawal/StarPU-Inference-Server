#pragma once

#include <cstddef>

namespace starpu_server { namespace InferLimits {
inline constexpr std::size_t MaxInputs = 16;
inline constexpr std::size_t MaxDims = 8;
inline constexpr std::size_t MaxModelsGPU = 32;
}}  // namespace starpu_server::InferLimits
