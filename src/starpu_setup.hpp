#pragma once

#include <starpu.h>

#include "args_parser.hpp"
#include "tensor_builder.hpp"

// ============================================================================
// InferenceCodelet: Wraps a StarPU codelet for TorchScript inference
// ============================================================================
class InferenceCodelet {
 public:
  InferenceCodelet();
  InferenceCodelet(const InferenceCodelet&) = delete;
  auto operator=(const InferenceCodelet&) -> InferenceCodelet& = delete;
  InferenceCodelet(InferenceCodelet&&) = delete;
  auto operator=(InferenceCodelet&&) -> InferenceCodelet& = delete;
  ~InferenceCodelet() = default;

  auto get_codelet() -> struct starpu_codelet*;

 private:
  static inline void cpu_inference_func(void* buffers[], void* cl_arg);
  static inline void cuda_inference_func(void* buffers[], void* cl_arg);

  struct starpu_codelet codelet_;
};

// ============================================================================
// StarPUSetup: Configures and manages the StarPU runtime environment
// ============================================================================
class StarPUSetup {
 public:
  // Initializes StarPU using the provided program options
  explicit StarPUSetup(const ProgramOptions& opts);

  // Shuts down StarPU on destruction
  ~StarPUSetup();

  // Returns the associated inference codelet
  auto get_codelet() -> struct starpu_codelet*;

  // Returns the map of CUDA workers grouped by device ID
  static auto get_cuda_workers_by_device(
      const std::vector<unsigned int>& device_ids)
      -> std::map<unsigned int, std::vector<int>>;

  // Prevent copy and move
  StarPUSetup(const StarPUSetup&) = delete;
  auto operator=(const StarPUSetup&) -> StarPUSetup& = delete;
  StarPUSetup(StarPUSetup&&) = delete;
  auto operator=(StarPUSetup&&) -> StarPUSetup& = delete;

 private:
  struct starpu_conf conf_;   // StarPU configuration structure
  InferenceCodelet codelet_;  // Wrapped inference codelet
};