#pragma once

#include <starpu.h>

#include "args_parser.hpp"
#include "tensor_builder.hpp"

// ============================================================================
// InferenceCodelet: Wraps a StarPU codelet for TorchScript inference
// ============================================================================
class InferenceCodelet {
 public:
  // Constructor initializes the StarPU codelet
  InferenceCodelet();
  InferenceCodelet(const InferenceCodelet&) = delete;
  InferenceCodelet& operator=(const InferenceCodelet&) = delete;

  // Returns a pointer to the configured StarPU codelet
  struct starpu_codelet* get_codelet();

 private:
  // CPU and CUDA execution functions
  static inline void cpu_inference_func(void* buffers[], void* cl_arg);
  static inline void cuda_inference_func(void* buffers[], void* cl_arg);

  // Internal StarPU codelet
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
  struct starpu_codelet* codelet();

  // Returns the map of CUDA workers grouped by device ID
  const std::map<unsigned int, std::vector<int>> get_cuda_workers_by_device(
      const std::vector<unsigned int>& device_ids);

 private:
  struct starpu_conf conf_;   // StarPU configuration structure
  InferenceCodelet codelet_;  // Wrapped inference codelet

  // Prevent copy and move
  StarPUSetup(const StarPUSetup&) = delete;
  StarPUSetup& operator=(const StarPUSetup&) = delete;
  StarPUSetup(StarPUSetup&&) = delete;
  StarPUSetup& operator=(StarPUSetup&&) = delete;
};