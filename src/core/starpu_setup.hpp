#pragma once

#include <starpu.h>

#include "runtime_config.hpp"
#include "tensor_builder.hpp"

namespace starpu_server {
// =============================================================================
// Encapsulates a StarPU codelet for CPU/GPU inference execution
// =============================================================================

class InferenceCodelet {
 public:
  InferenceCodelet();
  ~InferenceCodelet() = default;
  InferenceCodelet(const InferenceCodelet&) = delete;
  auto operator=(const InferenceCodelet&) -> InferenceCodelet& = delete;
  InferenceCodelet(InferenceCodelet&&) = delete;
  auto operator=(InferenceCodelet&&) -> InferenceCodelet& = delete;

  auto get_codelet() -> struct starpu_codelet*;

 private:
  static inline void cpu_inference_func(void* buffers[], void* cl_arg);
  static inline void cuda_inference_func(void* buffers[], void* cl_arg);

  struct starpu_codelet codelet_;
};

// =============================================================================
// Handles StarPU global configuration and codelet setup
// =============================================================================

class StarPUSetup {
 public:
  explicit StarPUSetup(const RuntimeConfig& opts);

  ~StarPUSetup();

  auto get_codelet() -> struct starpu_codelet*;

  static auto get_cuda_workers_by_device(const std::vector<int>& device_ids)
      -> std::map<int, std::vector<int>>;

  StarPUSetup(const StarPUSetup&) = delete;
  auto operator=(const StarPUSetup&) -> StarPUSetup& = delete;
  StarPUSetup(StarPUSetup&&) = delete;
  auto operator=(StarPUSetup&&) -> StarPUSetup& = delete;

 private:
  struct starpu_conf conf_;
  InferenceCodelet codelet_;
};
}  // namespace starpu_server
