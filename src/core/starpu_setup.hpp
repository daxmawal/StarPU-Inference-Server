#pragma once

#include <starpu.h>
#include <torch/script.h>

#include <map>
#include <string>
#include <vector>

#include "input_slot_pool.hpp"
#include "output_slot_pool.hpp"
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
  static void cpu_inference_func(void** buffers, void* cl_arg);
  static void cuda_inference_func(void** buffers, void* cl_arg);

  struct starpu_codelet codelet_;
};

auto extract_tensors_from_output(const c10::IValue& result)
    -> std::vector<at::Tensor>;

auto select_gpu_module(const InferenceParams& params, int device_id)
    -> torch::jit::script::Module*;

// =============================================================================
// Handles StarPU global configuration and codelet setup
// =============================================================================

class StarPUSetup {
 public:
  explicit StarPUSetup(const RuntimeConfig& opts);

  ~StarPUSetup();

  auto get_codelet() -> struct starpu_codelet*;

  auto input_pool() -> InputSlotPool& { return *input_pool_; }
  [[nodiscard]] bool has_input_pool() const
  {
    return static_cast<bool>(input_pool_);
  }
  auto output_pool() -> OutputSlotPool& { return *output_pool_; }
  [[nodiscard]] bool has_output_pool() const
  {
    return static_cast<bool>(output_pool_);
  }

  using StarpuInitFn = int (*)(struct starpu_conf*);
  using WorkerStreamQueryFn =
      int (*)(unsigned int, int*, enum starpu_worker_archtype);

  static void set_starpu_init_fn(StarpuInitFn fn);
  static void reset_starpu_init_fn();
  static void set_worker_stream_query_fn(WorkerStreamQueryFn fn);
  static void reset_worker_stream_query_fn();

  static auto get_cuda_workers_by_device(const std::vector<int>& device_ids)
      -> std::map<int, std::vector<int>>;

  StarPUSetup(const StarPUSetup&) = delete;
  auto operator=(const StarPUSetup&) -> StarPUSetup& = delete;
  StarPUSetup(StarPUSetup&&) = delete;
  auto operator=(StarPUSetup&&) -> StarPUSetup& = delete;

 private:
  std::string scheduler_name_;
  struct starpu_conf conf_;
  InferenceCodelet codelet_;
  std::unique_ptr<InputSlotPool> input_pool_;
  std::unique_ptr<OutputSlotPool> output_pool_;
};
}  // namespace starpu_server
