#pragma once
#include "args_parser.hpp"
#include "tensor_builder.hpp"

class InferenceCodelet {
 public:
  InferenceCodelet();
  struct starpu_codelet* get_codelet();

 private:
  static inline void cpu_inference_func(void* buffers[], void* cl_arg);
  static inline void cuda_inference_func(void* buffers[], void* cl_arg);

  struct starpu_codelet codelet_;
};

class StarPUSetup {
 public:
  explicit StarPUSetup(const ProgramOptions& opts);
  ~StarPUSetup();

  struct starpu_codelet* codelet();

 private:
  struct starpu_conf conf_;
  InferenceCodelet codelet_;
};