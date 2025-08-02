#include <gtest/gtest.h>
#include <starpu.h>
#include <torch/script.h>

#include "core/inference_params.hpp"
#include "core/starpu_setup.hpp"
#include "core/tensor_builder.hpp"

using namespace starpu_server;

static auto
make_add_one_model() -> torch::jit::script::Module
{
  torch::jit::script::Module m{"m"};
  m.define(R"JIT(
      def forward(self, x):
          return x + 1
  )JIT");
  return m;
}

TEST(StarPUSetupCodelet, GetCodeletNotNull)
{
  RuntimeConfig opts;
  opts.use_cpu = true;
  opts.use_cuda = false;
  StarPUSetup starpu(opts);
  EXPECT_NE(starpu.get_codelet(), nullptr);
}

TEST(StarPUSetupCodelet, GetCudaWorkersSingleDevice)
{
  RuntimeConfig opts;
  opts.use_cpu = true;
  opts.use_cuda = true;
  opts.device_ids = {0};
  StarPUSetup starpu(opts);
  auto workers = StarPUSetup::get_cuda_workers_by_device({0});
  EXPECT_FALSE(workers.empty());
}
