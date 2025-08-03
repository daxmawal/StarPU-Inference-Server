#include <gtest/gtest.h>
#include <starpu.h>
#include <torch/script.h>

#include <climits>

#include "core/inference_params.hpp"
#include "core/starpu_setup.hpp"
#include "core/tensor_builder.hpp"

using namespace starpu_server;

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

TEST(StarPUSetupCodelet, GetCudaWorkersNegativeDeviceThrows)
{
  EXPECT_THROW(
      StarPUSetup::get_cuda_workers_by_device({-1}), std::invalid_argument);
}

TEST(InferenceCodelet, FieldsAreInitialized)
{
  InferenceCodelet codelet;
  auto* cl = codelet.get_codelet();

  EXPECT_EQ(cl->nbuffers, STARPU_VARIABLE_NBUFFERS);
  EXPECT_NE(cl->cpu_funcs[0], nullptr);
  EXPECT_NE(cl->cuda_funcs[0], nullptr);
  EXPECT_EQ(cl->cuda_flags[0], 1U);
  EXPECT_EQ(cl->max_parallelism, INT_MAX);
}
