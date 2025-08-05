#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <starpu.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <chrono>
#include <climits>
#include <memory>

#include "core/inference_params.hpp"
#include "core/starpu_setup.hpp"
#include "core/tensor_builder.hpp"
#include "starpu_runtime_guard.hpp"
#include "test_utils.hpp"

class StarPUSetupCodeletTest : public ::testing::Test {
 protected:
  std::unique_ptr<starpu_server::StarPUSetup> starpu;
  void SetUp() override
  {
    if (!torch::cuda::is_available()) {
      GTEST_SKIP();
    }
    starpu_server::RuntimeConfig opts;
    opts.use_cpu = true;
    opts.use_cuda = true;
    opts.device_ids = {0};
    starpu = std::make_unique<starpu_server::StarPUSetup>(opts);
  }
};

TEST_F(StarPUSetupCodeletTest, GetCodeletNotNull)
{
  EXPECT_NE(starpu->get_codelet(), nullptr);
}

TEST_F(StarPUSetupCodeletTest, GetCudaWorkersSingleDevice)
{
  auto workers = starpu_server::StarPUSetup::get_cuda_workers_by_device({0});
  EXPECT_FALSE(workers.empty());
}

TEST(InferenceCodelet, FieldsAreInitialized)
{
  starpu_server::InferenceCodelet codelet;
  auto* cl = codelet.get_codelet();
  EXPECT_EQ(cl->nbuffers, STARPU_VARIABLE_NBUFFERS);
  EXPECT_NE(cl->cpu_funcs[0], nullptr);
  EXPECT_NE(cl->cuda_funcs[0], nullptr);
  EXPECT_EQ(cl->cuda_flags[0], 1U);
  EXPECT_EQ(cl->max_parallelism, INT_MAX);
}

TEST(InferenceCodelet, CpuInferenceFuncExecutesAndSetsMetadata)
{
  StarpuRuntimeGuard starpu_guard;
  auto buffers = make_test_buffers();
  auto timing = setup_timing_params(3);
  starpu_server::InferenceCodelet codelet;
  auto* cl = codelet.get_codelet();
  auto before = std::chrono::high_resolution_clock::now();
  cl->cpu_funcs[0](buffers.buffers, &timing.params);
  auto after = std::chrono::high_resolution_clock::now();
  EXPECT_FLOAT_EQ(buffers.output_data[0], 2.0f);
  EXPECT_FLOAT_EQ(buffers.output_data[1], 3.0f);
  EXPECT_FLOAT_EQ(buffers.output_data[2], 4.0f);
  EXPECT_EQ(timing.executed_on, starpu_server::DeviceType::CPU);
  EXPECT_TRUE(timing.start_time >= before);
  EXPECT_TRUE(timing.end_time <= after);
  EXPECT_TRUE(timing.end_time >= timing.start_time);
}
