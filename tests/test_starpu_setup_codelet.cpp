#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <starpu.h>
#include <torch/script.h>

#include <chrono>
#include <climits>
#include <memory>

#include "core/inference_params.hpp"
#include "core/starpu_setup.hpp"
#include "core/tensor_builder.hpp"
#include "starpu_runtime_guard.hpp"
#include "test_helpers.hpp"

using namespace starpu_server;

class StarPUSetupCodeletTest : public ::testing::Test {
 protected:
  std::unique_ptr<StarPUSetup> starpu;

  void SetUp() override
  {
    RuntimeConfig opts;
    opts.use_cpu = true;
    opts.use_cuda = true;
    opts.device_ids = {0};
    starpu = std::make_unique<StarPUSetup>(opts);
  }
};

TEST_F(StarPUSetupCodeletTest, GetCodeletNotNull)
{
  EXPECT_NE(starpu->get_codelet(), nullptr);
}

TEST_F(StarPUSetupCodeletTest, GetCudaWorkersSingleDevice)
{
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

TEST(InferenceCodelet, CpuInferenceFuncExecutesAndSetsMetadata)
{
  StarpuRuntimeGuard starpu_guard;

  float input_data[3] = {1.0f, 2.0f, 3.0f};
  float output_data[3] = {0.0f, 0.0f, 0.0f};

  auto input_iface = make_variable_interface(input_data);
  auto output_iface = make_variable_interface(output_data);

  auto params = make_basic_params(3);

  DeviceType executed_on = DeviceType::Unknown;
  params.device.executed_on = &executed_on;

  std::chrono::high_resolution_clock::time_point start_time;
  std::chrono::high_resolution_clock::time_point end_time;
  params.timing.codelet_start_time = &start_time;
  params.timing.codelet_end_time = &end_time;

  torch::jit::script::Module model("m");
  model.define(R"JIT(
        def forward(self, x):
            return x + 1
    )JIT");
  params.models.model_cpu = &model;

  void* buffers[] = {&input_iface, &output_iface};
  InferenceCodelet codelet;
  auto* cl = codelet.get_codelet();

  auto before = std::chrono::high_resolution_clock::now();
  cl->cpu_funcs[0](buffers, &params);
  auto after = std::chrono::high_resolution_clock::now();

  EXPECT_FLOAT_EQ(output_data[0], 2.0f);
  EXPECT_FLOAT_EQ(output_data[1], 3.0f);
  EXPECT_FLOAT_EQ(output_data[2], 4.0f);

  EXPECT_EQ(executed_on, DeviceType::CPU);

  EXPECT_TRUE(start_time >= before);
  EXPECT_TRUE(end_time <= after);
  EXPECT_TRUE(end_time >= start_time);
}
