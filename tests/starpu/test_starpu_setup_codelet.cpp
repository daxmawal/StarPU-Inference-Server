#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <starpu.h>
#include <torch/script.h>

#include <chrono>
#include <climits>
#include <memory>

#include "../test_helpers.hpp"
#include "core/inference_params.hpp"
#include "core/starpu_setup.hpp"
#include "core/tensor_builder.hpp"
#include "starpu_runtime_guard.hpp"

struct TestBuffers {
  float input_data[3];
  float output_data[3];
  starpu_variable_interface input_iface;
  starpu_variable_interface output_iface;
  void* buffers[2];
};

TestBuffers
make_test_buffers()
{
  TestBuffers t{};
  t.input_data[0] = 1.0f;
  t.input_data[1] = 2.0f;
  t.input_data[2] = 3.0f;
  t.output_data[0] = t.output_data[1] = t.output_data[2] = 0.0f;
  t.input_iface = starpu_server::make_variable_interface(t.input_data);
  t.output_iface = starpu_server::make_variable_interface(t.output_data);
  t.buffers[0] = &t.input_iface;
  t.buffers[1] = &t.output_iface;
  return t;
}

struct TimingParams {
  starpu_server::InferenceParams params;
  starpu_server::DeviceType executed_on = starpu_server::DeviceType::Unknown;
  std::chrono::high_resolution_clock::time_point start_time;
  std::chrono::high_resolution_clock::time_point end_time;
  torch::jit::script::Module model;
};

TimingParams
setup_timing_params(int elements)
{
  TimingParams t{starpu_server::make_basic_params(elements)};
  t.params.device.executed_on = &t.executed_on;
  t.params.timing.codelet_start_time = &t.start_time;
  t.params.timing.codelet_end_time = &t.end_time;
  t.model = torch::jit::script::Module("m");
  t.model.define(R"JIT(
        def forward(self, x):
            return x + 1
    )JIT");
  t.params.models.model_cpu = &t.model;
  return t;
}

class StarPUSetupCodeletTest : public ::testing::Test {
 protected:
  std::unique_ptr<starpu_server::StarPUSetup> starpu;

  void SetUp() override
  {
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

TEST(StarPUSetupCodelet, GetCudaWorkersNegativeDeviceThrows)
{
  EXPECT_THROW(
      starpu_server::StarPUSetup::get_cuda_workers_by_device({-1}),
      std::invalid_argument);
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
