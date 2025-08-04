#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <starpu.h>
#include <torch/script.h>

#include <chrono>
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

TEST(InferenceCodelet, CpuInferenceFuncExecutesAndSetsMetadata)
{
  starpu_init(nullptr);

  float input_data[3] = {1.0f, 2.0f, 3.0f};
  float output_data[3] = {0.0f, 0.0f, 0.0f};

  starpu_variable_interface input_iface;
  input_iface.ptr = reinterpret_cast<uintptr_t>(input_data);
  starpu_variable_interface output_iface;
  output_iface.ptr = reinterpret_cast<uintptr_t>(output_data);

  InferenceParams params{};
  params.num_inputs = 1;
  params.num_outputs = 1;
  params.layout.num_dims[0] = 1;
  params.layout.dims[0][0] = 3;
  params.layout.input_types[0] = at::kFloat;

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

  starpu_shutdown();
}

/*
TEST(InferenceCodelet, CudaInferenceFuncExecutesAndSetsMetadata)
{
  RuntimeConfig opts;
  opts.use_cuda = true;
  opts.device_ids = {0};

  starpu_conf conf;
  starpu_conf_init(&conf);
  conf.use_explicit_workers_cuda_gpuid = 1U;
  conf.ncuda = static_cast<int>(opts.device_ids.size());
  for (size_t idx = 0; idx < opts.device_ids.size(); ++idx) {
    conf.workers_cuda_gpuid[idx] =
        static_cast<unsigned int>(opts.device_ids[idx]);
  }
  starpu_init(&conf);

  float h_input[3] = {1.0f, 2.0f, 3.0f};
  float* d_input = nullptr;
  float* d_output = nullptr;
  cudaMalloc(&d_input, 3 * sizeof(float));
  cudaMalloc(&d_output, 3 * sizeof(float));
  cudaMemcpy(d_input, h_input, 3 * sizeof(float), cudaMemcpyHostToDevice);

  starpu_variable_interface input_iface;
  input_iface.ptr = reinterpret_cast<uintptr_t>(d_input);
  starpu_variable_interface output_iface;
  output_iface.ptr = reinterpret_cast<uintptr_t>(d_output);

  InferenceParams params{};
  params.num_inputs = 1;
  params.num_outputs = 1;
  params.layout.num_dims[0] = 1;
  params.layout.dims[0][0] = 3;
  params.layout.input_types[0] = at::kFloat;

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
  model.to(torch::Device(torch::kCUDA, 0));
  params.models.models_gpu[0] = &model;
  params.models.num_models_gpu = 1;

  void* buffers[] = {&input_iface, &output_iface};
  InferenceCodelet codelet;
  auto* cl = codelet.get_codelet();

  auto before = std::chrono::high_resolution_clock::now();
  cl->cuda_funcs[0](buffers, &params);
  cudaDeviceSynchronize();
  auto after = std::chrono::high_resolution_clock::now();

  float h_output[3] = {0.0f, 0.0f, 0.0f};
  cudaMemcpy(h_output, d_output, 3 * sizeof(float), cudaMemcpyDeviceToHost);

  EXPECT_FLOAT_EQ(h_output[0], 2.0f);
  EXPECT_FLOAT_EQ(h_output[1], 3.0f);
  EXPECT_FLOAT_EQ(h_output[2], 4.0f);

  EXPECT_EQ(executed_on, DeviceType::CUDA);

  EXPECT_TRUE(start_time >= before);
  EXPECT_TRUE(end_time <= after);
  EXPECT_TRUE(end_time >= start_time);

  cudaFree(d_input);
  cudaFree(d_output);
  starpu_shutdown();
}
*/