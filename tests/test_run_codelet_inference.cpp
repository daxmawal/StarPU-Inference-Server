#include <gtest/gtest.h>
#include <starpu.h>
#include <torch/script.h>

#include <chrono>

#include "core/inference_params.hpp"
#include "core/starpu_setup.hpp"
#include "core/tensor_builder.hpp"
#include "utils/device_type.hpp"

using namespace starpu_server;

/* TODO there is a core dump
TEST(RunCodeletInference, ExecutesAndUpdatesParams)
{
  RuntimeConfig opts;
  opts.use_cpu = true;
  opts.use_cuda = false;
  StarPUSetup starpu(opts);

  torch::jit::script::Module model("m");
  model.define(R"JIT(
        def forward(self, x):
            return x + 1
    )JIT");

  InferenceParams params{};
  params.models.model_cpu = &model;
  params.num_inputs = 1;
  params.num_outputs = 1;
  params.layout.num_dims[0] = 1;
  params.layout.dims[0][0] = 3;
  params.layout.input_types[0] = at::kFloat;

  DeviceType executed_on = DeviceType::CUDA;
  int device_id = -1;
  int worker_id = -1;
  params.device.executed_on = &executed_on;
  params.device.device_id = &device_id;
  params.device.worker_id = &worker_id;

  std::chrono::high_resolution_clock::time_point start_time, end_time;
  params.timing.codelet_start_time = &start_time;
  params.timing.codelet_end_time = &end_time;

  float input[3] = {1.f, 2.f, 3.f};
  float output[3] = {0.f, 0.f, 0.f};

  starpu_data_handle_t input_handle;
  starpu_data_handle_t output_handle;
  starpu_variable_data_register(
      &input_handle, STARPU_MAIN_RAM, reinterpret_cast<uintptr_t>(input),
      sizeof(input));
  starpu_variable_data_register(
      &output_handle, STARPU_MAIN_RAM, reinterpret_cast<uintptr_t>(output),
      sizeof(output));

  int ret = starpu_task_insert(
      starpu.get_codelet(), STARPU_R, input_handle, STARPU_W, output_handle,
      STARPU_VALUE, &params, sizeof(InferenceParams), 0);
  ASSERT_EQ(ret, 0);

  starpu_task_wait_for_all();

  starpu_data_unregister(input_handle);
  starpu_data_unregister(output_handle);

  EXPECT_FLOAT_EQ(output[0], 2.f);
  EXPECT_FLOAT_EQ(output[1], 3.f);
  EXPECT_FLOAT_EQ(output[2], 4.f);

  EXPECT_EQ(executed_on, DeviceType::CPU);
  EXPECT_GE(device_id, 0);
  EXPECT_GE(worker_id, 0);

  auto zero = std::chrono::high_resolution_clock::time_point{};
  EXPECT_GT(start_time, zero);
  EXPECT_GT(end_time, zero);
  EXPECT_LE(start_time, end_time);
}
*/