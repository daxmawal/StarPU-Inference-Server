#include <gtest/gtest.h>
#include <torch/script.h>

#include <chrono>
#include <functional>
#include <vector>

#include "../test_helpers.hpp"
#include "core/inference_params.hpp"
#include "core/tensor_builder.hpp"

namespace starpu_server {
void run_inference(
    InferenceParams* params, const std::vector<void*>& buffers,
    const torch::Device device, torch::jit::script::Module* model,
    const std::function<void(const at::Tensor&, void* buffer_ptr)>&
        copy_output_fn);
}

TEST(StarPUSetupRunInference, BuildsExecutesCopiesAndTimes)
{
  float input_data[3] = {1.0f, 2.0f, 3.0f};
  float output_data[3] = {0.0f, 0.0f, 0.0f};

  auto input_iface = starpu_server::make_variable_interface(input_data);
  auto output_iface = starpu_server::make_variable_interface(output_data);

  auto params = starpu_server::make_basic_params(3);

  std::chrono::high_resolution_clock::time_point inference_start;
  params.timing.inference_start_time = &inference_start;

  std::vector<void*> buffers = {&input_iface, &output_iface};

  torch::jit::script::Module model("m");
  model.define(R"JIT(
        def forward(self, x):
            return x + 1
    )JIT");

  auto before = std::chrono::high_resolution_clock::now();
  starpu_server::run_inference(
      &params, buffers, torch::Device(torch::kCPU), &model,
      [](const at::Tensor& out, void* buffer_ptr) {
        starpu_server::TensorBuilder::copy_output_to_buffer(
            out, buffer_ptr, out.numel());
      });
  auto after = std::chrono::high_resolution_clock::now();

  EXPECT_FLOAT_EQ(output_data[0], 2.0f);
  EXPECT_FLOAT_EQ(output_data[1], 3.0f);
  EXPECT_FLOAT_EQ(output_data[2], 4.0f);

  EXPECT_TRUE(inference_start >= before);
  EXPECT_TRUE(inference_start <= after);
}
