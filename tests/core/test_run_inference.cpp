#include <gtest/gtest.h>
#include <torch/script.h>

#include <chrono>
#include <functional>
#include <vector>

#include "core/inference_params.hpp"
#include "core/tensor_builder.hpp"
#include "test_helpers.hpp"

namespace starpu_server {
void run_inference(
    InferenceParams* params, const std::vector<void*>& buffers,
    const torch::Device device, torch::jit::script::Module* model,
    const std::function<void(const at::Tensor&, void* buffer_ptr)>&
        copy_output_fn);
}

TEST(StarPUSetupRunInference, BuildsExecutesCopiesAndTimes)
{
  std::array<float, 3> input{1.0F, 2.0F, 3.0F};
  std::array<float, 3> output{0.0F, 0.0F, 0.0F};
  auto input_iface = starpu_server::make_variable_interface(input.data());
  auto output_iface = starpu_server::make_variable_interface(output.data());
  auto params = starpu_server::make_basic_params(3);
  std::chrono::high_resolution_clock::time_point inference_start;
  params.timing.inference_start_time = &inference_start;
  std::vector<void*> buffers = {&input_iface, &output_iface};
  auto model = starpu_server::make_add_one_model();
  auto before = std::chrono::high_resolution_clock::now();
  starpu_server::run_inference(
      &params, buffers, torch::Device(torch::kCPU), &model,
      [](const at::Tensor& out, void* buffer_ptr) {
        starpu_server::TensorBuilder::copy_output_to_buffer(
            out, buffer_ptr, out.numel());
      });
  auto after = std::chrono::high_resolution_clock::now();
  EXPECT_FLOAT_EQ(output[0], 2.0F);
  EXPECT_FLOAT_EQ(output[1], 3.0F);
  EXPECT_FLOAT_EQ(output[2], 4.0F);
  EXPECT_TRUE(inference_start >= before);
  EXPECT_TRUE(inference_start <= after);
}
