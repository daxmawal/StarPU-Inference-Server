#include <gtest/gtest.h>
#include <torch/script.h>

#include <functional>
#include <vector>

#include "core/inference_params.hpp"
#include "core/starpu_setup.hpp"
#include "utils/exceptions.hpp"

using namespace starpu_server;

namespace starpu_server {
void run_inference(
    InferenceParams* params, const std::vector<void*>& buffers,
    const torch::Device device, torch::jit::script::Module* model,
    const std::function<void(const at::Tensor&, void* buffer_ptr)>&
        copy_output_fn);
}  // namespace starpu_server

TEST(StarPUSetupErrorsTest, ConstructorNegativeDeviceId)
{
  RuntimeConfig cfg;
  cfg.use_cuda = true;
  cfg.device_ids = {-1};
  EXPECT_THROW(StarPUSetup setup(cfg), std::invalid_argument);
}

TEST(StarPUSetupErrorsTest, RunInferenceMoreOutputsThanBuffers)
{
  torch::jit::script::Module model;
  model.define(
      "def forward(self):\n    return torch.tensor([1]), torch.tensor([2])");

  InferenceParams params;
  params.models.model_cpu = &model;
  params.num_outputs = 1;

  std::vector<void*> buffers(1, nullptr);

  auto call_run_inference = [&]() {
    run_inference(
        &params, buffers, torch::kCPU, &model, [](const at::Tensor&, void*) {});
  };

  EXPECT_THROW(
      try {
        call_run_inference();
      } catch (const UnsupportedModelOutputTypeException&) {
        throw;
      } catch (const std::exception& e) {
        throw StarPUCodeletException(e.what());
      },
      StarPUCodeletException);
}

TEST(StarPUSetupErrorsTest, GetCudaWorkersByDeviceNegativeId)
{
  EXPECT_THROW(
      StarPUSetup::get_cuda_workers_by_device({-1}), std::invalid_argument);
}
