#include <gtest/gtest.h>
#include <starpu.h>
#include <torch/script.h>

#include <memory>
#include <string>
#include <vector>

#include "core/inference_task.hpp"
#include "utils/exceptions.hpp"

class InferenceTaskLimitsTest_Robustesse : public ::testing::Test {
 protected:
  void SetUp() override
  {
    job_ = std::make_shared<starpu_server::InferenceJob>();
    model_cpu_ = torch::jit::script::Module{"m"};
    models_gpu_.clear();
    opts_ = starpu_server::RuntimeConfig{};
    task_ = std::make_unique<starpu_server::InferenceTask>(
        nullptr, job_, &model_cpu_, &models_gpu_, &opts_);
  }

  auto ModelsGpu() -> std::vector<torch::jit::script::Module>&
  {
    return models_gpu_;
  }
  auto Task() -> std::unique_ptr<starpu_server::InferenceTask>&
  {
    return task_;
  }
  auto Job() -> std::shared_ptr<starpu_server::InferenceJob>& { return job_; }
  auto ModelCpu() -> torch::jit::script::Module& { return model_cpu_; }
  auto Opts() -> starpu_server::RuntimeConfig& { return opts_; }

 private:
  std::shared_ptr<starpu_server::InferenceJob> job_;
  torch::jit::script::Module model_cpu_;
  std::vector<torch::jit::script::Module> models_gpu_;
  starpu_server::RuntimeConfig opts_;
  std::unique_ptr<starpu_server::InferenceTask> task_;
};

TEST_F(InferenceTaskLimitsTest_Robustesse, CheckLimitsTooManyInputs)
{
  const size_t num_inputs = starpu_server::InferLimits::MaxInputs + 1;
  EXPECT_THROW(
      Task()->check_limits(num_inputs),
      starpu_server::InferenceExecutionException);
}

TEST_F(InferenceTaskLimitsTest_Robustesse, CheckLimitsTooManyGpuModels)
{
  auto& models_gpu = ModelsGpu();
  models_gpu.reserve(starpu_server::InferLimits::MaxModelsGPU + 1);
  for (size_t i = 0; i < starpu_server::InferLimits::MaxModelsGPU + 1; ++i) {
    models_gpu.emplace_back(std::string{"m"} + std::to_string(i));
  }
  EXPECT_THROW(
      Task()->check_limits(1), starpu_server::TooManyGpuModelsException);
}

TEST_F(InferenceTaskLimitsTest_Robustesse, FillInputLayoutTooManyDims)
{
  std::vector<int64_t> dims(starpu_server::InferLimits::MaxDims + 1, 1);
  auto tensor = torch::ones(dims);
  Job()->set_input_tensors({tensor});
  Job()->set_input_types({tensor.scalar_type()});

  auto params = std::make_shared<starpu_server::InferenceParams>();
  EXPECT_THROW(
      Task()->fill_input_layout(params, 1),
      starpu_server::InferenceExecutionException);
}
