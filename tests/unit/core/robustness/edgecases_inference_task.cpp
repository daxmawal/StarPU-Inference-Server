#include <gtest/gtest.h>
#include <starpu.h>
#include <torch/script.h>

#include <memory>
#include <string>
#include <vector>

#include "core/inference_task.hpp"
#include "utils/exceptions.hpp"

TEST(InferenceTask_Robustesse, SubmitNullJobThrows)
{
  auto job = std::shared_ptr<starpu_server::InferenceJob>(nullptr);
  torch::jit::script::Module model_cpu;
  std::vector<torch::jit::script::Module> models_gpu;
  starpu_server::RuntimeConfig opts;

  starpu_server::InferenceTask task(
      nullptr, job, &model_cpu, &models_gpu, &opts);
  EXPECT_THROW(task.submit(), starpu_server::InvalidInferenceJobException);
}

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

TEST_F(
    InferenceTaskLimitsTest_Robustesse,
    FillInputLayoutUsesEffectiveBatchSizeWhenProvided)
{
  auto tensor = torch::ones({2, 3});
  Job()->set_input_tensors({tensor});
  Job()->set_input_types({tensor.scalar_type()});
  Job()->set_effective_batch_size(7);

  starpu_server::TensorConfig tensor_config;
  tensor_config.dims = {4, 5};
  tensor_config.type = tensor.scalar_type();

  starpu_server::ModelConfig model_config;
  model_config.inputs.push_back(tensor_config);
  Opts().model = model_config;

  auto params = std::make_shared<starpu_server::InferenceParams>();
  ASSERT_NO_THROW(Task()->fill_input_layout(params, 1));
  ASSERT_EQ(params->layout.dims.size(), 1);
  ASSERT_EQ(params->layout.dims[0].size(), 2);
  EXPECT_EQ(params->layout.dims[0][0], 7);
}

TEST_F(
    InferenceTaskLimitsTest_Robustesse,
    FillInputLayoutDimsFromConfigRespectLimits)
{
  auto tensor = torch::ones({2, 3});
  Job()->set_input_tensors({tensor});
  Job()->set_input_types({tensor.scalar_type()});

  starpu_server::TensorConfig tensor_config;
  tensor_config.dims = {4, 5, 6};
  tensor_config.type = tensor.scalar_type();

  starpu_server::ModelConfig model_config;
  model_config.inputs.push_back(tensor_config);
  auto& opts = Opts();
  opts.model = model_config;
  opts.limits.max_dims = 2;

  auto params = std::make_shared<starpu_server::InferenceParams>();
  EXPECT_THROW(
      Task()->fill_input_layout(params, 1),
      starpu_server::InferenceExecutionException);
}
