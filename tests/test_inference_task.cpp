#include <gtest/gtest.h>
#include <starpu.h>
#include <torch/script.h>

#include "core/inference_task.hpp"
#include "utils/exceptions.hpp"

using namespace starpu_server;

static auto
make_add_one_model() -> torch::jit::script::Module
{
  torch::jit::script::Module m{"m"};
  m.define(R"JIT(
      def forward(self, x):
          return x + 1
  )JIT");
  return m;
}

TEST(InferenceTask, TooManyInputs)
{
  const size_t num_inputs = InferLimits::MaxInputs + 1;
  std::vector<torch::Tensor> inputs(num_inputs);
  std::vector<at::ScalarType> types(num_inputs, at::kFloat);
  for (size_t i = 0; i < num_inputs; ++i) {
    inputs[i] = torch::ones({1}, torch::TensorOptions().dtype(at::kFloat));
  }

  auto job = std::make_shared<InferenceJob>();
  job->set_job_id(0);
  job->set_input_tensors(inputs);
  job->set_input_types(types);
  job->set_outputs_tensors({torch::zeros({1})});

  auto model_cpu = make_add_one_model();
  std::vector<torch::jit::script::Module> models_gpu;
  RuntimeConfig opts;
  InferenceTask task(nullptr, job, &model_cpu, &models_gpu, &opts);

  EXPECT_THROW(task.create_inference_params(), InferenceExecutionException);
}

TEST(InferenceTask, TooManyGpuModels)
{
  const size_t num_inputs = 1;
  std::vector<torch::Tensor> inputs(num_inputs, torch::ones({1}));
  std::vector<at::ScalarType> types(num_inputs, at::kFloat);

  auto job = std::make_shared<InferenceJob>();
  job->set_job_id(1);
  job->set_input_tensors(inputs);
  job->set_input_types(types);
  job->set_outputs_tensors({torch::zeros({1})});

  auto model_cpu = make_add_one_model();
  std::vector<torch::jit::script::Module> models_gpu;
  for (size_t i = 0; i < InferLimits::MaxModelsGPU + 1; ++i) {
    models_gpu.push_back(make_add_one_model());
  }
  RuntimeConfig opts;
  InferenceTask task(nullptr, job, &model_cpu, &models_gpu, &opts);

  EXPECT_THROW(task.create_inference_params(), TooManyGpuModelsException);
}

TEST(InferenceTask, InvalidFixedWorker)
{
  const size_t num_inputs = 1;
  std::vector<torch::Tensor> inputs(num_inputs, torch::ones({1}));
  std::vector<at::ScalarType> types(num_inputs, at::kFloat);

  auto job = std::make_shared<InferenceJob>();
  job->set_job_id(2);
  job->set_input_tensors(inputs);
  job->set_input_types(types);
  job->set_outputs_tensors({torch::zeros({1})});
  job->set_fixed_worker_id(-1);

  auto model_cpu = make_add_one_model();
  std::vector<torch::jit::script::Module> models_gpu;
  RuntimeConfig opts;
  InferenceTask task(nullptr, job, &model_cpu, &models_gpu, &opts);

  starpu_task task_struct{};
  EXPECT_THROW(
      task.assign_fixed_worker_if_needed(&task_struct), std::invalid_argument);
}