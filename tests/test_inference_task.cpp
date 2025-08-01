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

TEST(InferenceTask, AssignFixedWorkerValid)
{
  const size_t num_inputs = 1;
  std::vector<torch::Tensor> inputs(num_inputs, torch::ones({1}));
  std::vector<at::ScalarType> types(num_inputs, at::kFloat);

  auto job = std::make_shared<InferenceJob>();
  job->set_job_id(3);
  job->set_input_tensors(inputs);
  job->set_input_types(types);
  job->set_outputs_tensors({torch::zeros({1})});
  job->set_fixed_worker_id(2);

  auto model_cpu = make_add_one_model();
  std::vector<torch::jit::script::Module> models_gpu;
  RuntimeConfig opts;
  InferenceTask task(nullptr, job, &model_cpu, &models_gpu, &opts);

  starpu_task task_struct{};
  task.assign_fixed_worker_if_needed(&task_struct);

  EXPECT_EQ(task_struct.workerid, 2u);
  EXPECT_EQ(task_struct.execute_on_a_specific_worker, 1u);
}

TEST(InferenceTask, CreateInferenceParamsPopulatesFields)
{
  auto job = std::make_shared<InferenceJob>();
  job->set_job_id(4);
  job->set_input_tensors({torch::ones({2, 3})});
  job->set_input_types({at::kFloat});
  job->set_outputs_tensors({torch::zeros({2, 3})});

  auto model_cpu = make_add_one_model();
  std::vector<torch::jit::script::Module> models_gpu;
  RuntimeConfig opts;
  opts.verbosity = VerbosityLevel::Debug;
  InferenceTask task(nullptr, job, &model_cpu, &models_gpu, &opts);

  auto params = task.create_inference_params();

  ASSERT_EQ(params->num_inputs, 1u);
  ASSERT_EQ(params->num_outputs, 1u);
  EXPECT_EQ(params->job_id, 4);
  EXPECT_EQ(params->verbosity, opts.verbosity);
  EXPECT_EQ(params->models.model_cpu, &model_cpu);
  EXPECT_EQ(params->models.num_models_gpu, 0u);
  EXPECT_EQ(params->device.device_id, &job->get_device_id());
  EXPECT_EQ(params->device.worker_id, &job->get_worker_id());
  EXPECT_EQ(params->device.executed_on, &job->get_executed_on());
  EXPECT_EQ(params->layout.num_dims[0], 2);
  EXPECT_EQ(params->layout.dims[0][0], 2);
  EXPECT_EQ(params->layout.dims[0][1], 3);
  EXPECT_EQ(params->layout.input_types[0], at::kFloat);
}

TEST(InferenceTask, SafeRegisterTensorVectorThrows)
{
  torch::Tensor undef;
  EXPECT_THROW(
      InferenceTask::safe_register_tensor_vector(undef, "x"),
      StarPURegistrationException);
}