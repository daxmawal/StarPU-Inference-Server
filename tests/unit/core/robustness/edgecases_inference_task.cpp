#include "test_inference_task.hpp"

TEST_F(InferenceTaskTest, TooManyInputsThrows)
{
  const size_t n = starpu_server::InferLimits::MaxInputs + 1;
  auto job = make_job(0, n);
  auto task = make_task(job);
  EXPECT_THROW(
      task.create_inference_params(),
      starpu_server::InferenceExecutionException);
}

TEST_F(InferenceTaskTest, TooManyGpuModelsThrows)
{
  auto job = make_job(1, 1);
  auto task = make_task(job, starpu_server::InferLimits::MaxModelsGPU + 1);
  EXPECT_THROW(
      task.create_inference_params(), starpu_server::TooManyGpuModelsException);
}

TEST_F(InferenceTaskTest, AssignFixedWorkerNegativeThrows)
{
  auto job = make_job(2, 1);
  job->set_fixed_worker_id(-1);
  auto task = make_task(job);
  starpu_task task_struct{};
  EXPECT_THROW(
      task.assign_fixed_worker_if_needed(&task_struct), std::invalid_argument);
}

TEST(InferenceTaskTest, SafeRegisterTensorVectorUndefinedTensorThrows)
{
  torch::Tensor undef;
  EXPECT_THROW(
      starpu_server::InferenceTask::safe_register_tensor_vector(undef, "x"),
      starpu_server::StarPURegistrationException);
}

TEST(InferenceTaskTest, SafeRegisterTensorVectorGpuTensorThrows)
{
  SKIP_IF_NO_CUDA();
  auto tensor = torch::ones(
      {1}, torch::TensorOptions().dtype(at::kFloat).device(torch::kCUDA));
  EXPECT_THROW(
      starpu_server::InferenceTask::safe_register_tensor_vector(tensor, "x"),
      starpu_server::StarPURegistrationException);
}
