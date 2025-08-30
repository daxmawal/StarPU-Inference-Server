#include "test_inference_task.hpp"

TEST_F(InferenceTaskTest, TooManyInputsThrows)
{
  const size_t inputs_nb = starpu_server::InferLimits::MaxInputs + 1;
  auto job = make_job(0, inputs_nb);
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

TEST_F(InferenceTaskTest, AssignFixedWorkerOutOfRangeThrows)
{
  auto job = make_job(3, 1);
  const unsigned int total_workers = starpu_worker_get_count();
  job->set_fixed_worker_id(static_cast<int>(total_workers));
  auto task = make_task(job);
  starpu_task task_struct{};
  EXPECT_THROW(
      task.assign_fixed_worker_if_needed(&task_struct), std::out_of_range);
}
