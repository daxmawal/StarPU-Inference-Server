#include <gtest/gtest.h>
#include <starpu.h>
#include <torch/script.h>

#include <limits>
#include <vector>

#include "core/inference_task.hpp"
#include "inference_runner_test_utils.hpp"
#include "utils/exceptions.hpp"

namespace {
int unregister_call_count = 0;
std::vector<starpu_data_handle_t> unregister_handles;
}  // namespace

extern "C" void
starpu_data_unregister_submit(starpu_data_handle_t handle)
{
  ++unregister_call_count;
  unregister_handles.push_back(handle);
}

class InferenceTaskTest : public ::testing::Test {
 protected:
  auto make_job(int job_id, size_t num_inputs, bool set_outputs = true)
      -> std::shared_ptr<starpu_server::InferenceJob>
  {
    auto job = std::make_shared<starpu_server::InferenceJob>();
    job->set_job_id(job_id);
    std::vector<torch::Tensor> inputs(num_inputs);
    std::vector<at::ScalarType> types(num_inputs, at::kFloat);
    for (size_t i = 0; i < num_inputs; ++i) {
      inputs[i] = torch::ones({1}, torch::TensorOptions().dtype(at::kFloat));
    }
    job->set_input_tensors(inputs);
    job->set_input_types(types);
    if (set_outputs) {
      job->set_outputs_tensors({torch::zeros({1})});
    }
    return job;
  }

  auto make_task(
      const std::shared_ptr<starpu_server::InferenceJob>& job,
      size_t num_gpu_models = 0) -> starpu_server::InferenceTask
  {
    model_cpu_ = starpu_server::make_add_one_model();
    models_gpu_.clear();
    for (size_t i = 0; i < num_gpu_models; ++i) {
      models_gpu_.push_back(starpu_server::make_add_one_model());
    }
    opts_ = starpu_server::RuntimeConfig{};
    return starpu_server::InferenceTask(
        nullptr, job, &model_cpu_, &models_gpu_, &opts_);
  }
  torch::jit::script::Module model_cpu_;
  std::vector<torch::jit::script::Module> models_gpu_;
  starpu_server::RuntimeConfig opts_;
};

TEST_F(InferenceTaskTest, TooManyInputs)
{
  const size_t num_inputs = starpu_server::InferLimits::MaxInputs + 1;
  auto job = make_job(0, num_inputs);
  auto task = make_task(job);
  EXPECT_THROW(
      task.create_inference_params(),
      starpu_server::InferenceExecutionException);
}

TEST_F(InferenceTaskTest, TooManyGpuModels)
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

TEST_F(InferenceTaskTest, AssignFixedWorkerValid)
{
  auto job = make_job(3, 1);
  job->set_fixed_worker_id(2);
  auto task = make_task(job);
  starpu_task task_struct{};
  task.assign_fixed_worker_if_needed(&task_struct);
  EXPECT_EQ(task_struct.workerid, 2u);
  EXPECT_EQ(task_struct.execute_on_a_specific_worker, 1u);
}

TEST_F(InferenceTaskTest, CreateInferenceParamsPopulatesFields)
{
  auto job = make_job(4, 1);
  job->set_input_tensors({torch::ones({2, 3})});
  job->set_outputs_tensors({torch::zeros({2, 3})});
  auto task = make_task(job);
  opts_.verbosity = starpu_server::VerbosityLevel::Debug;
  auto params = task.create_inference_params();
  ASSERT_EQ(params->num_inputs, 1u);
  ASSERT_EQ(params->num_outputs, 1u);
  EXPECT_EQ(params->job_id, 4);
  EXPECT_EQ(params->verbosity, opts_.verbosity);
  EXPECT_EQ(params->models.model_cpu, &model_cpu_);
  EXPECT_EQ(params->models.num_models_gpu, 0u);
  EXPECT_EQ(params->device.device_id, &job->get_device_id());
  EXPECT_EQ(params->device.worker_id, &job->get_worker_id());
  EXPECT_EQ(params->device.executed_on, &job->get_executed_on());
  EXPECT_EQ(params->layout.num_dims[0], 2);
  EXPECT_EQ(params->layout.dims[0][0], 2);
  EXPECT_EQ(params->layout.dims[0][1], 3);
  EXPECT_EQ(params->layout.input_types[0], at::kFloat);
}

TEST(InferenceTask, SafeRegisterTensorVectorUndefinedTensorThrows)
{
  torch::Tensor undef;
  EXPECT_THROW(
      starpu_server::InferenceTask::safe_register_tensor_vector(undef, "x"),
      starpu_server::StarPURegistrationException);
}

TEST(InferenceTask, RecordAndRunCompletionCallback)
{
  auto job = std::make_shared<starpu_server::InferenceJob>();
  std::vector<torch::Tensor> outputs{torch::tensor({1})};
  job->set_outputs_tensors(outputs);
  bool called = false;
  std::vector<torch::Tensor> results_arg;
  double latency_ms = -1.0;
  job->set_on_complete(
      [&called, &results_arg, &latency_ms](
          const std::vector<torch::Tensor>& results, double latency) {
        called = true;
        results_arg = results;
        latency_ms = latency;
      });
  const auto start = std::chrono::high_resolution_clock::now();
  const auto end = start + std::chrono::milliseconds(5);
  job->set_start_time(start);
  starpu_server::RuntimeConfig opts;
  starpu_server::InferenceCallbackContext ctx(job, nullptr, &opts, 0, {}, {});
  starpu_server::InferenceTask::record_and_run_completion_callback(&ctx, end);
  EXPECT_TRUE(called);
  ASSERT_EQ(results_arg.size(), outputs.size());
  EXPECT_TRUE(torch::equal(results_arg[0], outputs[0]));
  const double expected_latency =
      std::chrono::duration<double, std::milli>(end - start).count();
  EXPECT_DOUBLE_EQ(latency_ms, expected_latency);
}

TEST(InferenceTask, CleanupUnregistersAndNullsHandles)
{
  unregister_call_count = 0;
  unregister_handles.clear();
  const auto h1 = reinterpret_cast<starpu_data_handle_t>(0x1);
  const auto h2 = reinterpret_cast<starpu_data_handle_t>(0x2);
  const auto h3 = reinterpret_cast<starpu_data_handle_t>(0x3);
  std::vector<starpu_data_handle_t> inputs{h1};
  std::vector<starpu_data_handle_t> outputs{h2, h3};
  auto ctx = std::make_shared<starpu_server::InferenceCallbackContext>(
      nullptr, nullptr, nullptr, 0, inputs, outputs);
  starpu_server::InferenceTask::cleanup(ctx);
  EXPECT_EQ(unregister_call_count, 3);
  ASSERT_EQ(unregister_handles.size(), 3u);
  EXPECT_EQ(unregister_handles[0], h1);
  EXPECT_EQ(unregister_handles[1], h2);
  EXPECT_EQ(unregister_handles[2], h3);
  EXPECT_EQ(ctx->inputs_handles[0], nullptr);
  EXPECT_EQ(ctx->outputs_handles[0], nullptr);
  EXPECT_EQ(ctx->outputs_handles[1], nullptr);
}

TEST(InferenceTaskBuffers, FillTaskBuffersOrdersDynHandlesAndModes)
{
  auto ctx = std::make_shared<starpu_server::InferenceCallbackContext>(
      nullptr, nullptr, nullptr, 0, std::vector<starpu_data_handle_t>{},
      std::vector<starpu_data_handle_t>{});
  starpu_task* task = starpu_task_create();
  starpu_server::InferenceTask::allocate_task_buffers(task, 3, ctx);
  starpu_data_handle_t h1 = reinterpret_cast<starpu_data_handle_t>(0x1);
  starpu_data_handle_t h2 = reinterpret_cast<starpu_data_handle_t>(0x2);
  starpu_data_handle_t h3 = reinterpret_cast<starpu_data_handle_t>(0x3);
  starpu_server::InferenceTask::fill_task_buffers(task, {h1, h2}, {h3});
  EXPECT_EQ(task->dyn_handles[0], h1);
  EXPECT_EQ(task->dyn_handles[1], h2);
  EXPECT_EQ(task->dyn_handles[2], h3);
  EXPECT_EQ(task->dyn_modes[0], STARPU_R);
  EXPECT_EQ(task->dyn_modes[1], STARPU_R);
  EXPECT_EQ(task->dyn_modes[2], STARPU_W);
}
