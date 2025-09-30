#include <array>
#include <stdexcept>

#include "core/output_slot_pool.hpp"
#include "test_inference_task.hpp"
#include "test_utils.hpp"

namespace {
inline auto
unregister_call_count_ref() -> int&
{
  static int value = 0;
  return value;
}

inline auto
unregister_handles_ref() -> std::vector<starpu_data_handle_t>&
{
  static std::vector<starpu_data_handle_t> handles;
  return handles;
}

inline auto
MakeHandle(int index) -> starpu_data_handle_t
{
  constexpr std::size_t kDummyStorageSize = 8;
  static std::array<int, kDummyStorageSize> dummy_storage{};
  void* ptr =
      &dummy_storage.at(static_cast<std::size_t>(index) % kDummyStorageSize);
  return static_cast<starpu_data_handle_t>(ptr);
}

auto
MakeRuntimeConfigWithSingleOutput() -> starpu_server::RuntimeConfig
{
  starpu_server::RuntimeConfig opts;
  starpu_server::ModelConfig model;
  model.name = "test_model";
  starpu_server::TensorConfig output;
  output.name = "output";
  output.dims = {1};
  output.type = at::kFloat;
  model.outputs.push_back(output);
  opts.models.push_back(std::move(model));
  return opts;
}
}  // namespace

extern "C" void
starpu_data_unregister_submit(starpu_data_handle_t handle)
{
  ++unregister_call_count_ref();
  unregister_handles_ref().push_back(handle);
}

extern "C" void
starpu_task_destroy(struct starpu_task* /*task*/)
{
}

namespace {
inline void*
AlwaysNullAllocator(size_t)
{
  return nullptr;
}

class InferenceTaskHandlesAllocationFailureTest : public InferenceTaskTest {
 protected:
  void SetUp() override
  {
    previous_allocator_ =
        starpu_server::InferenceTask::set_dyn_handles_allocator_for_testing(
            &AlwaysNullAllocator);
  }

  void TearDown() override
  {
    starpu_server::InferenceTask::set_dyn_handles_allocator_for_testing(
        previous_allocator_);
  }

 private:
  starpu_server::InferenceTask::AllocationFn previous_allocator_ = nullptr;
};

class InferenceTaskModesAllocationFailureTest : public InferenceTaskTest {
 protected:
  void SetUp() override
  {
    previous_allocator_ =
        starpu_server::InferenceTask::set_dyn_modes_allocator_for_testing(
            &AlwaysNullAllocator);
  }

  void TearDown() override
  {
    starpu_server::InferenceTask::set_dyn_modes_allocator_for_testing(
        previous_allocator_);
  }

 private:
  starpu_server::InferenceTask::AllocationFn previous_allocator_ = nullptr;
};
}  // namespace

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

TEST_F(InferenceTaskTest, AssignFixedWorkerValid)
{
  auto job = make_job(3, 1);
  const unsigned total_workers = starpu_worker_get_count();
  if (total_workers == 0) {
    GTEST_SKIP() << "No StarPU workers available";
  }

  const int worker_id =
      total_workers > 2U ? 2 : static_cast<int>(total_workers) - 1;
  job->set_fixed_worker_id(worker_id);
  auto task = make_task(job);
  starpu_task task_struct{};
  task.assign_fixed_worker_if_needed(&task_struct);
  EXPECT_EQ(task_struct.workerid, static_cast<unsigned>(worker_id));
  EXPECT_EQ(task_struct.execute_on_a_specific_worker, 1U);
}

TEST_F(InferenceTaskTest, CreateInferenceParamsPopulatesFields)
{
  auto job = make_job(4, 1);
  job->set_input_tensors({torch::ones({2, 3})});
  job->set_output_tensors({torch::zeros({2, 3})});
  auto task = make_task(job);
  opts_.verbosity = starpu_server::VerbosityLevel::Debug;
  auto params = task.create_inference_params();
  ASSERT_EQ(params->num_inputs, 1U);
  ASSERT_EQ(params->num_outputs, 1U);
  EXPECT_EQ(params->job_id, 4);
  EXPECT_EQ(params->verbosity, opts_.verbosity);
  EXPECT_EQ(params->models.model_cpu, &model_cpu_);
  EXPECT_EQ(params->models.num_models_gpu, 0U);
  EXPECT_EQ(params->device.device_id, &job->get_device_id());
  EXPECT_EQ(params->device.worker_id, &job->get_worker_id());
  EXPECT_EQ(params->device.executed_on, &job->get_executed_on());
  EXPECT_EQ(params->layout.num_dims[0], 2);
  EXPECT_EQ(params->layout.dims[0][0], 2);
  EXPECT_EQ(params->layout.dims[0][1], 3);
  EXPECT_EQ(params->layout.input_types[0], at::kFloat);
}

TEST(InferenceTask, RecordAndRunCompletionCallback)
{
  auto job = std::make_shared<starpu_server::InferenceJob>();
  std::vector<torch::Tensor> outputs{torch::tensor({1})};
  job->set_output_tensors(outputs);
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

TEST(InferenceTask, RecordAndRunCompletionCallbackLogsStdException)
{
  auto job = std::make_shared<starpu_server::InferenceJob>();
  job->set_output_tensors({torch::tensor({1})});
  job->set_on_complete([](const std::vector<torch::Tensor>&, double) {
    throw std::runtime_error("callback failure");
  });
  const auto start = std::chrono::high_resolution_clock::now();
  const auto end = start + std::chrono::milliseconds(5);
  job->set_start_time(start);
  starpu_server::RuntimeConfig opts;
  starpu_server::InferenceCallbackContext ctx(job, nullptr, &opts, 0, {}, {});
  starpu_server::CaptureStream capture{std::cerr};

  EXPECT_NO_THROW(
      starpu_server::InferenceTask::record_and_run_completion_callback(
          &ctx, end));
  const auto log = capture.str();
  EXPECT_NE(
      log.find("Exception in completion callback: callback failure"),
      std::string::npos);
}

TEST(InferenceTask, RecordAndRunCompletionCallbackLogsUnknownException)
{
  auto job = std::make_shared<starpu_server::InferenceJob>();
  job->set_output_tensors({torch::tensor({1})});
  job->set_on_complete(
      [](const std::vector<torch::Tensor>&, double) { throw 42; });
  const auto start = std::chrono::high_resolution_clock::now();
  const auto end = start + std::chrono::milliseconds(5);
  job->set_start_time(start);
  starpu_server::RuntimeConfig opts;
  starpu_server::InferenceCallbackContext ctx(job, nullptr, &opts, 0, {}, {});
  starpu_server::CaptureStream capture{std::cerr};

  EXPECT_NO_THROW(
      starpu_server::InferenceTask::record_and_run_completion_callback(
          &ctx, end));
  const auto log = capture.str();
  EXPECT_NE(
      log.find("Unknown exception in completion callback"), std::string::npos);
}

TEST(InferenceTask, CleanupUnregistersAndNullsHandles)
{
  unregister_call_count_ref() = 0;
  unregister_handles_ref().clear();
  auto* const handle_1 = MakeHandle(1);
  auto* const handle_2 = MakeHandle(2);
  auto* const handle_3 = MakeHandle(3);
  std::vector<starpu_data_handle_t> inputs{handle_1};
  std::vector<starpu_data_handle_t> outputs{handle_2, handle_3};
  auto ctx = std::make_shared<starpu_server::InferenceCallbackContext>(
      nullptr, nullptr, nullptr, 0, inputs, outputs);
  starpu_server::InferenceTask::cleanup(ctx);
  EXPECT_EQ(unregister_call_count_ref(), 3);
  ASSERT_EQ(unregister_handles_ref().size(), 3U);
  EXPECT_EQ(unregister_handles_ref()[0], handle_1);
  EXPECT_EQ(unregister_handles_ref()[1], handle_2);
  EXPECT_EQ(unregister_handles_ref()[2], handle_3);
  EXPECT_EQ(ctx->inputs_handles[0], nullptr);
  EXPECT_EQ(ctx->outputs_handles[0], nullptr);
  EXPECT_EQ(ctx->outputs_handles[1], nullptr);
}

TEST(InferenceTask, FinalizeInferenceTaskCopiesOutputs)
{
  StarpuRuntimeGuard starpu_guard;
  auto opts = MakeRuntimeConfigWithSingleOutput();
  starpu_server::OutputSlotPool pool(opts, 1);
  const int slot_id = pool.acquire();

  auto job = std::make_shared<starpu_server::InferenceJob>();
  job->set_output_tensors(
      {torch::zeros({1}, torch::TensorOptions().dtype(at::kFloat))});

  auto ctx = std::make_shared<starpu_server::InferenceCallbackContext>(
      job, nullptr, &opts, 0, std::vector<starpu_data_handle_t>{},
      std::vector<starpu_data_handle_t>{});
  ctx->output_pool = &pool;
  ctx->output_slot_id = slot_id;
  ctx->self_keep_alive = ctx;
  ctx->on_finished = [&pool, slot_id]() { pool.release(slot_id); };

  constexpr float kSentinelValue = 42.0F;
  const auto& base_ptrs = pool.base_ptrs(slot_id);
  ASSERT_EQ(base_ptrs.size(), 1U);
  *static_cast<float*>(base_ptrs[0]) = kSentinelValue;

  ASSERT_NO_THROW(
      starpu_server::InferenceTask::finalize_inference_task(ctx.get()));

  const auto& job_outputs = job->get_output_tensors();
  ASSERT_EQ(job_outputs.size(), 1U);
  ASSERT_TRUE(job_outputs[0].defined());
  EXPECT_FLOAT_EQ(job_outputs[0].item<float>(), kSentinelValue);

  auto reacquired = pool.try_acquire();
  ASSERT_TRUE(reacquired.has_value());
  EXPECT_EQ(*reacquired, slot_id);
  pool.release(*reacquired);
}

TEST(InferenceTask, FinalizeInferenceTaskHandlesCopyFailure)
{
  StarpuRuntimeGuard starpu_guard;
  auto opts = MakeRuntimeConfigWithSingleOutput();
  starpu_server::OutputSlotPool pool(opts, 1);
  const int slot_id = pool.acquire();

  auto job = std::make_shared<starpu_server::InferenceJob>();
  job->set_output_tensors(
      {torch::zeros({1}, torch::TensorOptions().dtype(at::kFloat))});

  auto ctx = std::make_shared<starpu_server::InferenceCallbackContext>(
      job, nullptr, &opts, 0, std::vector<starpu_data_handle_t>{},
      std::vector<starpu_data_handle_t>{});
  ctx->output_pool = &pool;
  ctx->output_slot_id = slot_id;
  ctx->self_keep_alive = ctx;
  ctx->on_finished = [&pool, slot_id]() { pool.release(slot_id); };

  auto& outputs =
      const_cast<std::vector<torch::Tensor>&>(job->get_output_tensors());
  outputs[0] = torch::Tensor();

  starpu_server::CaptureStream capture{std::cerr};

  EXPECT_NO_THROW(
      starpu_server::InferenceTask::finalize_inference_task(ctx.get()));
  const auto log = capture.str();
  EXPECT_NE(log.find("Output copy from pool failed"), std::string::npos);

  auto reacquired = pool.try_acquire();
  ASSERT_TRUE(reacquired.has_value());
  EXPECT_EQ(*reacquired, slot_id);
  pool.release(*reacquired);
}

TEST(InferenceTask, FinalizeInferenceTaskHandlesOnFinishedException)
{
  StarpuRuntimeGuard starpu_guard;
  auto opts = MakeRuntimeConfigWithSingleOutput();
  starpu_server::OutputSlotPool pool(opts, 1);
  const int slot_id = pool.acquire();

  auto job = std::make_shared<starpu_server::InferenceJob>();
  job->set_output_tensors(
      {torch::zeros({1}, torch::TensorOptions().dtype(at::kFloat))});

  auto ctx = std::make_shared<starpu_server::InferenceCallbackContext>(
      job, nullptr, &opts, 0, std::vector<starpu_data_handle_t>{},
      std::vector<starpu_data_handle_t>{});
  ctx->output_pool = &pool;
  ctx->output_slot_id = slot_id;
  ctx->self_keep_alive = ctx;
  ctx->on_finished = [&pool, slot_id]() {
    pool.release(slot_id);
    throw std::runtime_error("boom");
  };

  constexpr float kSentinelValue = 7.0F;
  const auto& base_ptrs = pool.base_ptrs(slot_id);
  ASSERT_EQ(base_ptrs.size(), 1U);
  *static_cast<float*>(base_ptrs[0]) = kSentinelValue;

  starpu_server::CaptureStream capture{std::cerr};

  EXPECT_NO_THROW(
      starpu_server::InferenceTask::finalize_inference_task(ctx.get()));
  const auto log = capture.str();
  EXPECT_NE(log.find("Exception in on_finished"), std::string::npos);

  const auto& job_outputs = job->get_output_tensors();
  ASSERT_EQ(job_outputs.size(), 1U);
  ASSERT_TRUE(job_outputs[0].defined());
  EXPECT_FLOAT_EQ(job_outputs[0].item<float>(), kSentinelValue);

  auto reacquired = pool.try_acquire();
  ASSERT_TRUE(reacquired.has_value());
  EXPECT_EQ(*reacquired, slot_id);
  pool.release(*reacquired);
}

TEST(InferenceTaskBuffers, FillTaskBuffersOrdersDynHandlesAndModes)
{
  auto ctx = std::make_shared<starpu_server::InferenceCallbackContext>(
      nullptr, nullptr, nullptr, 0, std::vector<starpu_data_handle_t>{},
      std::vector<starpu_data_handle_t>{});
  starpu_task* task = starpu_task_create();
  starpu_server::InferenceTask::allocate_task_buffers(task, 3, ctx);
  auto* handle_1 = MakeHandle(1);
  auto* handle_2 = MakeHandle(2);
  auto* handle_3 = MakeHandle(3);
  starpu_server::InferenceTask::fill_task_buffers(
      task, {handle_1, handle_2}, {handle_3});
  std::span<starpu_data_handle_t> handles(task->dyn_handles, 3);
  std::span<starpu_data_access_mode> modes(task->dyn_modes, 3);
  EXPECT_EQ(handles[0], handle_1);
  EXPECT_EQ(handles[1], handle_2);
  EXPECT_EQ(handles[2], handle_3);
  EXPECT_EQ(modes[0], STARPU_R);
  EXPECT_EQ(modes[1], STARPU_R);
  EXPECT_EQ(modes[2], STARPU_W);
}

TEST_F(
    InferenceTaskHandlesAllocationFailureTest,
    AllocateTaskBuffersThrowsWhenHandleAllocationFails)
{
  auto ctx = std::make_shared<starpu_server::InferenceCallbackContext>(
      nullptr, nullptr, nullptr, 0, std::vector<starpu_data_handle_t>{},
      std::vector<starpu_data_handle_t>{});
  starpu_task task{};
  EXPECT_THROW(
      starpu_server::InferenceTask::allocate_task_buffers(&task, 2, ctx),
      starpu_server::MemoryAllocationException);
  EXPECT_EQ(task.dyn_handles, nullptr);
  EXPECT_EQ(task.dyn_modes, nullptr);
}

TEST_F(
    InferenceTaskModesAllocationFailureTest,
    AllocateTaskBuffersThrowsWhenModeAllocationFails)
{
  auto ctx = std::make_shared<starpu_server::InferenceCallbackContext>(
      nullptr, nullptr, nullptr, 0, std::vector<starpu_data_handle_t>{},
      std::vector<starpu_data_handle_t>{});
  starpu_task task{};
  EXPECT_THROW(
      starpu_server::InferenceTask::allocate_task_buffers(&task, 2, ctx),
      starpu_server::MemoryAllocationException);
  EXPECT_EQ(task.dyn_handles, nullptr);
  EXPECT_EQ(task.dyn_modes, nullptr);
}
