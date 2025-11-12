#include <ATen/core/TensorBody.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>
#include <dlfcn.h>

#include <array>
#include <cstdlib>
#include <format>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>

#include "core/output_slot_pool.hpp"
#include "core/starpu_setup.hpp"
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
data_unregister_call_count_ref() -> int&
{
  static int value = 0;
  return value;
}

inline auto
data_unregister_handles_ref() -> std::vector<starpu_data_handle_t>&
{
  static std::vector<starpu_data_handle_t> handles;
  return handles;
}

inline auto
task_destroy_call_count_ref() -> int&
{
  static int value = 0;
  return value;
}

inline auto
last_destroyed_task_ref() -> starpu_task*&
{
  static starpu_task* ptr = nullptr;
  return ptr;
}

inline auto
task_submit_call_count_ref() -> int&
{
  static int value = 0;
  return value;
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

inline auto
MakeNullDataTensor() -> torch::Tensor
{
  auto data_ptr = c10::DataPtr(nullptr, c10::Device(c10::DeviceType::CPU));
  c10::Storage storage(
      c10::Storage::use_byte_size_t{},
      /*size_bytes=*/0, std::move(data_ptr),
      /*allocator=*/nullptr,
      /*resizable=*/false);
  return at::detail::make_tensor<c10::TensorImpl>(
      std::move(storage), c10::DispatchKeySet(c10::DispatchKey::CPU),
      c10::scalarTypeToTypeMeta(at::kFloat));
}

inline auto
MakeCudaTensor() -> torch::Tensor
{
  static float storage_value = 1.0F;
  auto device = c10::Device(c10::DeviceType::CUDA, 0);
  c10::DataPtr data_ptr(&storage_value, device);
  c10::Storage storage(
      c10::Storage::use_byte_size_t{}, sizeof(float), std::move(data_ptr),
      /*allocator=*/nullptr,
      /*resizable=*/false);
  auto tensor = at::detail::make_tensor<c10::TensorImpl>(
      std::move(storage), c10::DispatchKeySet(c10::DispatchKey::CUDA),
      c10::scalarTypeToTypeMeta(at::kFloat));
  tensor.unsafeGetTensorImpl()->set_storage_offset(0);
  tensor.unsafeGetTensorImpl()->set_sizes_contiguous({1});
  return tensor;
}

}  // namespace

extern "C" void
starpu_data_unregister_submit(starpu_data_handle_t handle)
{
  ++unregister_call_count_ref();
  unregister_handles_ref().push_back(handle);
}

namespace {
using DataUnregisterFn = void (*)(starpu_data_handle_t);
using VectorRegisterFn = decltype(&starpu_vector_data_register);

inline auto
resolve_real_starpu_data_unregister() -> DataUnregisterFn
{
  static DataUnregisterFn fn = []() -> DataUnregisterFn {
    void* symbol = dlsym(RTLD_NEXT, "starpu_data_unregister");
    if (symbol == nullptr) {
      throw std::runtime_error("Failed to resolve starpu_data_unregister");
    }
    return reinterpret_cast<DataUnregisterFn>(symbol);
  }();
  return fn;
}

inline auto
vector_register_override_ref() -> VectorRegisterFn&
{
  static VectorRegisterFn fn = nullptr;
  return fn;
}

inline auto
resolve_real_starpu_vector_data_register() -> VectorRegisterFn
{
  static VectorRegisterFn fn = []() -> VectorRegisterFn {
    void* symbol = dlsym(RTLD_NEXT, "starpu_vector_data_register");
    if (symbol == nullptr) {
      throw std::runtime_error("Failed to resolve starpu_vector_data_register");
    }
    return reinterpret_cast<VectorRegisterFn>(symbol);
  }();
  return fn;
}
}  // namespace

extern "C" void
starpu_data_unregister(starpu_data_handle_t handle)
{
  ++data_unregister_call_count_ref();
  data_unregister_handles_ref().push_back(handle);
  resolve_real_starpu_data_unregister()(handle);
}

extern "C" void
starpu_task_destroy(struct starpu_task* task)
{
  ++task_destroy_call_count_ref();
  last_destroyed_task_ref() = task;
}

extern "C" void
starpu_vector_data_register(
    starpu_data_handle_t* handle, int home_node, uintptr_t ptr, size_t nx,
    size_t elemsize)
{
  if (auto override = vector_register_override_ref()) {
    override(handle, home_node, ptr, nx, elemsize);
    return;
  }
  resolve_real_starpu_vector_data_register()(
      handle, home_node, ptr, nx, elemsize);
}

namespace {
using TaskSubmitOverrideFn = int (*)(starpu_task*);

inline auto
task_submit_override_ref() -> TaskSubmitOverrideFn&
{
  static TaskSubmitOverrideFn fn = nullptr;
  return fn;
}

inline auto
resolve_real_starpu_task_submit() -> TaskSubmitOverrideFn
{
  static TaskSubmitOverrideFn fn = []() -> TaskSubmitOverrideFn {
    void* symbol = dlsym(RTLD_NEXT, "starpu_task_submit");
    if (symbol == nullptr) {
      throw std::runtime_error("Failed to resolve starpu_task_submit");
    }
    return reinterpret_cast<TaskSubmitOverrideFn>(symbol);
  }();
  return fn;
}
}  // namespace

#ifdef starpu_task_submit
#define STARPU_TASK_SUBMIT_MACRO_WAS_DEFINED 1
#undef starpu_task_submit
#endif

extern "C" int
starpu_task_submit(starpu_task* task)
{
  if (auto override = task_submit_override_ref()) {
    return override(task);
  }
  return resolve_real_starpu_task_submit()(task);
}

#ifdef STARPU_TASK_SUBMIT_MACRO_WAS_DEFINED
#define starpu_task_submit(task) \
  starpu_task_submit_line((task), __FILE__, __LINE__)
#undef STARPU_TASK_SUBMIT_MACRO_WAS_DEFINED
#endif

namespace {
inline auto
AlwaysNullAllocator(size_t) -> void*
{
  return nullptr;
}

void
ThrowingStarpuOutputCallbackHook(starpu_server::InferenceCallbackContext*)
{
  throw starpu_server::StarPURegistrationException("forced failure");
}

auto
AlwaysFailingAcquire(
    starpu_data_handle_t, starpu_data_access_mode, void (*)(void*),
    void*) -> int
{
  return -42;
}

auto
AlwaysFailingTaskSubmit(starpu_task*) -> int
{
  ++task_submit_call_count_ref();
  return -99;
}

void
AlwaysFailingVectorRegister(
    starpu_data_handle_t* handle, int, uintptr_t, size_t, size_t)
{
  if (handle != nullptr) {
    *handle = nullptr;
  }
}

class ScopedTaskSubmitOverride {
 public:
  explicit ScopedTaskSubmitOverride(TaskSubmitOverrideFn fn)
  {
    task_submit_override_ref() = fn;
  }

  ScopedTaskSubmitOverride(const ScopedTaskSubmitOverride&) = delete;
  auto operator=(const ScopedTaskSubmitOverride&) -> ScopedTaskSubmitOverride& =
                                                         delete;
  ScopedTaskSubmitOverride(ScopedTaskSubmitOverride&&) = delete;
  auto operator=(ScopedTaskSubmitOverride&&) -> ScopedTaskSubmitOverride& =
                                                    delete;

  ~ScopedTaskSubmitOverride() { task_submit_override_ref() = nullptr; }
};

class ScopedVectorRegisterOverride {
 public:
  explicit ScopedVectorRegisterOverride(VectorRegisterFn fn)
  {
    vector_register_override_ref() = fn;
  }

  ScopedVectorRegisterOverride(const ScopedVectorRegisterOverride&) = delete;
  auto operator=(const ScopedVectorRegisterOverride&)
      -> ScopedVectorRegisterOverride& = delete;
  ScopedVectorRegisterOverride(ScopedVectorRegisterOverride&&) = delete;
  auto operator=(ScopedVectorRegisterOverride&&)
      -> ScopedVectorRegisterOverride& = delete;

  ~ScopedVectorRegisterOverride() { vector_register_override_ref() = nullptr; }
};
}  // namespace

TEST_F(InferenceTaskTest, TooManyInputsThrows)
{
  const size_t num_inputs = starpu_server::InferLimits::MaxInputs + 1;
  auto job = make_job(0, num_inputs);
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

TEST_F(InferenceTaskTest, FillModelPointersAllNegativeDeviceIds)
{
  auto job = make_job(1, 1);
  auto task = make_task(job, 1);
  opts_.devices.ids = {-1, -2};

  auto params = task.create_inference_params();

  EXPECT_TRUE(params->models.models_gpu.empty());
  EXPECT_EQ(params->models.num_models_gpu, models_gpu_.size());
}

TEST_F(InferenceTaskTest, FillModelPointersSkipsNegativeDeviceIds)
{
  auto job = make_job(2, 1);
  auto task = make_task(job, 2);
  opts_.devices.ids = {0, -1};

  auto params = task.create_inference_params();

  ASSERT_EQ(params->models.models_gpu.size(), 1U);
  EXPECT_EQ(params->models.models_gpu.at(0), &models_gpu_.at(0));
  EXPECT_EQ(params->models.num_models_gpu, models_gpu_.size());
}

TEST_F(InferenceTaskTest, AssignFixedWorkerValid)
{
  StarpuRuntimeGuard starpu_guard;
  const unsigned total_workers = starpu_worker_get_count();
  ASSERT_GT(total_workers, 0U) << "StarPU worker count must be positive";

  auto job = make_job(3, 1);

  const int worker_id =
      total_workers > 2U ? 2 : static_cast<int>(total_workers) - 1;
  job->set_fixed_worker_id(worker_id);
  auto task = make_task(job);
  starpu_task task_struct{};
  task.assign_fixed_worker_if_needed(&task_struct);
  EXPECT_EQ(task_struct.workerid, static_cast<unsigned>(worker_id));
  EXPECT_EQ(task_struct.execute_on_a_specific_worker, 1U);
}

TEST_F(InferenceTaskTest, CreateTaskThrowsWhenStarpuTaskCreateFails)
{
  starpu_server::InferenceTaskDependencies dependencies =
      starpu_server::kDefaultInferenceTaskDependencies;
  dependencies.task_create_fn = []() -> starpu_task* { return nullptr; };

  auto job = make_job(5, 0);
  auto task = make_task(job, 0, &dependencies);

  const std::vector<starpu_data_handle_t> inputs;
  const std::vector<starpu_data_handle_t> outputs;
  auto ctx = task.create_context(inputs, outputs);

  EXPECT_THROW(
      task.create_task(inputs, outputs, ctx),
      starpu_server::StarPUTaskCreationException);
}

TEST_F(
    InferenceTaskTest,
    CreateTaskAssignsDependenciesToContextWhenMissingDependenciesPointer)
{
  auto job = make_job(6, 0);
  starpu_server::InferenceTaskDependencies dependencies =
      starpu_server::kDefaultInferenceTaskDependencies;
  dependencies.task_create_fn = []() -> starpu_task* {
    return static_cast<starpu_task*>(std::calloc(1, sizeof(starpu_task)));
  };
  auto task = make_task(job, 0, &dependencies);

  const std::vector<starpu_data_handle_t> inputs;
  const std::vector<starpu_data_handle_t> outputs;
  auto params = task.create_inference_params();
  auto ctx =
      make_callback_context(job, &opts_, inputs, outputs, nullptr, params);
  ASSERT_EQ(ctx->dependencies, nullptr);

  auto* created_task = task.create_task(inputs, outputs, ctx);
  ASSERT_NE(created_task, nullptr);

  EXPECT_EQ(ctx->dependencies, &dependencies);

  ctx->self_keep_alive.reset();
  auto free_task = [](starpu_task* t) {
    if (t == nullptr) {
      return;
    }
    std::free(t->dyn_handles);
    std::free(t->dyn_modes);
    std::free(t);
  };
  free_task(created_task);
}

TEST(InferenceTask, RegisterInputsHandlesUnregistersHandlesOnFailure)
{
  StarpuRuntimeGuard starpu_guard;
  data_unregister_call_count_ref() = 0;
  data_unregister_handles_ref().clear();

  std::vector<torch::Tensor> tensors;
  tensors.push_back(torch::ones({1}, torch::TensorOptions().dtype(at::kFloat)));
  tensors.emplace_back();

  EXPECT_THROW(
      starpu_server::InferenceTask::register_inputs_handles(tensors),
      starpu_server::StarPURegistrationException);

  EXPECT_EQ(data_unregister_call_count_ref(), 1);
  ASSERT_EQ(data_unregister_handles_ref().size(), 1U);
  EXPECT_NE(data_unregister_handles_ref()[0], nullptr);

  data_unregister_call_count_ref() = 0;
  data_unregister_handles_ref().clear();
}

TEST(InferenceTask, SafeRegisterTensorVectorThrowsWhenDataPointerIsNull)
{
  StarpuRuntimeGuard starpu_guard;
  auto tensor = MakeNullDataTensor();
  ASSERT_TRUE(tensor.defined());
  ASSERT_EQ(tensor.data_ptr(), nullptr);

  EXPECT_THROW(
      starpu_server::InferenceTask::safe_register_tensor_vector(
          tensor, "null_data_tensor"),
      starpu_server::StarPURegistrationException);
}

TEST(InferenceTask, SafeRegisterTensorVectorThrowsWhenTensorNotOnCpu)
{
  StarpuRuntimeGuard starpu_guard;
  auto tensor = MakeCudaTensor();
  ASSERT_TRUE(tensor.defined());
  ASSERT_NE(tensor.data_ptr(), nullptr);
  ASSERT_FALSE(tensor.device().is_cpu());

  EXPECT_THROW(
      starpu_server::InferenceTask::safe_register_tensor_vector(
          tensor, "gpu_tensor"),
      starpu_server::StarPURegistrationException);
}

TEST(InferenceTask, SafeRegisterTensorVectorThrowsWhenStarpuRegistrationFails)
{
  StarpuRuntimeGuard starpu_guard;
  torch::Tensor tensor =
      torch::ones({1}, torch::TensorOptions().dtype(at::kFloat));
  ScopedVectorRegisterOverride override(&AlwaysFailingVectorRegister);

  EXPECT_THROW(
      starpu_server::InferenceTask::safe_register_tensor_vector(
          tensor, "failed_register_tensor"),
      starpu_server::StarPURegistrationException);
}

TEST_F(InferenceTaskTest, SubmitCleansUpAndThrowsOnTaskSubmissionFailure)
{
  unregister_call_count_ref() = 0;
  unregister_handles_ref().clear();
  task_destroy_call_count_ref() = 0;
  task_submit_call_count_ref() = 0;
  last_destroyed_task_ref() = nullptr;

  auto job = make_job(7, 1);
  starpu_server::RuntimeConfig opts;
  model_cpu_ = starpu_server::make_add_one_model();
  models_gpu_.clear();
  auto starpu_setup = std::make_unique<starpu_server::StarPUSetup>(opts);
  auto dependencies = starpu_server::kDefaultInferenceTaskDependencies;
  starpu_server::InferenceTask task(
      starpu_setup.get(), job, &model_cpu_, &models_gpu_, &opts, dependencies);

  ScopedTaskSubmitOverride submit_override(&AlwaysFailingTaskSubmit);

  EXPECT_THROW(task.submit(), starpu_server::StarPUTaskSubmissionException);

  EXPECT_EQ(task_submit_call_count_ref(), 1);
  EXPECT_EQ(task_destroy_call_count_ref(), 1);
  EXPECT_NE(last_destroyed_task_ref(), nullptr);
  EXPECT_EQ(unregister_call_count_ref(), 2);
  EXPECT_EQ(unregister_handles_ref().size(), 2U);

  unregister_handles_ref().clear();
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
  EXPECT_EQ(params->request_id, 4);
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
  auto ctx = make_callback_context(job, &opts);
  starpu_server::InferenceTask::record_and_run_completion_callback(
      ctx.get(), end);
  EXPECT_TRUE(called);
  ASSERT_EQ(results_arg.size(), outputs.size());
  EXPECT_TRUE(torch::equal(results_arg[0], outputs[0]));
  const double expected_latency =
      std::chrono::duration<double, std::milli>(end - start).count();
  EXPECT_DOUBLE_EQ(latency_ms, expected_latency);
}

TEST_F(InferenceTaskTest, StarpuOutputCallbackLogsInferenceEngineException)
{
  auto job = make_job(7, 1);
  starpu_server::InferenceTaskDependencies dependencies =
      starpu_server::kDefaultInferenceTaskDependencies;
  dependencies.starpu_output_callback_hook =
      starpu_server::InferenceTaskDependencies::OutputCallbackHook(
          &ThrowingStarpuOutputCallbackHook);
  starpu_server::RuntimeConfig opts;
  auto ctx =
      make_callback_context(job, &opts, {}, {MakeHandle(0)}, &dependencies);

  starpu_server::CaptureStream capture{std::cerr};

  EXPECT_NO_THROW(
      starpu_server::InferenceTask::starpu_output_callback(ctx.get()));

  const auto log = capture.str();
  EXPECT_NE(log.find("starpu_output_callback"), std::string::npos);
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
  auto ctx = make_callback_context(job, &opts);
  starpu_server::CaptureStream capture{std::cerr};

  EXPECT_NO_THROW(
      starpu_server::InferenceTask::record_and_run_completion_callback(
          ctx.get(), end));
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
  auto ctx = make_callback_context(job, &opts);
  starpu_server::CaptureStream capture{std::cerr};

  EXPECT_NO_THROW(
      starpu_server::InferenceTask::record_and_run_completion_callback(
          ctx.get(), end));
  const auto log = capture.str();
  EXPECT_NE(
      log.find("Unknown exception in completion callback"), std::string::npos);
}

TEST_F(InferenceTaskTest, AcquireOutputHandleLogsAndThrowsOnFailure)
{
  auto job = make_job(3, 0);
  auto handle = MakeHandle(0);
  starpu_server::InferenceTaskDependencies dependencies =
      starpu_server::kDefaultInferenceTaskDependencies;
  dependencies.starpu_data_acquire_fn = &AlwaysFailingAcquire;
  auto outputs = std::vector<starpu_data_handle_t>{handle};
  auto ctx = make_callback_context(job, &opts_, {}, outputs, &dependencies);
  ctx->remaining_outputs_to_acquire = static_cast<int>(outputs.size());
  starpu_server::CaptureStream capture{std::cerr};

  EXPECT_THROW(
      starpu_server::InferenceTask::acquire_output_handle(handle, ctx.get()),
      starpu_server::StarPURegistrationException);

  const auto log = capture.str();
  const auto expected = starpu_server::expected_log_line(
      starpu_server::ErrorLevel,
      std::format("starpu_data_acquire_cb failed with code {}", -42));
  EXPECT_NE(log.find(expected), std::string::npos);
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
  auto ctx = make_callback_context(nullptr, nullptr, inputs, outputs);
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
  constexpr float kSentinelValue = 42.0F;
  OutputContextFixture fixture({.sentinel_value = kSentinelValue});

  ASSERT_NO_THROW(
      starpu_server::InferenceTask::finalize_inference_task(fixture.ctx.get()));

  const auto& job_outputs = fixture.job->get_output_tensors();
  ASSERT_EQ(job_outputs.size(), 1U);
  ASSERT_TRUE(job_outputs[0].defined());
  EXPECT_FLOAT_EQ(job_outputs[0].item<float>(), kSentinelValue);

  auto reacquired = fixture.pool.try_acquire();
  ASSERT_TRUE(reacquired.has_value());
  EXPECT_EQ(*reacquired, fixture.slot_id);
  fixture.pool.release(*reacquired);
}

TEST(InferenceTask, FinalizeInferenceTaskHandlesCopyFailure)
{
  OutputContextFixture fixture({.mutate_job_outputs = [](auto& outputs) {
    outputs[0] = torch::Tensor();
  }});

  starpu_server::CaptureStream capture{std::cerr};

  EXPECT_NO_THROW(
      starpu_server::InferenceTask::finalize_inference_task(fixture.ctx.get()));
  const auto log = capture.str();
  EXPECT_NE(log.find("Output copy from pool failed"), std::string::npos);

  auto reacquired = fixture.pool.try_acquire();
  ASSERT_TRUE(reacquired.has_value());
  EXPECT_EQ(*reacquired, fixture.slot_id);
  fixture.pool.release(*reacquired);
}

TEST(InferenceTask, FinalizeInferenceTaskHandlesOnFinishedException)
{
  constexpr float kSentinelValue = 7.0F;
  OutputContextFixture fixture({
      .sentinel_value = kSentinelValue,
      .on_finished =
          [](auto& pool, int slot_id) {
            pool.release(slot_id);
            throw std::runtime_error("boom");
          },
  });

  starpu_server::CaptureStream capture{std::cerr};

  EXPECT_NO_THROW(
      starpu_server::InferenceTask::finalize_inference_task(fixture.ctx.get()));
  const auto log = capture.str();
  EXPECT_NE(log.find("Exception in on_finished"), std::string::npos);

  const auto& job_outputs = fixture.job->get_output_tensors();
  ASSERT_EQ(job_outputs.size(), 1U);
  ASSERT_TRUE(job_outputs[0].defined());
  EXPECT_FLOAT_EQ(job_outputs[0].item<float>(), kSentinelValue);

  auto reacquired = fixture.pool.try_acquire();
  ASSERT_TRUE(reacquired.has_value());
  EXPECT_EQ(*reacquired, fixture.slot_id);
  fixture.pool.release(*reacquired);
}

TEST(InferenceTask, ProcessOutputHandleNullHandleFinalizes)
{
  auto ctx = make_callback_context();
  ctx->remaining_outputs_to_acquire = 1;
  ctx->self_keep_alive = ctx;

  bool finished = false;
  ctx->on_finished = [&]() { finished = true; };

  starpu_server::InferenceTask::process_output_handle(nullptr, ctx.get());

  EXPECT_TRUE(finished);
  EXPECT_EQ(ctx->remaining_outputs_to_acquire.load(), 0);
  EXPECT_EQ(ctx->self_keep_alive, nullptr);
}

TEST(InferenceTaskBuffers, FillTaskBuffersOrdersDynHandlesAndModes)
{
  auto ctx = make_callback_context();
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

TEST(InferenceTaskBuffers, AllocateTaskBuffersThrowsWhenHandleAllocationFails)
{
  starpu_server::InferenceTaskDependencies dependencies =
      starpu_server::kDefaultInferenceTaskDependencies;
  dependencies.dyn_handles_allocator = &AlwaysNullAllocator;
  auto ctx = make_callback_context(nullptr, nullptr, {}, {}, &dependencies);
  starpu_task task{};
  EXPECT_THROW(
      starpu_server::InferenceTask::allocate_task_buffers(&task, 2, ctx),
      starpu_server::MemoryAllocationException);
  EXPECT_EQ(task.dyn_handles, nullptr);
  EXPECT_EQ(task.dyn_modes, nullptr);
}

TEST(InferenceTaskBuffers, AllocateTaskBuffersThrowsWhenModeAllocationFails)
{
  starpu_server::InferenceTaskDependencies dependencies =
      starpu_server::kDefaultInferenceTaskDependencies;
  dependencies.dyn_modes_allocator = &AlwaysNullAllocator;
  auto ctx = make_callback_context(nullptr, nullptr, {}, {}, &dependencies);
  starpu_task task{};
  EXPECT_THROW(
      starpu_server::InferenceTask::allocate_task_buffers(&task, 2, ctx),
      starpu_server::MemoryAllocationException);
  EXPECT_EQ(task.dyn_handles, nullptr);
  EXPECT_EQ(task.dyn_modes, nullptr);
}
