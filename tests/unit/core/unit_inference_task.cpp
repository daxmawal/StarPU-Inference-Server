#include <ATen/core/TensorBody.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorImpl.h>
#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <format>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include "core/output_slot_pool.hpp"
#include "core/starpu_setup.hpp"
#include "support/inference_task_test_hooks.hpp"
#include "support/starpu_task_submit_override.hpp"
#include "test_inference_task.hpp"
#include "test_utils.hpp"

namespace {

starpu_server::testing::ScopedStarpuSilent g_starpu_silent{};

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
data_release_throw_count_ref() -> int&
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

auto
gpu_replica_worker_query_stub(
    unsigned int device_id, int* worker_ids,
    enum starpu_worker_archtype worker_type) -> int
{
  if (worker_type != STARPU_CUDA_WORKER || worker_ids == nullptr) {
    return 0;
  }
  if (device_id == 0U) {
    worker_ids[0] = 7;
    worker_ids[1] = 9;
    return 2;
  }
  return 0;
}

class WorkerStreamQueryGuard {
 public:
  WorkerStreamQueryGuard()
  {
    starpu_server::StarPUSetup::set_worker_stream_query_fn(
        &gpu_replica_worker_query_stub);
  }

  ~WorkerStreamQueryGuard()
  {
    starpu_server::StarPUSetup::reset_worker_stream_query_fn();
  }

  WorkerStreamQueryGuard(const WorkerStreamQueryGuard&) = delete;
  auto operator=(const WorkerStreamQueryGuard&) -> WorkerStreamQueryGuard& =
                                                       delete;
};

template <typename Func>
auto
CaptureStderr(Func&& func) -> std::string
{
  starpu_server::CaptureStream capture{std::cerr};
  std::forward<Func>(func)();
  return capture.str();
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

inline auto
resolve_real_starpu_data_unregister() -> DataUnregisterFn
{
  static DataUnregisterFn fn = []() {
    void* symbol = dlsym(RTLD_NEXT, "starpu_data_unregister");
    if (symbol == nullptr) {
      throw std::runtime_error("Failed to resolve starpu_data_unregister");
    }
    return reinterpret_cast<DataUnregisterFn>(symbol);
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

namespace {

auto
AlwaysNullAllocator(size_t) -> void*
{
  return nullptr;
}

int g_owner_handles_alloc_calls = 0;
int g_owner_modes_alloc_calls = 0;

auto
OwnerHandlesAllocator(size_t size) -> void*
{
  ++g_owner_handles_alloc_calls;
  return std::malloc(size);
}

auto
OwnerModesAllocator(size_t size) -> void*
{
  ++g_owner_modes_alloc_calls;
  return std::malloc(size);
}

void
OwnerAllocatorFree(void* ptr)
{
  std::free(ptr);
}

void
ThrowingStarpuOutputCallbackHook(starpu_server::InferenceCallbackContext*)
{
  throw starpu_server::StarPURegistrationException("forced failure");
}

inline auto
throwing_once_output_callback_calls_ref() -> int&
{
  static int value = 0;
  return value;
}

void
ThrowingStdStarpuOutputCallbackHook(starpu_server::InferenceCallbackContext*)
{
  throw std::runtime_error("forced std callback failure");
}

void
ThrowingUnknownStarpuOutputCallbackHook(
    starpu_server::InferenceCallbackContext*)
{
  throw 42;
}

void
ThrowingOnceStdStarpuOutputCallbackHook(
    starpu_server::InferenceCallbackContext*)
{
  ++throwing_once_output_callback_calls_ref();
  throw std::runtime_error("forced bypass callback failure");
}

auto
AlwaysFailingAcquire(
    starpu_data_handle_t, starpu_data_access_mode, void (*)(void*),
    void*) -> int
{
  return -42;
}

auto
ImmediateCallbackAcquire(
    starpu_data_handle_t, starpu_data_access_mode, void (*callback)(void*),
    void* ctx) -> int
{
  if (callback != nullptr) {
    callback(ctx);
  }
  return 0;
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

void
ThrowingDataRelease(starpu_data_handle_t)
{
  ++data_release_throw_count_ref();
  throw starpu_server::InvalidInferenceJobException(
      "forced data release failure");
}

void
ThrowingStdExceptionDataRelease(starpu_data_handle_t)
{
  throw std::runtime_error("forced std data release failure");
}

void
ThrowingUnknownDataRelease(starpu_data_handle_t)
{
  throw 42;
}

void
NoopDataRelease(starpu_data_handle_t)
{
}

class ScopedDefaultDataAcquireNullifier {
 public:
  ScopedDefaultDataAcquireNullifier()
      : deps_(&starpu_server::kDefaultInferenceTaskDependencies),
        original_(deps_->starpu_data_acquire_fn)
  {
    deps_->starpu_data_acquire_fn = nullptr;
  }

  ScopedDefaultDataAcquireNullifier(const ScopedDefaultDataAcquireNullifier&) =
      delete;
  auto operator=(const ScopedDefaultDataAcquireNullifier&)
      -> ScopedDefaultDataAcquireNullifier& = delete;
  ScopedDefaultDataAcquireNullifier(ScopedDefaultDataAcquireNullifier&&) =
      delete;
  auto operator=(ScopedDefaultDataAcquireNullifier&&)
      -> ScopedDefaultDataAcquireNullifier& = delete;

  ~ScopedDefaultDataAcquireNullifier()
  {
    deps_->starpu_data_acquire_fn = original_;
  }

  [[nodiscard]] auto is_valid() const -> bool { return original_ != nullptr; }

 private:
  starpu_server::InferenceTaskDependencies* deps_;
  starpu_server::InferenceTaskDependencies::DataAcquireFn original_ = nullptr;
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
}

TEST_F(InferenceTaskTest, FillModelPointersSkipsNegativeDeviceIds)
{
  auto job = make_job(2, 1);
  auto task = make_task(job, 2);
  opts_.devices.ids = {0, -1};

  auto params = task.create_inference_params();

  ASSERT_EQ(params->models.models_gpu.size(), 1U);
  EXPECT_EQ(params->models.models_gpu.at(0), &models_gpu_.at(0));
  EXPECT_TRUE(params->models.worker_ids.empty());
}

TEST_F(InferenceTaskTest, FillModelPointersPerWorkerReplicationUsesWorkers)
{
  WorkerStreamQueryGuard query_guard;

  auto job = make_job(4, 1);
  auto task = make_task(job, 2);
  opts_.devices.use_cuda = true;
  opts_.devices.ids = {0};
  opts_.devices.gpu_model_replication =
      starpu_server::GpuModelReplicationPolicy::PerWorker;

  auto params = task.create_inference_params();

  ASSERT_EQ(params->models.models_gpu.size(), 2U);
  EXPECT_EQ(params->models.device_ids, (std::vector<int>{0, 0}));
  EXPECT_EQ(params->models.worker_ids, (std::vector<int>{7, 9}));
  EXPECT_EQ(params->models.models_gpu.at(0), &models_gpu_.at(0));
  EXPECT_EQ(params->models.models_gpu.at(1), &models_gpu_.at(1));
}

TEST_F(InferenceTaskTest, FillModelPointersUsesProvidedReplicaAssignments)
{
  const std::vector<starpu_server::detail::GpuReplicaAssignment> assignments = {
      {.device_id = 0, .worker_id = 11}, {.device_id = 0, .worker_id = 13}};

  auto job = make_job(5, 1);
  auto task = make_task(job, 2, nullptr, &assignments);
  opts_.devices.use_cuda = true;
  opts_.devices.ids = {0};
  opts_.devices.gpu_model_replication =
      starpu_server::GpuModelReplicationPolicy::PerWorker;

  auto params = task.create_inference_params();

  ASSERT_EQ(params->models.models_gpu.size(), 2U);
  EXPECT_EQ(params->models.device_ids, (std::vector<int>{0, 0}));
  EXPECT_EQ(params->models.worker_ids, (std::vector<int>{11, 13}));
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

TEST_F(InferenceTaskTest, CreateTaskCleansUpWhenAssignFixedWorkerThrows)
{
  unregister_call_count_ref() = 0;
  unregister_handles_ref().clear();
  task_destroy_call_count_ref() = 0;
  last_destroyed_task_ref() = nullptr;

  starpu_server::InferenceTaskDependencies dependencies =
      starpu_server::kDefaultInferenceTaskDependencies;
  dependencies.task_create_fn = []() {
    return static_cast<starpu_task*>(std::calloc(1, sizeof(starpu_task)));
  };

  auto job = make_job(6, 2);
  job->set_fixed_worker_id(-1);
  auto task = make_task(job, 0, &dependencies);

  const std::vector<starpu_data_handle_t> inputs{MakeHandle(1), MakeHandle(2)};
  const std::vector<starpu_data_handle_t> outputs{MakeHandle(3)};
  auto ctx = task.create_context(inputs, outputs);

  EXPECT_THROW(task.create_task(inputs, outputs, ctx), std::invalid_argument);

  EXPECT_EQ(task_destroy_call_count_ref(), 1);
  ASSERT_NE(last_destroyed_task_ref(), nullptr);
  EXPECT_EQ(unregister_call_count_ref(), 3);
  ASSERT_EQ(unregister_handles_ref().size(), 3U);
  EXPECT_EQ(unregister_handles_ref()[0], inputs[0]);
  EXPECT_EQ(unregister_handles_ref()[1], inputs[1]);
  EXPECT_EQ(unregister_handles_ref()[2], outputs[0]);
  EXPECT_EQ(ctx->inputs_handles[0], nullptr);
  EXPECT_EQ(ctx->inputs_handles[1], nullptr);
  EXPECT_EQ(ctx->outputs_handles[0], nullptr);

  std::free(last_destroyed_task_ref());
}

TEST_F(
    InferenceTaskTest,
    CreateTaskAssignsDependenciesToContextWhenMissingDependenciesPointer)
{
  auto job = make_job(6, 0);
  starpu_server::InferenceTaskDependencies dependencies =
      starpu_server::kDefaultInferenceTaskDependencies;
  dependencies.task_create_fn = []() {
    return static_cast<starpu_task*>(std::calloc(1, sizeof(starpu_task)));
  };
  auto task = make_task(job, 0, &dependencies);

  const std::vector<starpu_data_handle_t> inputs;
  const std::vector<starpu_data_handle_t> outputs;
  auto params = task.create_inference_params();
  auto ctx = make_callback_context(job, inputs, outputs, nullptr, params);
  ASSERT_EQ(ctx->dependencies, nullptr);

  auto* created_task = task.create_task(inputs, outputs, ctx);
  ASSERT_NE(created_task, nullptr);

  EXPECT_NE(ctx->dependencies, nullptr);
  EXPECT_TRUE(ctx->dependencies_owner);
  EXPECT_EQ(ctx->dependencies->task_create_fn, dependencies.task_create_fn);

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

TEST_F(InferenceTaskTest, CreateTaskUsesDependenciesOwnerWhenProvided)
{
  auto job = make_job(7, 0);
  starpu_server::InferenceTaskDependencies dependencies =
      starpu_server::kDefaultInferenceTaskDependencies;
  dependencies.task_create_fn = []() {
    return static_cast<starpu_task*>(std::calloc(1, sizeof(starpu_task)));
  };
  auto task = make_task(job, 0, &dependencies);

  auto owner = std::make_shared<starpu_server::InferenceTaskDependencies>(
      starpu_server::kDefaultInferenceTaskDependencies);

  const std::vector<starpu_data_handle_t> inputs;
  const std::vector<starpu_data_handle_t> outputs;
  auto params = task.create_inference_params();
  auto ctx = make_callback_context(job, inputs, outputs, nullptr, params);
  ctx->dependencies_owner = owner;
  ASSERT_EQ(ctx->dependencies, nullptr);

  auto* created_task = task.create_task(inputs, outputs, ctx);
  ASSERT_NE(created_task, nullptr);

  EXPECT_EQ(ctx->dependencies, owner.get());
  EXPECT_EQ(ctx->dependencies_owner.get(), owner.get());

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

TEST_F(InferenceTaskTest, AllocateTaskBuffersUsesDependenciesOwner)
{
  auto job = make_job(8, 0);
  auto ctx = make_callback_context(job);

  g_owner_handles_alloc_calls = 0;
  g_owner_modes_alloc_calls = 0;

  auto owner = std::make_shared<starpu_server::InferenceTaskDependencies>(
      starpu_server::kDefaultInferenceTaskDependencies);
  owner->dyn_handles_allocator = &OwnerHandlesAllocator;
  owner->dyn_handles_deallocator = &OwnerAllocatorFree;
  owner->dyn_modes_allocator = &OwnerModesAllocator;
  owner->dyn_modes_deallocator = &OwnerAllocatorFree;

  ctx->dependencies_owner = owner;
  ctx->dependencies = nullptr;

  starpu_task task{};
  EXPECT_NO_THROW(
      starpu_server::InferenceTask::allocate_task_buffers(&task, 2, ctx));
  EXPECT_EQ(g_owner_handles_alloc_calls, 1);
  EXPECT_EQ(g_owner_modes_alloc_calls, 1);
  ASSERT_NE(task.dyn_handles, nullptr);
  ASSERT_NE(task.dyn_modes, nullptr);

  std::free(task.dyn_handles);
  std::free(task.dyn_modes);
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
  starpu_test::ScopedStarpuVectorRegisterOverride override(
      &AlwaysFailingVectorRegister);

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

  starpu_test::ScopedStarpuTaskSubmitOverride submit_override(
      &AlwaysFailingTaskSubmit);

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
  EXPECT_EQ(params->device.device_id, nullptr);
  EXPECT_EQ(params->device.worker_id, nullptr);
  EXPECT_EQ(params->device.executed_on, nullptr);
  ASSERT_TRUE(static_cast<bool>(params->device.set_executed_on));
  ASSERT_TRUE(static_cast<bool>(params->device.set_device_id));
  ASSERT_TRUE(static_cast<bool>(params->device.set_worker_id));
  ASSERT_TRUE(static_cast<bool>(params->timing.set_codelet_start_time));
  ASSERT_TRUE(static_cast<bool>(params->timing.set_codelet_end_time));
  ASSERT_TRUE(static_cast<bool>(params->timing.set_inference_start_time));

  const auto codelet_start = starpu_server::MonotonicClock::now();
  const auto inference_start = codelet_start + std::chrono::microseconds(100);
  const auto codelet_end = codelet_start + std::chrono::microseconds(250);
  params->device.set_executed_on(starpu_server::DeviceType::CUDA);
  params->device.set_device_id(3);
  params->device.set_worker_id(8);
  params->timing.set_codelet_start_time(codelet_start);
  params->timing.set_inference_start_time(inference_start);
  params->timing.set_codelet_end_time(codelet_end);

  const auto timing = job->timing_info_snapshot();
  EXPECT_EQ(job->get_executed_on(), starpu_server::DeviceType::CUDA);
  EXPECT_EQ(job->get_device_id(), 3);
  EXPECT_EQ(job->get_worker_id(), 8);
  EXPECT_EQ(timing.codelet_start_time, codelet_start);
  EXPECT_EQ(timing.inference_start_time, inference_start);
  EXPECT_EQ(timing.codelet_end_time, codelet_end);
  EXPECT_EQ(params->layout.num_dims[0], 2);
  EXPECT_EQ(params->layout.dims[0][0], 2);
  EXPECT_EQ(params->layout.dims[0][1], 3);
  EXPECT_EQ(params->layout.input_types[0], at::kFloat);
}

TEST_F(InferenceTaskTest, CreateInferenceParamsInfersInputTypesFromTensors)
{
  auto job = std::make_shared<starpu_server::InferenceJob>();
  job->set_request_id(8);
  job->set_input_tensors(
      {torch::ones({2}, torch::TensorOptions().dtype(at::kDouble)),
       torch::ones({3}, torch::TensorOptions().dtype(at::kInt))});
  job->set_input_types({});
  job->set_output_tensors({torch::zeros({2})});

  auto task = make_task(job);

  auto params = task.create_inference_params();

  ASSERT_EQ(params->layout.input_types.size(), 2U);
  EXPECT_EQ(params->layout.input_types[0], at::kDouble);
  EXPECT_EQ(params->layout.input_types[1], at::kInt);
}

TEST_F(InferenceTaskTest, CreateInferenceParamsThrowsWhenInputTensorUndefined)
{
  auto job = std::make_shared<starpu_server::InferenceJob>();
  job->set_request_id(10);
  job->set_input_tensors({torch::Tensor()});
  job->set_input_types({});

  auto task = make_task(job);

  EXPECT_THROW(
      {
        try {
          task.create_inference_params();
        }
        catch (const starpu_server::InferenceExecutionException& ex) {
          EXPECT_STREQ(
              "Input tensor is undefined; cannot infer input type.", ex.what());
          throw;
        }
      },
      starpu_server::InferenceExecutionException);
}

TEST_F(InferenceTaskTest, CreateInferenceParamsThrowsWhenRuntimeConfigNull)
{
  auto job = make_job(9, 1);
  model_cpu_ = starpu_server::make_add_one_model();
  models_gpu_.clear();
  starpu_server::InferenceTask task(
      nullptr, job, &model_cpu_, &models_gpu_, nullptr,
      starpu_server::kDefaultInferenceTaskDependencies);

  EXPECT_THROW(
      task.create_inference_params(),
      starpu_server::InferenceExecutionException);
}

TEST(InferenceTask, RecordAndRunCompletionCallback)
{
  auto job = std::make_shared<starpu_server::InferenceJob>();
  std::vector<torch::Tensor> outputs{torch::tensor({1})};
  job->set_output_tensors(outputs);
  bool called = false;
  std::vector<torch::Tensor> results_arg;
  double latency_ms = -1.0;
  job->completion().set_on_complete(
      [&called, &results_arg, &latency_ms](
          const std::vector<torch::Tensor>& results, double latency) {
        called = true;
        results_arg = results;
        latency_ms = latency;
      });
  const auto start = starpu_server::MonotonicClock::now();
  const auto end = start + std::chrono::milliseconds(5);
  job->set_start_time(start);
  auto ctx = make_callback_context(job);
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
  auto ctx = make_callback_context(job, {}, {nullptr}, &dependencies);
  const auto log = CaptureStderr([&] {
    EXPECT_NO_THROW(
        starpu_server::InferenceTask::starpu_output_callback(ctx.get()));
  });
  EXPECT_NE(log.find("starpu_output_callback"), std::string::npos);
}

TEST(InferenceTask, StarpuOutputCallbackLogsWhenContextIsNull)
{
  const auto log = CaptureStderr([&] {
    EXPECT_NO_THROW(
        starpu_server::InferenceTask::starpu_output_callback(nullptr));
  });
  EXPECT_NE(
      log.find("starpu_output_callback received null context"),
      std::string::npos);
}

TEST(InferenceTask, StarpuOutputCallbackFinalizesWhenOutputsEmpty)
{
  auto ctx = make_callback_context();
  ctx->self_keep_alive = ctx;
  bool finished = false;
  ctx->on_finished = [&]() { finished = true; };

  EXPECT_NO_THROW(
      starpu_server::InferenceTask::starpu_output_callback(ctx.get()));

  EXPECT_TRUE(finished);
  EXPECT_EQ(ctx->remaining_outputs_to_acquire.load(), 0);
  EXPECT_EQ(ctx->self_keep_alive, nullptr);
}

TEST_F(InferenceTaskTest, StarpuOutputCallbackAcquireFailureStillFinalizes)
{
  auto job = make_job(11, 1);
  auto handle = MakeHandle(0);
  starpu_server::InferenceTaskDependencies dependencies =
      starpu_server::kDefaultInferenceTaskDependencies;
  dependencies.starpu_data_acquire_fn = &AlwaysFailingAcquire;
  auto ctx = make_callback_context(job, {}, {handle}, &dependencies);
  ctx->self_keep_alive = ctx;
  bool finished = false;
  ctx->on_finished = [&]() { finished = true; };
  starpu_test::ScopedStarpuDataReleaseOverride release_override(
      &NoopDataRelease);

  const auto log = CaptureStderr([&] {
    EXPECT_NO_THROW(
        starpu_server::InferenceTask::starpu_output_callback(ctx.get()));
  });

  const auto failure = job->completion().failure_info();
  ASSERT_TRUE(failure.has_value());
  EXPECT_EQ(failure->stage, "callback");
  EXPECT_EQ(failure->reason, "output_acquire_failed");
  EXPECT_TRUE(finished);
  EXPECT_EQ(ctx->remaining_outputs_to_acquire.load(), 0);
  EXPECT_EQ(ctx->self_keep_alive, nullptr);
  EXPECT_NE(
      log.find("starpu_data_acquire_cb failed with code -42"),
      std::string::npos);
}

TEST_F(
    InferenceTaskTest,
    StarpuOutputCallbackImmediateAcquireKeepsContextAliveUntilLoopEnds)
{
  auto job = make_job(15, 1);
  job->set_output_tensors({torch::zeros({1}), torch::zeros({1})});

  starpu_server::InferenceTaskDependencies dependencies =
      starpu_server::kDefaultInferenceTaskDependencies;
  dependencies.starpu_data_acquire_fn = &ImmediateCallbackAcquire;

  auto ctx = make_callback_context(
      job, {}, {MakeHandle(24), MakeHandle(25)}, &dependencies);
  ctx->self_keep_alive = ctx;
  bool finished = false;
  ctx->on_finished = [&]() { finished = true; };
  starpu_test::ScopedStarpuDataReleaseOverride release_override(
      &NoopDataRelease);

  EXPECT_NO_THROW(
      starpu_server::InferenceTask::starpu_output_callback(ctx.get()));

  EXPECT_TRUE(finished);
  EXPECT_EQ(ctx->remaining_outputs_to_acquire.load(), 0);
  EXPECT_EQ(ctx->self_keep_alive, nullptr);
}

TEST_F(
    InferenceTaskTest,
    StarpuOutputCallbackBypassesRemainingHandlesAfterFirstFailure)
{
  auto job = make_job(12, 1);
  starpu_server::InferenceTaskDependencies dependencies =
      starpu_server::kDefaultInferenceTaskDependencies;
  dependencies.starpu_output_callback_hook =
      starpu_server::InferenceTaskDependencies::OutputCallbackHook(
          &ThrowingOnceStdStarpuOutputCallbackHook);

  auto ctx = make_callback_context(
      job, {}, {MakeHandle(20), MakeHandle(21)}, &dependencies);
  ctx->self_keep_alive = ctx;
  bool finished = false;
  ctx->on_finished = [&]() { finished = true; };
  throwing_once_output_callback_calls_ref() = 0;

  const auto log = CaptureStderr([&] {
    EXPECT_NO_THROW(
        starpu_server::InferenceTask::starpu_output_callback(ctx.get()));
  });

  EXPECT_TRUE(finished);
  EXPECT_EQ(ctx->remaining_outputs_to_acquire.load(), 0);
  EXPECT_EQ(ctx->self_keep_alive, nullptr);
  EXPECT_EQ(throwing_once_output_callback_calls_ref(), 1);
  ASSERT_EQ(ctx->outputs_handles_to_release.size(), 2U);
  EXPECT_EQ(ctx->outputs_handles_to_release[0], nullptr);
  EXPECT_EQ(ctx->outputs_handles_to_release[1], nullptr);
  EXPECT_NE(
      log.find(
          "std::exception in starpu_output_callback: forced bypass callback "
          "failure"),
      std::string::npos);
}

TEST_F(
    InferenceTaskTest,
    StarpuOutputCallbackUnknownExceptionMarksUnknownCallbackFailure)
{
  auto job = make_job(13, 1);
  starpu_server::InferenceTaskDependencies dependencies =
      starpu_server::kDefaultInferenceTaskDependencies;
  dependencies.starpu_output_callback_hook =
      starpu_server::InferenceTaskDependencies::OutputCallbackHook(
          &ThrowingUnknownStarpuOutputCallbackHook);

  auto ctx = make_callback_context(job, {}, {MakeHandle(22)}, &dependencies);
  ctx->self_keep_alive = ctx;
  bool finished = false;
  ctx->on_finished = [&]() { finished = true; };

  const auto log = CaptureStderr([&] {
    EXPECT_NO_THROW(
        starpu_server::InferenceTask::starpu_output_callback(ctx.get()));
  });

  const auto failure = job->completion().failure_info();
  ASSERT_TRUE(failure.has_value());
  EXPECT_EQ(failure->stage, "callback");
  EXPECT_EQ(failure->reason, "output_callback_unknown_exception");
  EXPECT_EQ(
      failure->message, "Unknown non-standard exception in output callback.");
  EXPECT_TRUE(finished);
  EXPECT_EQ(ctx->remaining_outputs_to_acquire.load(), 0);
  EXPECT_EQ(ctx->self_keep_alive, nullptr);
  EXPECT_NE(
      log.find("Unknown exception in starpu_output_callback"),
      std::string::npos);
}

TEST_F(
    InferenceTaskTest,
    StarpuOutputCallbackStdExceptionMarksOutputCallbackException)
{
  auto job = make_job(14, 1);
  starpu_server::InferenceTaskDependencies dependencies =
      starpu_server::kDefaultInferenceTaskDependencies;
  dependencies.starpu_output_callback_hook =
      starpu_server::InferenceTaskDependencies::OutputCallbackHook(
          &ThrowingStdStarpuOutputCallbackHook);

  auto ctx = make_callback_context(job, {}, {MakeHandle(23)}, &dependencies);
  ctx->self_keep_alive = ctx;
  bool finished = false;
  ctx->on_finished = [&]() { finished = true; };

  const auto log = CaptureStderr([&] {
    EXPECT_NO_THROW(
        starpu_server::InferenceTask::starpu_output_callback(ctx.get()));
  });

  const auto failure = job->completion().failure_info();
  ASSERT_TRUE(failure.has_value());
  EXPECT_EQ(failure->stage, "callback");
  EXPECT_EQ(failure->reason, "output_callback_exception");
  EXPECT_EQ(failure->message, "forced std callback failure");
  EXPECT_TRUE(finished);
  EXPECT_EQ(ctx->remaining_outputs_to_acquire.load(), 0);
  EXPECT_EQ(ctx->self_keep_alive, nullptr);
  EXPECT_NE(
      log.find("std::exception in starpu_output_callback: forced std callback "
               "failure"),
      std::string::npos);
}

TEST_F(InferenceTaskTest, StarpuOutputCallbackHookFailureWithNullJobFinalizes)
{
  starpu_server::InferenceTaskDependencies dependencies =
      starpu_server::kDefaultInferenceTaskDependencies;
  dependencies.starpu_output_callback_hook =
      starpu_server::InferenceTaskDependencies::OutputCallbackHook(
          &ThrowingStarpuOutputCallbackHook);

  auto ctx = make_callback_context(nullptr, {}, {MakeHandle(7)}, &dependencies);
  ctx->self_keep_alive = ctx;
  bool finished = false;
  ctx->on_finished = [&]() { finished = true; };

  starpu_server::CaptureStream capture{std::cerr};
  EXPECT_NO_THROW(
      starpu_server::InferenceTask::starpu_output_callback(ctx.get()));

  EXPECT_TRUE(finished);
  EXPECT_EQ(ctx->remaining_outputs_to_acquire.load(), 0);
  EXPECT_EQ(ctx->self_keep_alive, nullptr);
  EXPECT_NE(
      capture.str().find("std::exception in starpu_output_callback"),
      std::string::npos);
}

TEST(InferenceTask, FinalizeOrFailOnceLogsWhenContextIsNull)
{
  const auto log = CaptureStderr([&] {
    EXPECT_NO_THROW(starpu_server::testing::finalize_or_fail_once_for_tests(
        nullptr, false, "finalize_or_fail_once_test"));
  });

  EXPECT_NE(
      log.find("finalize_or_fail_once_test received a null callback context"),
      std::string::npos);
}

TEST(InferenceTask, FinalizeOrFailOnceSetsFailureWhenStatusIsFailure)
{
  auto job = std::make_shared<starpu_server::InferenceJob>();
  job->set_output_tensors({torch::tensor({1.0F})});

  auto ctx = make_callback_context(job);
  ctx->self_keep_alive = ctx;

  bool finished = false;
  ctx->on_finished = [&]() { finished = true; };

  EXPECT_NO_THROW(starpu_server::testing::finalize_or_fail_once_for_tests(
      ctx.get(), true, "finalize_or_fail_once_test"));

  const auto failure = job->completion().failure_info();
  ASSERT_TRUE(failure.has_value());
  EXPECT_EQ(failure->stage, "callback");
  EXPECT_EQ(failure->reason, "terminal_failure");
  EXPECT_EQ(
      failure->message,
      "Inference callback finalized after an unrecoverable error.");
  EXPECT_TRUE(job->get_output_tensors().empty());
  EXPECT_TRUE(finished);
  EXPECT_EQ(ctx->self_keep_alive, nullptr);
}

TEST(InferenceTask, FinalizeOrFailOnceSkipsWhenTerminalPathAlreadyStarted)
{
  auto job = std::make_shared<starpu_server::InferenceJob>();
  job->set_output_tensors({torch::tensor({2.0F})});

  auto ctx = make_callback_context(job);
  ctx->self_keep_alive = ctx;
  ctx->terminal_path_started.store(true, std::memory_order_release);

  bool finished = false;
  ctx->on_finished = [&]() { finished = true; };

  EXPECT_NO_THROW(starpu_server::testing::finalize_or_fail_once_for_tests(
      ctx.get(), false, "finalize_or_fail_once_test"));

  EXPECT_FALSE(finished);
  EXPECT_FALSE(job->completion().failure_info().has_value());
  EXPECT_FALSE(job->get_output_tensors().empty());
  EXPECT_NE(ctx->self_keep_alive, nullptr);
}

TEST(InferenceTask, FinalizeOrFailOnceLogsStdExceptionFromTerminalPath)
{
  auto ctx = make_callback_context(nullptr, {}, {MakeHandle(10)});
  starpu_test::ScopedStarpuDataReleaseOverride release_override(
      &ThrowingStdExceptionDataRelease);

  const auto log = CaptureStderr([&] {
    EXPECT_NO_THROW(starpu_server::testing::finalize_or_fail_once_for_tests(
        ctx.get(), false, "finalize_or_fail_once_std_exception_test"));
  });

  EXPECT_NE(
      log.find(
          "std::exception in finalize_or_fail_once_std_exception_test terminal "
          "path: forced std data release failure"),
      std::string::npos);
}

TEST(InferenceTask, FinalizeOrFailOnceLogsUnknownExceptionFromTerminalPath)
{
  auto ctx = make_callback_context(nullptr, {}, {MakeHandle(11)});
  starpu_test::ScopedStarpuDataReleaseOverride release_override(
      &ThrowingUnknownDataRelease);

  const auto log = CaptureStderr([&] {
    EXPECT_NO_THROW(starpu_server::testing::finalize_or_fail_once_for_tests(
        ctx.get(), false, "finalize_or_fail_once_unknown_exception_test"));
  });

  EXPECT_NE(
      log.find("Unknown exception in "
               "finalize_or_fail_once_unknown_exception_test terminal path"),
      std::string::npos);
}

TEST(InferenceTask, RecordAndRunCompletionCallbackLogsStdException)
{
  auto job = std::make_shared<starpu_server::InferenceJob>();
  job->set_output_tensors({torch::tensor({1})});
  job->completion().set_on_complete(
      [](const std::vector<torch::Tensor>&, double) {
        throw std::runtime_error("callback failure");
      });
  const auto start = starpu_server::MonotonicClock::now();
  const auto end = start + std::chrono::milliseconds(5);
  job->set_start_time(start);
  auto ctx = make_callback_context(job);
  const auto log = CaptureStderr([&] {
    EXPECT_NO_THROW(
        starpu_server::InferenceTask::record_and_run_completion_callback(
            ctx.get(), end));
  });
  EXPECT_NE(
      log.find("Exception in completion callback: callback failure"),
      std::string::npos);
}

TEST(InferenceTask, RecordAndRunCompletionCallbackLogsUnknownException)
{
  auto job = std::make_shared<starpu_server::InferenceJob>();
  job->set_output_tensors({torch::tensor({1})});
  job->completion().set_on_complete(
      [](const std::vector<torch::Tensor>&, double) { throw 42; });
  const auto start = starpu_server::MonotonicClock::now();
  const auto end = start + std::chrono::milliseconds(5);
  job->set_start_time(start);
  auto ctx = make_callback_context(job);
  const auto log = CaptureStderr([&] {
    EXPECT_NO_THROW(
        starpu_server::InferenceTask::record_and_run_completion_callback(
            ctx.get(), end));
  });
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
  auto ctx = make_callback_context(job, {}, outputs, &dependencies);
  ctx->remaining_outputs_to_acquire = static_cast<int>(outputs.size());
  const auto log = CaptureStderr([&] {
    EXPECT_THROW(
        starpu_server::InferenceTask::acquire_output_handle(handle, ctx.get()),
        starpu_server::StarPURegistrationException);
  });
  const auto expected = starpu_server::expected_log_line(
      starpu_server::ErrorLevel,
      std::format("starpu_data_acquire_cb failed with code {}", -42));
  EXPECT_NE(log.find(expected), std::string::npos);
}

TEST(InferenceTask, AcquireOutputHandleLogsInferenceEngineException)
{
  StarpuRuntimeGuard starpu_guard;
  starpu_test::ScopedStarpuDataReleaseOverride release_override(
      &ThrowingDataRelease);
  OutputContextFixture fixture;
  auto ctx = fixture.ctx;
  ctx->self_keep_alive = ctx;
  auto handle = MakeHandle(0);
  ctx->outputs_handles = {handle};
  ctx->remaining_outputs_to_acquire = 1;
  data_release_throw_count_ref() = 0;
  starpu_server::InferenceTaskDependencies dependencies =
      starpu_server::kDefaultInferenceTaskDependencies;
  dependencies.starpu_data_acquire_fn = &ImmediateCallbackAcquire;
  ctx->dependencies = &dependencies;
  const auto log = CaptureStderr([&] {
    EXPECT_NO_THROW(
        starpu_server::InferenceTask::acquire_output_handle(handle, ctx.get()));
  });
  ctx->self_keep_alive.reset();
  ctx->outputs_handles.clear();
  EXPECT_EQ(data_release_throw_count_ref(), 1);
  EXPECT_NE(log.find("forced data release failure"), std::string::npos);
}

TEST(InferenceTask, AcquireOutputHandleThrowsWhenDataAcquireFunctionMissing)
{
  StarpuRuntimeGuard starpu_guard;
  ScopedDefaultDataAcquireNullifier nullifier;
  ASSERT_TRUE(nullifier.is_valid()) << "Failed to patch default dependencies";
  ASSERT_EQ(
      starpu_server::kDefaultInferenceTaskDependencies.starpu_data_acquire_fn,
      nullptr);
  auto handle = MakeHandle(1);
  auto outputs = std::vector<starpu_data_handle_t>{handle};
  auto ctx = make_callback_context(nullptr, {}, outputs);
  ctx->remaining_outputs_to_acquire = static_cast<int>(outputs.size());
  const auto log = CaptureStderr([&] {
    EXPECT_THROW(
        starpu_server::InferenceTask::acquire_output_handle(handle, ctx.get()),
        starpu_server::StarPURegistrationException);
  });
  EXPECT_NE(
      log.find("starpu_data_acquire_fn is null; cannot acquire output handle."),
      std::string::npos);
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
  auto ctx = make_callback_context(nullptr, inputs, outputs);
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

TEST(InferenceTask, FinalizeInferenceTaskBindsOutputViewsFromPool)
{
  constexpr float kSentinelValue = 42.0F;
  OutputContextFixture fixture({.sentinel_value = kSentinelValue});

  ASSERT_NO_THROW(
      starpu_server::InferenceTask::finalize_inference_task(fixture.ctx.get()));

  const auto& job_outputs = fixture.job->get_output_tensors();
  ASSERT_EQ(job_outputs.size(), 1U);
  ASSERT_TRUE(job_outputs[0].defined());
  EXPECT_EQ(
      job_outputs[0].data_ptr(), fixture.pool.base_ptrs(fixture.slot_id)[0]);
  EXPECT_FLOAT_EQ(job_outputs[0].item<float>(), kSentinelValue);

  EXPECT_FALSE(fixture.pool.try_acquire().has_value());

  fixture.job->set_output_tensors({});

  auto reacquired = fixture.pool.try_acquire();
  ASSERT_TRUE(reacquired.has_value());
  EXPECT_EQ(*reacquired, fixture.slot_id);
  fixture.pool.release(*reacquired);
}

TEST(InferenceTask, FinalizeInferenceTaskHandlesOutputViewBindingFailure)
{
  OutputContextFixture fixture({.mutate_job_outputs = [](auto& outputs) {
    outputs[0] = torch::Tensor();
  }});
  const auto log = CaptureStderr([&] {
    EXPECT_NO_THROW(starpu_server::InferenceTask::finalize_inference_task(
        fixture.ctx.get()));
  });
  EXPECT_NE(
      log.find("Output view binding from pool failed"), std::string::npos);

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
      .on_finished = [](auto&, int) { throw std::runtime_error("boom"); },
  });
  const auto log = CaptureStderr([&] {
    EXPECT_NO_THROW(starpu_server::InferenceTask::finalize_inference_task(
        fixture.ctx.get()));
  });
  EXPECT_NE(log.find("Exception in on_finished"), std::string::npos);

  const auto& job_outputs = fixture.job->get_output_tensors();
  ASSERT_EQ(job_outputs.size(), 1U);
  ASSERT_TRUE(job_outputs[0].defined());
  EXPECT_FLOAT_EQ(job_outputs[0].item<float>(), kSentinelValue);

  EXPECT_FALSE(fixture.pool.try_acquire().has_value());

  fixture.job->set_output_tensors({});

  auto reacquired = fixture.pool.try_acquire();
  ASSERT_TRUE(reacquired.has_value());
  EXPECT_EQ(*reacquired, fixture.slot_id);
  fixture.pool.release(*reacquired);
}

TEST(InferenceTask, FinalizeInferenceTaskKeepsOutputSlotAliveForCallbackCopies)
{
  constexpr float kSentinelValue = 13.0F;
  OutputContextFixture fixture({.sentinel_value = kSentinelValue});

  std::vector<torch::Tensor> callback_outputs;
  fixture.job->completion().set_on_complete(
      [&callback_outputs](std::vector<torch::Tensor> outputs, double) {
        callback_outputs = std::move(outputs);
      });

  ASSERT_NO_THROW(
      starpu_server::InferenceTask::finalize_inference_task(fixture.ctx.get()));

  ASSERT_EQ(callback_outputs.size(), 1U);
  EXPECT_FLOAT_EQ(callback_outputs[0].item<float>(), kSentinelValue);
  EXPECT_FALSE(fixture.pool.try_acquire().has_value());

  fixture.job->set_output_tensors({});
  EXPECT_FALSE(fixture.pool.try_acquire().has_value());

  callback_outputs.clear();

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
  auto ctx = make_callback_context(nullptr, {}, {}, &dependencies);
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
  auto ctx = make_callback_context(nullptr, {}, {}, &dependencies);
  starpu_task task{};
  EXPECT_THROW(
      starpu_server::InferenceTask::allocate_task_buffers(&task, 2, ctx),
      starpu_server::MemoryAllocationException);
  EXPECT_EQ(task.dyn_handles, nullptr);
  EXPECT_EQ(task.dyn_modes, nullptr);
}
