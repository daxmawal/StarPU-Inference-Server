#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <starpu.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <format>
#include <limits>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "core/input_slot_pool.hpp"
#include "core/starpu_setup.hpp"
#include "support/input_slot_pool_test_hooks.hpp"
#include "support/output_slot_pool_test_hooks.hpp"
#include "test_utils.hpp"
#include "utils/runtime_config.hpp"

namespace {

std::vector<std::byte*> g_observed_base_ptrs;
std::vector<starpu_data_handle_t> g_observed_handles;
bool g_failure_observer_called = false;

int g_succeed_then_fail_register_call_count = 0;
starpu_server::testing::StarpuVectorRegisterFn
    g_succeed_then_fail_previous_hook = nullptr;
std::vector<starpu_data_handle_t>* g_pre_cleanup_registered_handles = nullptr;

std::vector<std::byte*> g_output_observed_base_ptrs;
std::vector<starpu_data_handle_t> g_output_observed_handles;
std::vector<starpu_server::OutputSlotPool::HostBufferInfo>
    g_output_observed_host_buffer_infos;
bool g_output_failure_observer_called = false;

void
capture_slot_state(const starpu_server::InputSlotPool::SlotInfo& slot)
{
  g_failure_observer_called = true;
  g_observed_base_ptrs.assign(slot.base_ptrs.begin(), slot.base_ptrs.end());
  g_observed_handles.assign(slot.handles.begin(), slot.handles.end());
}

void
capture_output_slot_state(
    const starpu_server::OutputSlotPool::SlotInfo& slot,
    const std::vector<starpu_server::OutputSlotPool::HostBufferInfo>&
        buffer_infos)
{
  g_output_failure_observer_called = true;
  g_output_observed_base_ptrs.assign(
      slot.base_ptrs.begin(), slot.base_ptrs.end());
  g_output_observed_handles.assign(slot.handles.begin(), slot.handles.end());
  g_output_observed_host_buffer_infos.assign(
      buffer_infos.begin(), buffer_infos.end());
}

template <typename Fn>
struct FunctionArguments;

template <typename R, typename... Args>
struct FunctionArguments<R (*)(Args...)> {
  using Tuple = std::tuple<Args...>;
};

using StarpuRegisterArgs =
    FunctionArguments<starpu_server::testing::StarpuVectorRegisterFn>::Tuple;

void
failing_starpu_vector_register(
    std::tuple_element_t<0, StarpuRegisterArgs> handle,
    std::tuple_element_t<1, StarpuRegisterArgs> /*home_node*/,
    std::tuple_element_t<2, StarpuRegisterArgs> /*ptr*/,
    std::tuple_element_t<3, StarpuRegisterArgs> /*numel*/,
    std::tuple_element_t<4, StarpuRegisterArgs> /*element_size*/)
{
  *handle = nullptr;
}

void
succeed_then_fail_starpu_vector_register(
    std::tuple_element_t<0, StarpuRegisterArgs> handle,
    std::tuple_element_t<1, StarpuRegisterArgs> home_node,
    std::tuple_element_t<2, StarpuRegisterArgs> ptr,
    std::tuple_element_t<3, StarpuRegisterArgs> numel,
    std::tuple_element_t<4, StarpuRegisterArgs> element_size)
{
  ++g_succeed_then_fail_register_call_count;
  if (g_succeed_then_fail_register_call_count == 1) {
    if (g_succeed_then_fail_previous_hook != nullptr) {
      g_succeed_then_fail_previous_hook(
          handle, home_node, ptr, numel, element_size);
    }
    if (g_pre_cleanup_registered_handles != nullptr) {
      g_pre_cleanup_registered_handles->push_back(*handle);
    }
    return;
  }

  *handle = nullptr;
}

auto
failing_host_allocator(void** ptr, size_t /*alignment*/, size_t /*size*/) -> int
{
  if (ptr != nullptr) {
    *ptr = nullptr;
  }
  return 1;
}

auto
force_cuda_host_alloc_failure(
    size_t /*bytes*/, bool /*use_pinned*/, bool /*default_cuda_pinned*/) -> bool
{
  return false;
}

auto
starpu_memory_pin_success(void* /*ptr*/, size_t /*size*/) -> int
{
  return 0;
}

constexpr int kStarpuPinTestError = -42;

auto
starpu_memory_pin_failure(void* /*ptr*/, size_t /*size*/) -> int
{
  return kStarpuPinTestError;
}

auto
failing_starpu_init(struct starpu_conf*) -> int
{
  return -1;
}

auto
stub_starpu_init(struct starpu_conf*) -> int
{
  return 0;
}

class StarPUSetupInitOverrideTest : public ::testing::Test {
 protected:
  void TearDown() override
  {
    starpu_server::StarPUSetup::reset_starpu_init_fn();
  }
};

class StarPUSetupInitStubTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    starpu_server::StarPUSetup::set_starpu_init_fn(&stub_starpu_init);
  }

  void TearDown() override
  {
    starpu_server::StarPUSetup::reset_starpu_init_fn();
  }
};

}  // namespace

TEST_F(StarPUSetupInitOverrideTest, FailingStarpuInitThrows)
{
  starpu_server::RuntimeConfig opts;

  starpu_server::StarPUSetup::set_starpu_init_fn(&failing_starpu_init);

  EXPECT_THROW(
      {
        try {
          starpu_server::StarPUSetup setup(opts);
        }
        catch (const starpu_server::StarPUInitializationException& ex) {
          EXPECT_STREQ("[ERROR] StarPU initialization error", ex.what());
          throw;
        }
      },
      starpu_server::StarPUInitializationException);
}

TEST(StarPUSetup_Unit, DuplicateDeviceIdsThrows)
{
  starpu_server::RuntimeConfig opts;
  opts.devices.use_cuda = true;
  opts.devices.ids = {0, 0};
  EXPECT_THROW(
      { starpu_server::StarPUSetup setup(opts); }, std::invalid_argument);
}

TEST(OutputSlotPool_Unit, CheckedTotalNumelGuard)
{
  EXPECT_NO_THROW(
      starpu_server::OutputSlotPoolTestHook::checked_total_numel(16, 8));

  EXPECT_THROW(
      {
        starpu_server::OutputSlotPoolTestHook::checked_total_numel(
            std::numeric_limits<size_t>::max(), 2);
      },
      std::overflow_error);
}

TEST(OutputSlotPool_Unit, FreeHostBufferNullPointerNoWarning)
{
  starpu_server::OutputSlotPool::HostBufferInfo info{};
  starpu_server::CaptureStream capture{std::cerr};
  starpu_server::OutputSlotPoolTestHook::free_host_buffer_for_tests(
      nullptr, info);
  EXPECT_TRUE(capture.str().empty());
}

TEST(OutputSlotPool_Unit, FreeHostBufferStarpuUnpinFailureLogsWarning)
{
  constexpr size_t kBytes = 32;
  auto* ptr = static_cast<std::byte*>(std::malloc(kBytes));
  ASSERT_NE(ptr, nullptr);

  starpu_server::OutputSlotPool::HostBufferInfo info{};
  info.starpu_pinned = true;
  info.bytes = kBytes;

  starpu_server::CaptureStream capture{std::cerr};
  starpu_server::OutputSlotPoolTestHook::free_host_buffer_for_tests(ptr, info);
  const std::string log = capture.str();
  EXPECT_NE(log.find("starpu_memory_unpin failed"), std::string::npos);
}

TEST(OutputSlotPool_Unit, FreeHostBufferCudaFreeHostFailureLogsWarning)
{
  int device_count = 0;
  const cudaError_t device_rc = cudaGetDeviceCount(&device_count);
  if (device_rc == cudaErrorInsufficientDriver ||
      device_rc == cudaErrorNoDevice) {
    GTEST_SKIP() << "CUDA runtime unavailable for test: rc="
                 << static_cast<int>(device_rc);
  }
  if (device_rc != cudaSuccess) {
    GTEST_SKIP() << "CUDA runtime check failed: rc="
                 << static_cast<int>(device_rc);
  }

  constexpr size_t kBytes = 32;
  auto* ptr = static_cast<std::byte*>(std::malloc(kBytes));
  ASSERT_NE(ptr, nullptr);

  starpu_server::OutputSlotPool::HostBufferInfo info{};
  info.cuda_pinned = true;
  info.bytes = kBytes;

  starpu_server::CaptureStream capture{std::cerr};
  starpu_server::OutputSlotPoolTestHook::free_host_buffer_for_tests(ptr, info);
  const std::string log = capture.str();
  EXPECT_NE(log.find("cudaFreeHost failed"), std::string::npos);

  if (log.find("cudaFreeHost failed") != std::string::npos) {
    std::free(ptr);
  }
}

TEST(StarPUSetup_Unit, TooManyDeviceIdsThrows)
{
  starpu_server::RuntimeConfig opts;
  opts.devices.use_cuda = true;

  opts.devices.ids.reserve(STARPU_NMAXWORKERS + 1);
  for (int idx = 0; idx < STARPU_NMAXWORKERS + 1; ++idx) {
    opts.devices.ids.push_back(idx);
  }

  EXPECT_THROW(
      {
        try {
          starpu_server::StarPUSetup setup(opts);
        }
        catch (const std::invalid_argument& ex) {
          EXPECT_STREQ(
              std::format(
                  "[ERROR] Number of CUDA device IDs exceeds maximum of {}",
                  STARPU_NMAXWORKERS)
                  .c_str(),
              ex.what());
          throw;
        }
      },
      std::invalid_argument);
}

TEST_F(
    StarPUSetupInitStubTest, StarPUSetup_InputPoolInitFailureLogsAndPropagates)
{
  starpu_server::RuntimeConfig opts;
  opts.batching.input_slots = 1;

  starpu_server::TensorConfig invalid_input;
  invalid_input.name = "invalid_input";
  invalid_input.dims = {0, 1};
  invalid_input.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "invalid_input_model";
  model.inputs.push_back(invalid_input);
  opts.models.push_back(model);

  testing::internal::CaptureStderr();
  EXPECT_THROW(
      {
        try {
          starpu_server::StarPUSetup setup(opts);
        }
        catch (const std::invalid_argument& ex) {
          EXPECT_STREQ("dims[0] (batch) must be positive", ex.what());
          throw;
        }
      },
      std::invalid_argument);
  const std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_NE(
      log_output.find("Failed to initialize InputSlotPool"), std::string::npos);
}

TEST_F(
    StarPUSetupInitStubTest, StarPUSetup_OutputPoolInitFailureLogsAndPropagates)
{
  starpu_server::RuntimeConfig opts;
  opts.batching.input_slots = 1;

  starpu_server::TensorConfig invalid_output;
  invalid_output.name = "invalid_output";
  invalid_output.dims = {0, 1};
  invalid_output.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "invalid_output_model";
  model.outputs.push_back(invalid_output);
  opts.models.push_back(model);

  testing::internal::CaptureStderr();
  EXPECT_THROW(
      {
        try {
          starpu_server::StarPUSetup setup(opts);
        }
        catch (const std::invalid_argument& ex) {
          EXPECT_STREQ("dims[0] (batch) must be positive", ex.what());
          throw;
        }
      },
      std::invalid_argument);
  const std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_NE(
      log_output.find("Failed to initialize OutputSlotPool"),
      std::string::npos);
}

TEST(InputSlotPool_Unit, AllocateSlotBuffersOverflowThrows)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = std::numeric_limits<int>::max();
  opts.batching.input_slots = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "large_input";
  tensor.dims = {1, 65536, 65536};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "overflow_model";
  model.inputs.push_back(tensor);
  opts.models.push_back(model);

  EXPECT_THROW(
      { starpu_server::InputSlotPool pool(opts, 1); }, std::overflow_error);
}

TEST(InputSlotPool_Unit, AllocateSlotBuffersNumelOverflowThrows)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 5;
  opts.batching.input_slots = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "numel_overflow_input";
  tensor.dims = {1, 4611686018427387905};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "numel_overflow_model";
  model.inputs.push_back(tensor);
  opts.models.push_back(model);

  EXPECT_THROW(
      { starpu_server::InputSlotPool pool(opts, 1); }, std::overflow_error);
}

TEST(InputSlotPool_Unit, ConstructionWithoutModelsThrows)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;

  EXPECT_THROW(
      { starpu_server::InputSlotPool pool(opts, 1); }, std::invalid_argument);
}

TEST(InputSlotPool_Unit, NonPositiveBatchDimensionThrows)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "invalid_batch_input";
  tensor.dims = {0, 1};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "invalid_batch_model";
  model.inputs.push_back(tensor);
  opts.models.push_back(model);

  EXPECT_THROW(
      {
        try {
          starpu_server::InputSlotPool pool(opts, 1);
        }
        catch (const std::invalid_argument& ex) {
          EXPECT_STREQ("dims[0] (batch) must be positive", ex.what());
          throw;
        }
      },
      std::invalid_argument);
}

TEST(InputSlotPool_Unit, BatchDimensionExceedsIntMaxThrows)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "exceeds_int_max_input";
  tensor.dims = {static_cast<int64_t>(std::numeric_limits<int>::max()) + 1, 1};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "exceeds_int_max_model";
  model.inputs.push_back(tensor);
  opts.models.push_back(model);

  EXPECT_THROW(
      {
        try {
          starpu_server::InputSlotPool pool(opts, 1);
        }
        catch (const std::invalid_argument& ex) {
          EXPECT_STREQ("dims[0] (batch) exceeds int max", ex.what());
          throw;
        }
      },
      std::invalid_argument);
}

TEST(InputSlotPool_Unit, NonPositiveDimensionThrows)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 5;
  opts.batching.input_slots = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "non_positive_dims";
  tensor.dims = {1, 0, 8};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "non_positive_model";
  model.inputs.push_back(tensor);
  opts.models.push_back(model);

  EXPECT_THROW(
      {
        try {
          starpu_server::InputSlotPool pool(opts, 1);
        }
        catch (const std::invalid_argument& ex) {
          EXPECT_NE(
              std::string(ex.what()).find("dims must be positive"),
              std::string::npos);
          throw;
        }
      },
      std::invalid_argument);
}

TEST(InputSlotPool_Unit, DimensionProductOverflowThrows)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 5;
  opts.batching.input_slots = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "product_overflow_input";
  tensor.dims = {1, std::numeric_limits<int64_t>::max(), 3};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "product_overflow_model";
  model.inputs.push_back(tensor);
  opts.models.push_back(model);

  EXPECT_THROW(
      { starpu_server::InputSlotPool pool(opts, 1); }, std::overflow_error);
}

TEST(InputSlotPool_Unit, SlotInfoProvidesConsistentReferences)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "minimal_input";
  tensor.dims = {1, 1};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "minimal_model";
  model.inputs.push_back(tensor);
  opts.models.push_back(model);

  starpu_server::InputSlotPool pool(opts, 1);

  const int slot_id = pool.acquire();
  const auto& info = pool.slot_info(slot_id);

  EXPECT_EQ(info.id, slot_id);
  ASSERT_EQ(info.base_ptrs.size(), model.inputs.size());
  ASSERT_EQ(info.handles.size(), model.inputs.size());

  const auto& base_ptrs_ref = pool.base_ptrs(slot_id);
  const auto& handles_ref = pool.handles(slot_id);

  EXPECT_EQ(base_ptrs_ref.size(), info.base_ptrs.size());
  EXPECT_EQ(handles_ref.size(), info.handles.size());

  EXPECT_EQ(
      static_cast<const void*>(&base_ptrs_ref),
      static_cast<const void*>(&info.base_ptrs));
  EXPECT_EQ(
      static_cast<const void*>(&handles_ref),
      static_cast<const void*>(&info.handles));

  EXPECT_THROW(
      static_cast<void>(pool.slot_info(slot_id + 1)), std::out_of_range);
  EXPECT_THROW(
      static_cast<void>(pool.base_ptrs(slot_id + 1)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(pool.handles(slot_id + 1)), std::out_of_range);

  pool.release(slot_id);
}

TEST(InputSlotPool_Unit, TryAcquireEmptyPoolReturnsNullopt)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 1;
  opts.batching.input_slots = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "minimal_input";
  tensor.dims = {1, 1};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "minimal_model";
  model.inputs.push_back(tensor);
  opts.models.push_back(model);

  starpu_server::InputSlotPool pool(opts, 1);

  const int slot_id = pool.acquire();
  EXPECT_EQ(pool.try_acquire(), std::nullopt);

  pool.release(slot_id);

  const auto reacquired = pool.try_acquire();
  ASSERT_TRUE(reacquired.has_value());
  EXPECT_EQ(*reacquired, slot_id);
  pool.release(*reacquired);
}

TEST(InputSlotPool_Unit, HostBufferInfoIndicatesCudaPinningAttempt)
{
  StarpuRuntimeGuard starpu_guard;

  void* probe_ptr = nullptr;
  const cudaError_t probe_err =
      cudaHostAlloc(&probe_ptr, 1, cudaHostAllocPortable);
  if (probe_err == cudaSuccess) {
    cudaFreeHost(probe_ptr);
  } else if (
      probe_err == cudaErrorNotSupported ||
      probe_err == cudaErrorInsufficientDriver ||
      probe_err == cudaErrorNoDevice) {
    GTEST_SKIP() << "cudaHostAlloc unsupported: "
                 << cudaGetErrorString(probe_err);
  }

  starpu_server::RuntimeConfig opts;
  opts.devices.use_cuda = true;
  opts.batching.max_batch_size = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "cuda_probe_input";
  tensor.dims = {1, 1};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "cuda_probe_model";
  model.inputs.push_back(tensor);
  opts.models.push_back(model);

  starpu_server::InputSlotPool pool(opts, 1);

  const int slot_id = pool.acquire();
  const auto& buffer_infos = pool.host_buffer_infos(slot_id);
  ASSERT_EQ(buffer_infos.size(), 1);

  const auto& info = buffer_infos.front();

  if (info.cuda_pinned) {
    EXPECT_TRUE(info.cuda_pinned);
  } else {
    EXPECT_FALSE(info.cuda_pinned);
    EXPECT_TRUE(info.starpu_pinned || info.starpu_pin_rc != 0)
        << "Fallback StarPU pinning should report a result";
  }

  pool.release(slot_id);
}

TEST(OutputSlotPool_Unit, HostBufferInfoIndicatesCudaPinningAttempt)
{
  StarpuRuntimeGuard starpu_guard;

  void* probe_ptr = nullptr;
  const cudaError_t probe_err =
      cudaHostAlloc(&probe_ptr, 1, cudaHostAllocPortable);
  if (probe_err == cudaSuccess) {
    cudaFreeHost(probe_ptr);
  } else if (
      probe_err == cudaErrorNotSupported ||
      probe_err == cudaErrorInsufficientDriver ||
      probe_err == cudaErrorNoDevice) {
    GTEST_SKIP() << "cudaHostAlloc unsupported: "
                 << cudaGetErrorString(probe_err);
  }

  starpu_server::RuntimeConfig opts;
  opts.devices.use_cuda = true;
  opts.batching.max_batch_size = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "cuda_probe_output";
  tensor.dims = {1, 1};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "cuda_probe_model";
  model.outputs.push_back(tensor);
  opts.models.push_back(model);

  starpu_server::OutputSlotPool pool(opts, 1);

  const int slot_id = pool.acquire();
  const auto& buffer_infos =
      starpu_server::OutputSlotPoolTestHook::host_buffer_infos(pool, slot_id);
  ASSERT_EQ(buffer_infos.size(), 1);

  const auto& info = buffer_infos.front();

  if (info.cuda_pinned) {
    EXPECT_TRUE(info.cuda_pinned);
  } else {
    EXPECT_FALSE(info.cuda_pinned);
    EXPECT_TRUE(info.starpu_pinned || info.starpu_pin_rc != 0)
        << "Fallback StarPU pinning should report a result";
  }

  pool.release(slot_id);
}

TEST(
    OutputSlotPool_Unit,
    HostBufferInfoReportsStarpuPinSuccessWhenCudaHostAllocFails)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.devices.use_cuda = true;
  opts.batching.max_batch_size = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "starpu_pin_success";
  tensor.dims = {1, 1};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "starpu_pin_success_model";
  model.outputs.push_back(tensor);
  opts.models.push_back(model);

  const auto previous_cuda_override =
      starpu_server::testing::set_output_cuda_pinned_override_for_tests(
          &force_cuda_host_alloc_failure);
  const auto previous_pin_hook =
      starpu_server::testing::set_output_starpu_memory_pin_hook_for_tests(
          &starpu_memory_pin_success);

  auto restore_hooks = [&]() {
    starpu_server::testing::set_output_starpu_memory_pin_hook_for_tests(
        previous_pin_hook);
    starpu_server::testing::set_output_cuda_pinned_override_for_tests(
        previous_cuda_override);
  };

  try {
    starpu_server::OutputSlotPool pool(opts, 1);

    const int slot_id = pool.acquire();
    const auto& buffer_infos =
        starpu_server::OutputSlotPoolTestHook::host_buffer_infos(pool, slot_id);
    ASSERT_EQ(buffer_infos.size(), 1);

    const auto& info = buffer_infos.front();
    EXPECT_FALSE(info.cuda_pinned);
    EXPECT_TRUE(info.starpu_pinned);
    EXPECT_EQ(info.starpu_pin_rc, 0);

    pool.release(slot_id);
    restore_hooks();
  }
  catch (...) {
    restore_hooks();
    throw;
  }
}

TEST(
    OutputSlotPool_Unit,
    HostBufferInfoCapturesStarpuPinFailureWhenCudaHostAllocFails)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.devices.use_cuda = true;
  opts.batching.max_batch_size = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "starpu_pin_failure";
  tensor.dims = {1, 1};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "starpu_pin_failure_model";
  model.outputs.push_back(tensor);
  opts.models.push_back(model);

  const auto previous_cuda_override =
      starpu_server::testing::set_output_cuda_pinned_override_for_tests(
          &force_cuda_host_alloc_failure);
  const auto previous_pin_hook =
      starpu_server::testing::set_output_starpu_memory_pin_hook_for_tests(
          &starpu_memory_pin_failure);

  auto restore_hooks = [&]() {
    starpu_server::testing::set_output_starpu_memory_pin_hook_for_tests(
        previous_pin_hook);
    starpu_server::testing::set_output_cuda_pinned_override_for_tests(
        previous_cuda_override);
  };

  try {
    starpu_server::OutputSlotPool pool(opts, 1);

    const int slot_id = pool.acquire();
    const auto& buffer_infos =
        starpu_server::OutputSlotPoolTestHook::host_buffer_infos(pool, slot_id);
    ASSERT_EQ(buffer_infos.size(), 1);

    const auto& info = buffer_infos.front();
    EXPECT_FALSE(info.cuda_pinned);
    EXPECT_FALSE(info.starpu_pinned);
    EXPECT_EQ(info.starpu_pin_rc, kStarpuPinTestError);

    pool.release(slot_id);
    restore_hooks();
  }
  catch (...) {
    restore_hooks();
    throw;
  }
}

TEST(OutputSlotPool_Unit, HostAllocatorFailureThrowsBadAlloc)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "host_allocator_failure";
  tensor.dims = {1, 1};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "host_allocator_failure_model";
  model.outputs.push_back(tensor);
  opts.models.push_back(model);

  const auto previous_allocator =
      starpu_server::testing::set_output_host_allocator_for_tests(
          &failing_host_allocator);

  auto restore_allocator = [&]() {
    starpu_server::testing::set_output_host_allocator_for_tests(
        previous_allocator);
  };

  EXPECT_THROW(
      {
        try {
          starpu_server::OutputSlotPool pool(opts, 1);
          restore_allocator();
          FAIL() << "Expected host allocator failure";
        }
        catch (...) {
          restore_allocator();
          throw;
        }
      },
      std::bad_alloc);
}

TEST(InputSlotPool_Unit, RegisterFailureResetsSlotState)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 1;
  opts.batching.input_slots = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "failing_input";
  tensor.dims = {1, 1};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "failing_model";
  model.inputs.push_back(tensor);
  opts.models.push_back(model);

  g_observed_base_ptrs.clear();
  g_observed_handles.clear();
  g_failure_observer_called = false;

  const auto previous_hook =
      starpu_server::testing::set_starpu_vector_register_hook_for_tests(
          &failing_starpu_vector_register);
  const auto previous_observer =
      starpu_server::testing::set_starpu_register_failure_observer_for_tests(
          &capture_slot_state);

  auto restore_hooks = [&]() {
    starpu_server::testing::set_starpu_register_failure_observer_for_tests(
        previous_observer);
    starpu_server::testing::set_starpu_vector_register_hook_for_tests(
        previous_hook);
  };

  EXPECT_THROW(
      {
        try {
          starpu_server::InputSlotPool pool(opts, 1);
          restore_hooks();
          FAIL() << "Expected StarPU handle registration failure";
        }
        catch (...) {
          restore_hooks();
          throw;
        }
      },
      std::runtime_error);

  ASSERT_TRUE(g_failure_observer_called);
  ASSERT_EQ(g_observed_base_ptrs.size(), model.inputs.size());
  ASSERT_EQ(g_observed_handles.size(), model.inputs.size());

  for (std::byte* base_ptr : g_observed_base_ptrs) {
    EXPECT_EQ(base_ptr, nullptr);
  }

  for (auto handle : g_observed_handles) {
    EXPECT_EQ(handle, nullptr);
  }
}

TEST(InputSlotPool_Unit, PartialRegisterFailureResetsSlotState)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 1;
  opts.batching.input_slots = 1;

  starpu_server::TensorConfig first_tensor;
  first_tensor.name = "first_input";
  first_tensor.dims = {1, 1};
  first_tensor.type = at::ScalarType::Float;

  starpu_server::TensorConfig second_tensor;
  second_tensor.name = "second_input";
  second_tensor.dims = {1, 1};
  second_tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "partial_failure_model";
  model.inputs.push_back(first_tensor);
  model.inputs.push_back(second_tensor);
  opts.models.push_back(model);

  g_observed_base_ptrs.clear();
  g_observed_handles.clear();
  g_failure_observer_called = false;

  g_succeed_then_fail_register_call_count = 0;
  std::vector<starpu_data_handle_t> registered_handles_before_cleanup;
  g_pre_cleanup_registered_handles = &registered_handles_before_cleanup;

  const auto previous_hook =
      starpu_server::testing::set_starpu_vector_register_hook_for_tests(
          &succeed_then_fail_starpu_vector_register);
  g_succeed_then_fail_previous_hook = previous_hook;
  const auto previous_observer =
      starpu_server::testing::set_starpu_register_failure_observer_for_tests(
          &capture_slot_state);

  int call_count_at_failure = 0;
  auto restore_hooks = [&]() {
    call_count_at_failure = g_succeed_then_fail_register_call_count;
    g_pre_cleanup_registered_handles = nullptr;
    g_succeed_then_fail_previous_hook = nullptr;
    g_succeed_then_fail_register_call_count = 0;
    starpu_server::testing::set_starpu_register_failure_observer_for_tests(
        previous_observer);
    starpu_server::testing::set_starpu_vector_register_hook_for_tests(
        previous_hook);
  };

  EXPECT_THROW(
      {
        try {
          starpu_server::InputSlotPool pool(opts, 1);
          restore_hooks();
          FAIL() << "Expected StarPU handle registration failure";
        }
        catch (...) {
          restore_hooks();
          throw;
        }
      },
      std::runtime_error);

  ASSERT_EQ(call_count_at_failure, 2);
  ASSERT_TRUE(g_failure_observer_called);
  ASSERT_EQ(g_observed_base_ptrs.size(), model.inputs.size());
  ASSERT_EQ(g_observed_handles.size(), model.inputs.size());

  ASSERT_EQ(registered_handles_before_cleanup.size(), 1);
  EXPECT_NE(registered_handles_before_cleanup.front(), nullptr);

  ASSERT_FALSE(g_observed_base_ptrs.empty());
  ASSERT_FALSE(g_observed_handles.empty());
  EXPECT_EQ(g_observed_base_ptrs.front(), nullptr);
  EXPECT_EQ(g_observed_handles.front(), nullptr);

  for (std::byte* base_ptr : g_observed_base_ptrs) {
    EXPECT_EQ(base_ptr, nullptr);
  }

  for (auto handle : g_observed_handles) {
    EXPECT_EQ(handle, nullptr);
  }
}

TEST(OutputSlotPool_Unit, RegisterFailureResetsSlotState)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 1;
  opts.batching.input_slots = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "failing_output";
  tensor.dims = {1, 1};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "failing_output_model";
  model.outputs.push_back(tensor);
  opts.models.push_back(model);

  g_output_observed_base_ptrs.clear();
  g_output_observed_handles.clear();
  g_output_observed_host_buffer_infos.clear();
  g_output_failure_observer_called = false;

  const auto previous_hook =
      starpu_server::testing::set_output_starpu_vector_register_hook_for_tests(
          &failing_starpu_vector_register);
  const auto previous_observer =
      starpu_server::testing::set_output_register_failure_observer_for_tests(
          &capture_output_slot_state);

  auto restore_hooks = [&]() {
    starpu_server::testing::set_output_register_failure_observer_for_tests(
        previous_observer);
    starpu_server::testing::set_output_starpu_vector_register_hook_for_tests(
        previous_hook);
  };

  EXPECT_THROW(
      {
        try {
          starpu_server::OutputSlotPool pool(opts, 1);
          restore_hooks();
          FAIL() << "Expected StarPU handle registration failure";
        }
        catch (...) {
          restore_hooks();
          throw;
        }
      },
      std::runtime_error);

  ASSERT_TRUE(g_output_failure_observer_called);
  ASSERT_EQ(g_output_observed_base_ptrs.size(), model.outputs.size());
  ASSERT_EQ(g_output_observed_handles.size(), model.outputs.size());
  ASSERT_EQ(g_output_observed_host_buffer_infos.size(), model.outputs.size());

  for (std::byte* base_ptr : g_output_observed_base_ptrs) {
    EXPECT_EQ(base_ptr, nullptr);
  }

  for (auto handle : g_output_observed_handles) {
    EXPECT_EQ(handle, nullptr);
  }

  for (const auto& info : g_output_observed_host_buffer_infos) {
    EXPECT_FALSE(info.cuda_pinned);
    EXPECT_FALSE(info.starpu_pinned);
    EXPECT_EQ(info.starpu_pin_rc, 0);
    EXPECT_EQ(info.bytes, 0U);
  }
}

TEST(OutputSlotPool_Unit, AllocateSlotBuffersOverflowThrows)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = std::numeric_limits<int>::max();
  opts.batching.input_slots = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "large_output";
  tensor.dims = {1, 65536, 65536};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "overflow_model";
  model.outputs.push_back(tensor);
  opts.models.push_back(model);

  EXPECT_THROW(
      { starpu_server::OutputSlotPool pool(opts, 1); }, std::overflow_error);
}

TEST(OutputSlotPool_Unit, ConstructionWithoutModelsThrows)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;

  EXPECT_THROW(
      { starpu_server::OutputSlotPool pool(opts, 1); }, std::invalid_argument);
}

TEST(OutputSlotPool_Unit, NonPositiveBatchDimensionThrows)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "invalid_batch_output";
  tensor.dims = {0, 1};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "invalid_batch_model";
  model.outputs.push_back(tensor);
  opts.models.push_back(model);

  EXPECT_THROW(
      {
        try {
          starpu_server::OutputSlotPool pool(opts, 1);
        }
        catch (const std::invalid_argument& ex) {
          EXPECT_STREQ("dims[0] (batch) must be positive", ex.what());
          throw;
        }
      },
      std::invalid_argument);
}

TEST(OutputSlotPool_Unit, NonBatchDimensionNonPositiveThrows)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 5;

  starpu_server::TensorConfig tensor;
  tensor.name = "non_positive_dims_output";
  tensor.dims = {1, 1, 0};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "non_positive_dims_model";
  model.outputs.push_back(tensor);
  opts.models.push_back(model);

  EXPECT_THROW(
      {
        try {
          starpu_server::OutputSlotPool pool(opts, 1);
        }
        catch (const std::invalid_argument& ex) {
          EXPECT_STREQ("dimensions must be positive", ex.what());
          throw;
        }
      },
      std::invalid_argument);
}

TEST(OutputSlotPool_Unit, BatchDimensionExceedsIntMaxThrows)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "exceeds_int_max_output";
  tensor.dims = {static_cast<int64_t>(std::numeric_limits<int>::max()) + 1, 1};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "exceeds_int_max_model";
  model.outputs.push_back(tensor);
  opts.models.push_back(model);

  EXPECT_THROW(
      {
        try {
          starpu_server::OutputSlotPool pool(opts, 1);
        }
        catch (const std::invalid_argument& ex) {
          EXPECT_STREQ("dims[0] (batch) exceeds int max", ex.what());
          throw;
        }
      },
      std::invalid_argument);
}

TEST(OutputSlotPool_Unit, NonBatchDimensionOverflowThrows)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 5;

  starpu_server::TensorConfig tensor;
  tensor.name = "dimension_product_overflow_output";
  tensor.dims = {
      1, static_cast<int64_t>(1ULL << 32), static_cast<int64_t>(1ULL << 32)};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "dimension_product_overflow_model";
  model.outputs.push_back(tensor);
  opts.models.push_back(model);

  EXPECT_THROW(
      {
        try {
          starpu_server::OutputSlotPool pool(opts, 1);
        }
        catch (const std::overflow_error& ex) {
          EXPECT_STREQ("dimension product overflow", ex.what());
          throw;
        }
      },
      std::overflow_error);
}

TEST(OutputSlotPool_Unit, ReleaseReturnsSlotToPool)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "simple_output";
  tensor.dims = {1, 1};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "simple_model";
  model.outputs.push_back(tensor);
  opts.models.push_back(model);

  starpu_server::OutputSlotPool pool(opts, 1);

  const int slot_id = pool.acquire();
  EXPECT_GE(slot_id, 0);

  EXPECT_FALSE(pool.try_acquire().has_value());

  pool.release(slot_id);

  auto maybe_slot = pool.try_acquire();
  ASSERT_TRUE(maybe_slot.has_value());
  EXPECT_EQ(*maybe_slot, slot_id);

  pool.release(*maybe_slot);
}

TEST(OutputSlotPool_Unit, SlotInfoProvidesConsistentReferences)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "minimal_output";
  tensor.dims = {1, 1};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "minimal_model";
  model.outputs.push_back(tensor);
  opts.models.push_back(model);

  starpu_server::OutputSlotPool pool(opts, 1);

  const int slot_id = pool.acquire();
  const auto& info = pool.slot_info(slot_id);

  EXPECT_EQ(info.id, slot_id);
  ASSERT_EQ(info.base_ptrs.size(), model.outputs.size());
  ASSERT_EQ(info.handles.size(), model.outputs.size());

  const auto& base_ptrs_ref = pool.base_ptrs(slot_id);
  const auto& handles_ref = pool.handles(slot_id);

  EXPECT_EQ(base_ptrs_ref.size(), info.base_ptrs.size());
  EXPECT_EQ(handles_ref.size(), info.handles.size());

  EXPECT_EQ(
      static_cast<const void*>(&base_ptrs_ref),
      static_cast<const void*>(&info.base_ptrs));
  EXPECT_EQ(
      static_cast<const void*>(&handles_ref),
      static_cast<const void*>(&info.handles));

  EXPECT_THROW(
      static_cast<void>(pool.slot_info(slot_id + 1)), std::out_of_range);
  EXPECT_THROW(
      static_cast<void>(pool.base_ptrs(slot_id + 1)), std::out_of_range);
  EXPECT_THROW(static_cast<void>(pool.handles(slot_id + 1)), std::out_of_range);

  pool.release(slot_id);
}

TEST(OutputSlotPool_Unit, CleanupSlotBuffersReleasesResources)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::OutputSlotPool::SlotInfo slot;
  slot.handles.resize(1);
  slot.base_ptrs.resize(1);

  auto* raw_ptr = std::malloc(sizeof(int));
  ASSERT_NE(raw_ptr, nullptr);
  slot.base_ptrs[0] = static_cast<std::byte*>(raw_ptr);

  std::vector<starpu_server::OutputSlotPool::HostBufferInfo> buffer_infos(1);
  buffer_infos[0].bytes = sizeof(int);

  starpu_data_handle_t handle = nullptr;
  starpu_variable_data_register(
      &handle, STARPU_MAIN_RAM, reinterpret_cast<uintptr_t>(raw_ptr),
      sizeof(int));
  ASSERT_NE(handle, nullptr);
  slot.handles[0] = handle;

  starpu_server::OutputSlotPoolTestHook::cleanup_slot_buffers(
      slot, buffer_infos, buffer_infos.size());

  EXPECT_EQ(slot.handles[0], nullptr);
  EXPECT_EQ(slot.base_ptrs[0], nullptr);
  EXPECT_FALSE(buffer_infos[0].cuda_pinned);
  EXPECT_FALSE(buffer_infos[0].starpu_pinned);
  EXPECT_EQ(buffer_infos[0].starpu_pin_rc, 0);
  EXPECT_EQ(buffer_infos[0].bytes, 0U);
}

TEST(OutputSlotPool_Unit, DefaultSlotCountUsesWorkerCount)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.batching.max_batch_size = 1;

  starpu_server::TensorConfig tensor;
  tensor.name = "single_output";
  tensor.dims = {1, 1};
  tensor.type = at::ScalarType::Float;

  starpu_server::ModelConfig model;
  model.name = "single_model";
  model.outputs.push_back(tensor);
  opts.models.push_back(model);

  const int expected_slots =
      std::max(2, static_cast<int>(starpu_worker_get_count()));

  starpu_server::OutputSlotPool pool(opts, 0);

  std::vector<int> acquired_ids;
  acquired_ids.reserve(static_cast<size_t>(expected_slots));

  for (int i = 0; i < expected_slots; ++i) {
    auto maybe_slot = pool.try_acquire();
    ASSERT_TRUE(maybe_slot.has_value())
        << "Expected to acquire slot " << i << " of " << expected_slots;
    acquired_ids.push_back(*maybe_slot);
  }

  EXPECT_FALSE(pool.try_acquire().has_value());

  std::unordered_set<int> unique_ids(acquired_ids.begin(), acquired_ids.end());
  EXPECT_EQ(unique_ids.size(), acquired_ids.size());
  EXPECT_EQ(acquired_ids.size(), static_cast<size_t>(expected_slots));

  for (int slot_id : acquired_ids) {
    pool.release(slot_id);
  }
}

TEST(OutputSlotPool_Unit, CleanupSlotBuffersUnpinsStarpuMemory)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::OutputSlotPool::SlotInfo slot;
  slot.handles.resize(1);
  slot.base_ptrs.resize(1);

  auto* raw_ptr = std::malloc(sizeof(int));
  ASSERT_NE(raw_ptr, nullptr);
  slot.base_ptrs[0] = static_cast<std::byte*>(raw_ptr);

  ASSERT_EQ(starpu_memory_pin(raw_ptr, sizeof(int)), 0);

  std::vector<starpu_server::OutputSlotPool::HostBufferInfo> buffer_infos(1);
  buffer_infos[0].bytes = sizeof(int);
  buffer_infos[0].starpu_pinned = true;
  buffer_infos[0].starpu_pin_rc = 0;

  starpu_data_handle_t handle = nullptr;
  starpu_variable_data_register(
      &handle, STARPU_MAIN_RAM, reinterpret_cast<uintptr_t>(raw_ptr),
      sizeof(int));
  ASSERT_NE(handle, nullptr);
  slot.handles[0] = handle;

  // Failures in starpu_memory_unpin are reported via warnings; this test
  // exercises the successful cleanup path.
  starpu_server::OutputSlotPoolTestHook::cleanup_slot_buffers(
      slot, buffer_infos, buffer_infos.size());

  EXPECT_EQ(slot.handles[0], nullptr);
  EXPECT_EQ(slot.base_ptrs[0], nullptr);
  EXPECT_FALSE(buffer_infos[0].cuda_pinned);
  EXPECT_FALSE(buffer_infos[0].starpu_pinned);
  EXPECT_EQ(buffer_infos[0].starpu_pin_rc, 0);
  EXPECT_EQ(buffer_infos[0].bytes, 0U);
}

TEST(OutputSlotPool_Unit, CleanupSlotBuffersFreesCudaPinnedMemory)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::OutputSlotPool::SlotInfo slot;
  slot.handles.resize(1);
  slot.base_ptrs.resize(1);

  void* raw_ptr = nullptr;
  cudaError_t alloc_rc =
      cudaHostAlloc(&raw_ptr, sizeof(int), cudaHostAllocPortable);
  if (alloc_rc != cudaSuccess) {
    GTEST_SKIP() << "cudaHostAlloc not supported: rc="
                 << static_cast<int>(alloc_rc);
  }
  ASSERT_NE(raw_ptr, nullptr);
  slot.base_ptrs[0] = static_cast<std::byte*>(raw_ptr);

  std::vector<starpu_server::OutputSlotPool::HostBufferInfo> buffer_infos(1);
  buffer_infos[0].bytes = sizeof(int);
  buffer_infos[0].cuda_pinned = true;

  starpu_data_handle_t handle = nullptr;
  starpu_variable_data_register(
      &handle, STARPU_MAIN_RAM, reinterpret_cast<uintptr_t>(raw_ptr),
      sizeof(int));
  ASSERT_NE(handle, nullptr);
  slot.handles[0] = handle;

  // Failures in cudaFreeHost are reported via warnings; this test exercises the
  // successful cleanup path.
  starpu_server::OutputSlotPoolTestHook::cleanup_slot_buffers(
      slot, buffer_infos, buffer_infos.size());

  EXPECT_EQ(slot.handles[0], nullptr);
  EXPECT_EQ(slot.base_ptrs[0], nullptr);
  EXPECT_FALSE(buffer_infos[0].cuda_pinned);
  EXPECT_FALSE(buffer_infos[0].starpu_pinned);
  EXPECT_EQ(buffer_infos[0].starpu_pin_rc, 0);
  EXPECT_EQ(buffer_infos[0].bytes, 0U);
}
