#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <starpu.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/input_slot_pool.hpp"
#include "core/output_slot_pool.hpp"
#include "core/starpu_setup.hpp"
#include "test_utils.hpp"
#include "utils/runtime_config.hpp"

TEST(StarPUSetup_Unit, DuplicateDeviceIdsThrows)
{
  starpu_server::RuntimeConfig opts;
  opts.use_cuda = true;
  opts.device_ids = {0, 0};
  EXPECT_THROW(
      { starpu_server::StarPUSetup setup(opts); }, std::invalid_argument);
}

TEST(InputSlotPool_Unit, AllocateSlotBuffersOverflowThrows)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.max_batch_size = std::numeric_limits<int>::max();
  opts.input_slots = 1;

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
  opts.max_batch_size = 5;
  opts.input_slots = 1;

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
  opts.max_batch_size = 1;

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
  opts.max_batch_size = 1;

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
  opts.max_batch_size = 5;
  opts.input_slots = 1;

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
  opts.max_batch_size = 5;
  opts.input_slots = 1;

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
  opts.max_batch_size = 1;

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
  opts.max_batch_size = 1;
  opts.input_slots = 1;

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
  opts.use_cuda = true;
  opts.max_batch_size = 1;

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

TEST(OutputSlotPool_Unit, AllocateSlotBuffersOverflowThrows)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.max_batch_size = std::numeric_limits<int>::max();
  opts.input_slots = 1;

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
  opts.max_batch_size = 1;

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

TEST(OutputSlotPool_Unit, BatchDimensionExceedsIntMaxThrows)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.max_batch_size = 1;

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

TEST(OutputSlotPool_Unit, ReleaseReturnsSlotToPool)
{
  StarpuRuntimeGuard starpu_guard;

  starpu_server::RuntimeConfig opts;
  opts.max_batch_size = 1;

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
  opts.max_batch_size = 1;

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
  slot.base_ptrs[0] = raw_ptr;

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
  opts.max_batch_size = 1;

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
  slot.base_ptrs[0] = raw_ptr;

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
  slot.base_ptrs[0] = raw_ptr;

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
