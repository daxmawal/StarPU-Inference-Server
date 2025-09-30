#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <limits>
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
