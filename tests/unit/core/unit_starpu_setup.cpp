#include <gtest/gtest.h>

#include <limits>

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
