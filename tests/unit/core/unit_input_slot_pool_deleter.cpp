#include <gtest/gtest.h>

#define starpu_server starpu_server_input_slot_pool_test_copy
#include "core/input_slot_pool.cpp"
#undef starpu_server

namespace input_slot_pool_test_copy = starpu_server_input_slot_pool_test_copy;

TEST(InputSlotPoolHostBufferDeleter, NullptrNoop)
{
  input_slot_pool_test_copy::HostBufferDeleter deleter;
  deleter.info.cuda_pinned = true;
  deleter.info.starpu_pinned = true;
  deleter.info.bytes = 1024;

  EXPECT_NO_THROW(deleter(nullptr));
}
