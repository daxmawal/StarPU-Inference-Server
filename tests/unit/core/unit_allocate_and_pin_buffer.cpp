#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <new>

#include "core/input_slot_pool.hpp"

int g_test_mock_starpu_pin_rc = 0;
bool g_test_mock_cuda_pinned = false;
std::byte* g_test_mock_allocated_ptr = nullptr;
constexpr size_t kMockBufferSize = 65536;
static std::byte g_test_mock_buffer[kMockBufferSize];

namespace starpu_server_allocate_pin_test {

struct AllocatedHostBuffer {
  std::byte* ptr = nullptr;
  starpu_server::InputSlotPool::HostBufferInfo info;
};

auto
alloc_host_buffer(size_t bytes, bool use_pinned, bool& cuda_pinned_out)
    -> std::byte*
{
  (void)use_pinned;
  (void)bytes;
  cuda_pinned_out = g_test_mock_cuda_pinned;
  g_test_mock_allocated_ptr = g_test_mock_buffer;
  return g_test_mock_allocated_ptr;
}

auto
allocate_and_pin_buffer(
    std::size_t bytes, bool want_pinned, int slot_id,
    std::size_t input_index) -> AllocatedHostBuffer
{
  bool cuda_pinned = false;
  std::byte* ptr = alloc_host_buffer(bytes, want_pinned, cuda_pinned);

  AllocatedHostBuffer allocation;
  allocation.ptr = ptr;
  allocation.info.cuda_pinned = cuda_pinned;
  allocation.info.bytes = bytes;

  const bool should_starpu_pin = want_pinned && !cuda_pinned;
  if (should_starpu_pin) {
    const int pin_result = starpu_memory_pin(static_cast<void*>(ptr), bytes);
    allocation.info.starpu_pin_rc = pin_result;
    if (pin_result == 0) {
      allocation.info.starpu_pinned = true;
    } else {
      (void)slot_id;
      (void)input_index;
    }
  }

  return allocation;
}

}  // namespace starpu_server_allocate_pin_test

extern "C" {
int
starpu_memory_pin(void* ptr, size_t bytes)
{
  (void)ptr;
  (void)bytes;
  return g_test_mock_starpu_pin_rc;
}
}

namespace allocate_pin_test = starpu_server_allocate_pin_test;

class AllocateAndPinBufferTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    g_test_mock_starpu_pin_rc = 0;
    g_test_mock_cuda_pinned = false;
    g_test_mock_allocated_ptr = nullptr;
  }
};

TEST_F(AllocateAndPinBufferTest, SkipsPinningWhenCudaPinned)
{
  g_test_mock_cuda_pinned = true;
  g_test_mock_starpu_pin_rc = 1;
  const size_t test_size = 4096;

  auto result =
      allocate_pin_test::allocate_and_pin_buffer(test_size, true, 0, 0);

  EXPECT_TRUE(result.info.cuda_pinned);
  EXPECT_FALSE(result.info.starpu_pinned);
  EXPECT_EQ(result.info.bytes, test_size);
  EXPECT_EQ(result.info.starpu_pin_rc, 0);
}

TEST_F(AllocateAndPinBufferTest, SkipsPinningWhenNotWanted)
{
  g_test_mock_cuda_pinned = false;
  g_test_mock_starpu_pin_rc = 1;
  const size_t test_size = 2048;

  auto result =
      allocate_pin_test::allocate_and_pin_buffer(test_size, false, 1, 0);

  EXPECT_FALSE(result.info.cuda_pinned);
  EXPECT_FALSE(result.info.starpu_pinned);
  EXPECT_EQ(result.info.bytes, test_size);
  EXPECT_EQ(result.info.starpu_pin_rc, 0);
}

TEST_F(AllocateAndPinBufferTest, SuccessfulStarPUPinning)
{
  g_test_mock_cuda_pinned = false;
  g_test_mock_starpu_pin_rc = 0;
  const size_t test_size = 1024;

  auto result =
      allocate_pin_test::allocate_and_pin_buffer(test_size, true, 2, 0);

  EXPECT_FALSE(result.info.cuda_pinned);
  EXPECT_TRUE(result.info.starpu_pinned);
  EXPECT_EQ(result.info.bytes, test_size);
  EXPECT_EQ(result.info.starpu_pin_rc, 0);
}

TEST_F(AllocateAndPinBufferTest, FailedStarPUPinningWithErrorCode1)
{
  g_test_mock_cuda_pinned = false;
  g_test_mock_starpu_pin_rc = 1;
  const size_t test_size = 512;

  auto result =
      allocate_pin_test::allocate_and_pin_buffer(test_size, true, 3, 1);

  EXPECT_FALSE(result.info.cuda_pinned);
  EXPECT_FALSE(result.info.starpu_pinned);
  EXPECT_EQ(result.info.bytes, test_size);
  EXPECT_EQ(result.info.starpu_pin_rc, 1);
}

TEST_F(AllocateAndPinBufferTest, FailedStarPUPinningWithNegativeErrorCode)
{
  g_test_mock_cuda_pinned = false;
  g_test_mock_starpu_pin_rc = -1;
  const size_t test_size = 256;

  auto result =
      allocate_pin_test::allocate_and_pin_buffer(test_size, true, 4, 2);

  EXPECT_FALSE(result.info.cuda_pinned);
  EXPECT_FALSE(result.info.starpu_pinned);
  EXPECT_EQ(result.info.bytes, test_size);
  EXPECT_EQ(result.info.starpu_pin_rc, -1);
}

TEST_F(AllocateAndPinBufferTest, PreservesBufferPointer)
{
  g_test_mock_cuda_pinned = false;
  g_test_mock_starpu_pin_rc = 0;
  const size_t test_size = 512;

  auto result =
      allocate_pin_test::allocate_and_pin_buffer(test_size, true, 5, 0);

  EXPECT_EQ(result.ptr, g_test_mock_allocated_ptr);
  EXPECT_NE(result.ptr, nullptr);
}

TEST_F(AllocateAndPinBufferTest, PreservesBufferMetadata)
{
  g_test_mock_cuda_pinned = true;
  g_test_mock_starpu_pin_rc = 0;
  const size_t test_size = 8192;

  auto result =
      allocate_pin_test::allocate_and_pin_buffer(test_size, true, 6, 0);

  EXPECT_EQ(result.info.bytes, test_size);
  EXPECT_TRUE(result.info.cuda_pinned);
  EXPECT_FALSE(result.info.starpu_pinned);
  EXPECT_NE(result.ptr, nullptr);
}
