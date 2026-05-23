#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <iostream>
#include <new>
#include <sstream>
#include <string>

static int g_starpu_memory_pin_rc = 0;
static int g_starpu_memory_pin_calls = 0;
static int g_starpu_memory_unpin_rc = 0;
static int g_starpu_memory_unpin_calls = 0;

namespace {
class StreamCapture {
 public:
  explicit StreamCapture(std::ostream& stream)
      : stream_{stream}, old_buf_{stream.rdbuf(buffer_.rdbuf())}
  {
  }
  ~StreamCapture() { stream_.rdbuf(old_buf_); }
  [[nodiscard]] auto str() const -> std::string { return buffer_.str(); }

 private:
  std::ostream& stream_;
  std::ostringstream buffer_;
  std::streambuf* old_buf_;
};
}  // namespace

extern "C" cudaError_t
cudaHostAlloc_test(void** ptr, size_t /*size*/, unsigned int /*flags*/)
{
  if (ptr != nullptr) {
    *ptr = nullptr;
  }
  return cudaErrorMemoryAllocation;
}

extern "C" int
starpu_memory_pin_test(void* /*ptr*/, size_t /*bytes*/)
{
  ++g_starpu_memory_pin_calls;
  return g_starpu_memory_pin_rc;
}

extern "C" int
starpu_memory_unpin_test(void* /*ptr*/, size_t /*bytes*/)
{
  ++g_starpu_memory_unpin_calls;
  return g_starpu_memory_unpin_rc;
}

#define cudaHostAlloc cudaHostAlloc_test
#define starpu_memory_pin starpu_memory_pin_test
#define starpu_memory_unpin starpu_memory_unpin_test
#define starpu_server starpu_server_input_slot_pool_test_copy
#include "core/input_slot_pool.cpp"
#undef starpu_server
#undef starpu_memory_unpin
#undef starpu_memory_pin
#undef cudaHostAlloc

namespace input_slot_pool_test_copy = starpu_server_input_slot_pool_test_copy;

TEST(InputSlotPoolHostBufferDeleter, NullptrNoop)
{
  input_slot_pool_test_copy::HostBufferDeleter deleter;
  deleter.info.cuda_pinned = true;
  deleter.info.starpu_pinned = true;
  deleter.info.bytes = 1024;

  EXPECT_NO_THROW(deleter(nullptr));
}

TEST(InputSlotPoolFreeHostBuffer, NullptrNoop)
{
  input_slot_pool_test_copy::InputSlotPool::HostBufferInfo info{};
  info.cuda_pinned = true;
  info.starpu_pinned = true;
  info.bytes = 1024;

  EXPECT_NO_THROW(input_slot_pool_test_copy::free_host_buffer(nullptr, info));
}

TEST(InputSlotPoolHostBufferDeleter, LogsWarningOnUnpinFailure)
{
  g_starpu_memory_unpin_rc = -42;
  g_starpu_memory_unpin_calls = 0;

  input_slot_pool_test_copy::InputSlotPool::HostBufferInfo info{};
  info.starpu_pinned = true;
  info.bytes = 64;

  auto* ptr =
      static_cast<std::byte*>(::operator new(info.bytes, std::align_val_t{64}));

  std::string log;
  {
    StreamCapture capture{std::cerr};
    input_slot_pool_test_copy::free_host_buffer(ptr, info);
    log = capture.str();
  }

  EXPECT_EQ(g_starpu_memory_unpin_calls, 1);
  EXPECT_NE(
      log.find("starpu_memory_unpin failed for input buffer"),
      std::string::npos);
}

TEST(AllocateAndPinBuffer, MarksStarpuPinnedOnSuccess)
{
  g_starpu_memory_pin_rc = 0;
  g_starpu_memory_pin_calls = 0;
  g_starpu_memory_unpin_rc = 0;
  g_starpu_memory_unpin_calls = 0;

  auto allocation = input_slot_pool_test_copy::allocate_and_pin_buffer(
      /*bytes=*/64, /*want_pinned=*/true, /*slot_id=*/2, /*input_index=*/1);

  EXPECT_EQ(g_starpu_memory_pin_calls, 1);
  EXPECT_TRUE(allocation.info.starpu_pinned);
  EXPECT_EQ(allocation.info.starpu_pin_rc, 0);
  EXPECT_FALSE(allocation.info.cuda_pinned);
  EXPECT_EQ(allocation.info.bytes, 64U);
  ASSERT_NE(allocation.ptr, nullptr);

  input_slot_pool_test_copy::free_host_buffer(allocation.ptr, allocation.info);
  EXPECT_EQ(g_starpu_memory_unpin_calls, 1);
}

TEST(AllocateAndPinBuffer, LeavesStarpuUnpinnedOnFailure)
{
  g_starpu_memory_pin_rc = -7;
  g_starpu_memory_pin_calls = 0;

  std::string log;
  {
    StreamCapture capture{std::cerr};
    auto allocation = input_slot_pool_test_copy::allocate_and_pin_buffer(
        /*bytes=*/128, /*want_pinned=*/true, /*slot_id=*/3,
        /*input_index=*/0);

    EXPECT_EQ(g_starpu_memory_pin_calls, 1);
    EXPECT_FALSE(allocation.info.starpu_pinned);
    EXPECT_EQ(allocation.info.starpu_pin_rc, -7);
    EXPECT_FALSE(allocation.info.cuda_pinned);
    EXPECT_EQ(allocation.info.bytes, 128U);
    ASSERT_NE(allocation.ptr, nullptr);

    input_slot_pool_test_copy::free_host_buffer(
        allocation.ptr, allocation.info);
    log = capture.str();
  }

  EXPECT_NE(
      log.find("starpu_memory_pin failed for input slot 3, index 0"),
      std::string::npos);
}
