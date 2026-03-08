#include <gtest/gtest.h>

#include <cstddef>

#include "core/slot_pool_base.hpp"

namespace {

class TestSlotPool final : public starpu_server::SlotPoolBase<> {
 public:
  explicit TestSlotPool(int slot_count)
  {
    auto& slots_storage = slots();
    auto& free_ids_storage = free_ids();
    slots_storage.resize(static_cast<std::size_t>(slot_count));
    free_ids_storage.reserve(static_cast<std::size_t>(slot_count));
    for (int i = 0; i < slot_count; ++i) {
      slots_storage[static_cast<std::size_t>(i)].id = i;
      free_ids_storage.push_back(i);
    }
  }

  using SlotPoolBase<>::acquire;
  using SlotPoolBase<>::release;
  using SlotPoolBase<>::try_acquire;
};

TEST(SlotPoolBase, TryAcquireReleaseRoundTrip)
{
  TestSlotPool pool(2);
  auto slot = pool.try_acquire();
  ASSERT_TRUE(slot.has_value());
  pool.release(*slot);

  auto reacquired = pool.try_acquire();
  ASSERT_TRUE(reacquired.has_value());
  EXPECT_EQ(*reacquired, *slot);
  pool.release(*reacquired);
}

TEST(SlotPoolBase, ReleaseRejectsOutOfBoundsSlotIdInDebug)
{
#if defined(NDEBUG)
  GTEST_SKIP() << "Debug assertions are disabled (NDEBUG defined)";
#else
  TestSlotPool pool(1);
  EXPECT_DEATH({ pool.release(1); }, "SlotPoolBase detected invalid slot id");
#endif
}

TEST(SlotPoolBase, ReleaseRejectsDoubleReleaseInDebug)
{
#if defined(NDEBUG)
  GTEST_SKIP() << "Debug assertions are disabled (NDEBUG defined)";
#else
  TestSlotPool pool(1);
  const int slot = pool.acquire();
  pool.release(slot);
  EXPECT_DEATH(
      { pool.release(slot); },
      "SlotPoolBase::release detected double release or never-acquired slot");
#endif
}

}  // namespace
