#pragma once

#include <atomic>
#include <string>
#include <string_view>

// NOLINTNEXTLINE(cppcoreguidelines-macro-to-enum,modernize-macro-to-enum,cppcoreguidelines-macro-usage)
#define STARPU_SERVER_NVTX_TESTING 1

namespace testing_nvtx {

inline auto
PushCounter() -> std::atomic<int>&
{
  static std::atomic<int> counter{0};
  return counter;
}

inline auto
PopCounter() -> std::atomic<int>&
{
  static std::atomic<int> counter{0};
  return counter;
}

inline auto
Reset() -> void
{
  PushCounter().store(0);
  PopCounter().store(0);
}

inline auto
PushCount() -> int
{
  return PushCounter().load();
}

inline auto
PopCount() -> int
{
  return PopCounter().load();
}

inline auto
TrackPush() -> void
{
  PushCounter().fetch_add(1);
}

inline auto
TrackPop() -> void
{
  PopCounter().fetch_add(1);
}

inline auto
TrackPushHook(std::string_view /*unused*/) -> void
{
  TrackPush();
}

inline auto
TrackPopHook() -> void
{
  TrackPop();
}

}  // namespace testing_nvtx

inline auto
nvtxRangePushA([[maybe_unused]] const char* label) -> int
{
  testing_nvtx::TrackPush();
  return 0;
}

inline auto
nvtxRangePop() -> int
{
  testing_nvtx::TrackPop();
  return 0;
}
