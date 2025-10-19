#pragma once

#include <atomic>
#include <string>
#include <string_view>

#define STARPU_SERVER_NVTX_TESTING \
  1  // NOLINT(cppcoreguidelines-macro-to-enum,modernize-macro-to-enum,cppcoreguidelines-macro-usage)

namespace testing_nvtx {

inline std::atomic<int> push_count{
    0};  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
inline std::atomic<int> pop_count{
    0};  // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

inline auto
Reset() -> void
{
  push_count.store(0);
  pop_count.store(0);
}

inline auto
PushCount() -> int
{
  return push_count.load();
}

inline auto
PopCount() -> int
{
  return pop_count.load();
}

inline auto
TrackPush() -> void
{
  push_count.fetch_add(1);
}

inline auto
TrackPop() -> void
{
  pop_count.fetch_add(1);
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
