#pragma once

#include <atomic>
#include <string>

#define STARPU_SERVER_NVTX_TESTING 1

namespace testing_nvtx {

inline std::atomic<int> push_count{0};
inline std::atomic<int> pop_count{0};

inline void
Reset()
{
  push_count.store(0);
  pop_count.store(0);
}

inline int
PushCount()
{
  return push_count.load();
}

inline int
PopCount()
{
  return pop_count.load();
}

inline void
TrackPush()
{
  push_count.fetch_add(1);
}

inline void
TrackPop()
{
  pop_count.fetch_add(1);
}

inline void
TrackPushHook(const std::string&)
{
  TrackPush();
}

inline void
TrackPopHook()
{
  TrackPop();
}

}  // namespace testing_nvtx

inline int
nvtxRangePushA(const char*)
{
  testing_nvtx::TrackPush();
  return 0;
}

inline int
nvtxRangePop()
{
  testing_nvtx::TrackPop();
  return 0;
}
