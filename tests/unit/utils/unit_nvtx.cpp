#define HAVE_NVTX 1

#include <gtest/gtest.h>

#include <string>

#include "nvtx3/nvToolsExt.h"
#include "utils/nvtx.hpp"

using namespace starpu_server;

namespace {

void
TestPushHook(const std::string&)
{
  testing_nvtx::TrackPush();
}

void
TestPopHook()
{
  testing_nvtx::TrackPop();
}

class NvtxHookGuard {
 public:
  NvtxHookGuard() { NvtxRange::SetHooks(&TestPushHook, &TestPopHook); }

  ~NvtxHookGuard() { NvtxRange::ResetHooks(); }
};

}  // namespace

TEST(NvtxRange, TracksPushPopWithCString)
{
  NvtxHookGuard guard{};
  testing_nvtx::Reset();
  {
    NvtxRange range{"c-string label"};
    EXPECT_EQ(testing_nvtx::PushCount(), 1);
    EXPECT_EQ(testing_nvtx::PopCount(), 0);
  }
  EXPECT_EQ(testing_nvtx::PushCount(), 1);
  EXPECT_EQ(testing_nvtx::PopCount(), 1);
}

TEST(NvtxRange, DefaultHooksUseNvtxFunctions)
{
  NvtxRange::ResetHooks();
  testing_nvtx::Reset();

  {
    NvtxRange range{"default label"};
    EXPECT_EQ(testing_nvtx::PushCount(), 1);
    EXPECT_EQ(testing_nvtx::PopCount(), 0);
  }

  EXPECT_EQ(testing_nvtx::PushCount(), 1);
  EXPECT_EQ(testing_nvtx::PopCount(), 1);

  NvtxRange::ResetHooks();
}

TEST(NvtxRange, TracksPushPopWithString)
{
  NvtxHookGuard guard{};
  testing_nvtx::Reset();
  {
    std::string label{"string label"};
    NvtxRange range{label};
    EXPECT_EQ(testing_nvtx::PushCount(), 1);
    EXPECT_EQ(testing_nvtx::PopCount(), 0);
  }
  EXPECT_EQ(testing_nvtx::PushCount(), 1);
  EXPECT_EQ(testing_nvtx::PopCount(), 1);
}
