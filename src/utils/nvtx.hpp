#pragma once

#include <string>
#include <string_view>

#ifdef HAVE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace starpu_server {

namespace detail {

using NvtxPushHook = void (*)(std::string_view);
using NvtxPopHook = void (*)();

inline void
DefaultPushHook(std::string_view label)
{
#ifdef HAVE_NVTX
  const std::string materialized_label{label};
  nvtxRangePushA(materialized_label.c_str());
#else
  (void)label;
#endif
}

inline void
DefaultPopHook()
{
#ifdef HAVE_NVTX
  nvtxRangePop();
#endif
}

inline auto&
PushHookRef()
{
  static NvtxPushHook hook = &DefaultPushHook;
  return hook;
}

inline auto&
PopHookRef()
{
  static NvtxPopHook hook = &DefaultPopHook;
  return hook;
}

inline void
SetHooks(NvtxPushHook push, NvtxPopHook pop)
{
  PushHookRef() = push ? push : &DefaultPushHook;
  PopHookRef() = pop ? pop : &DefaultPopHook;
}

inline void
InvokePush(std::string_view label)
{
  PushHookRef()(label);
}

inline void
InvokePop()
{
  PopHookRef()();
}

}  // namespace detail

class NvtxRange {
 public:
  using PushHook = detail::NvtxPushHook;
  using PopHook = detail::NvtxPopHook;

  explicit NvtxRange(const char* name) { detail::InvokePush(name); }

  explicit NvtxRange(std::string name) { detail::InvokePush(name); }

  ~NvtxRange() { detail::InvokePop(); }

  static void SetHooks(PushHook push, PopHook pop)
  {
    detail::SetHooks(push, pop);
  }

  static void ResetHooks() { detail::SetHooks(nullptr, nullptr); }

  NvtxRange(const NvtxRange&) = delete;
  auto operator=(const NvtxRange&) -> NvtxRange& = delete;
};

}  // namespace starpu_server
