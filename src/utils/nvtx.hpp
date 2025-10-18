#pragma once

#include <functional>
#include <string>
#include <string_view>
#include <utility>

#ifdef HAVE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace starpu_server {

namespace detail {

using NvtxPushHook = std::function<void(std::string_view)>;
using NvtxPopHook = std::function<void()>;

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
  PushHookRef() = push ? std::move(push) : NvtxPushHook{&DefaultPushHook};
  PopHookRef() = pop ? std::move(pop) : NvtxPopHook{&DefaultPopHook};
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
  using PushHook = std::function<void(std::string_view)>;
  using PopHook = std::function<void()>;

  explicit NvtxRange(const char* name) { detail::InvokePush(name); }

  explicit NvtxRange(const std::string& name) { detail::InvokePush(name); }

  ~NvtxRange() { detail::InvokePop(); }

  static void SetHooks(PushHook push, PopHook pop)
  {
    detail::SetHooks(std::move(push), std::move(pop));
  }

  static void ResetHooks() { detail::SetHooks({}, {}); }

  NvtxRange(const NvtxRange&) = delete;
  auto operator=(const NvtxRange&) -> NvtxRange& = delete;
};

}  // namespace starpu_server
