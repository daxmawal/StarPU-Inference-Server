#pragma once

#include <string>

#ifdef HAVE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace starpu_server {

namespace detail {

using NvtxPushHook = void (*)(const std::string&);
using NvtxPopHook = void (*)();

inline void
DefaultPushHook(const std::string& label)
{
#ifdef HAVE_NVTX
  nvtxRangePushA(label.c_str());
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
InvokePush(const std::string& label)
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

  explicit NvtxRange(const char* name) : label_(name)
  {
    detail::InvokePush(label_);
  }

  explicit NvtxRange(std::string name) : label_(std::move(name))
  {
    detail::InvokePush(label_);
  }

  ~NvtxRange() { detail::InvokePop(); }

  static void SetHooks(PushHook push, PopHook pop)
  {
    detail::SetHooks(push, pop);
  }

  static void ResetHooks() { detail::SetHooks(nullptr, nullptr); }

  NvtxRange(const NvtxRange&) = delete;
  auto operator=(const NvtxRange&) -> NvtxRange& = delete;

 private:
  std::string label_{};
};

}  // namespace starpu_server
