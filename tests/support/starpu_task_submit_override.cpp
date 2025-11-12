#include "support/starpu_task_submit_override.hpp"

#include <dlfcn.h>

#include <stdexcept>

namespace {

using starpu_test::StarpuDataAcquireOverrideFn;
using starpu_test::StarpuDataReleaseOverrideFn;
using starpu_test::StarpuTaskSubmitOverrideFn;
using starpu_test::StarpuVectorRegisterOverrideFn;

auto&
task_submit_override_ref()
{
  static StarpuTaskSubmitOverrideFn fn = nullptr;
  return fn;
}

auto
resolve_real_starpu_task_submit() -> StarpuTaskSubmitOverrideFn
{
  static StarpuTaskSubmitOverrideFn fn = [] {
    void* symbol = dlsym(RTLD_NEXT, "starpu_task_submit");
    if (symbol == nullptr) {
      throw std::runtime_error("Failed to resolve starpu_task_submit");
    }
    return reinterpret_cast<StarpuTaskSubmitOverrideFn>(symbol);
  }();
  return fn;
}

auto&
vector_register_override_ref()
{
  static StarpuVectorRegisterOverrideFn fn = nullptr;
  return fn;
}

auto
resolve_real_starpu_vector_data_register() -> StarpuVectorRegisterOverrideFn
{
  static StarpuVectorRegisterOverrideFn fn = [] {
    void* symbol = dlsym(RTLD_NEXT, "starpu_vector_data_register");
    if (symbol == nullptr) {
      throw std::runtime_error("Failed to resolve starpu_vector_data_register");
    }
    return reinterpret_cast<StarpuVectorRegisterOverrideFn>(symbol);
  }();
  return fn;
}

auto&
data_release_override_ref()
{
  static StarpuDataReleaseOverrideFn fn = nullptr;
  return fn;
}

auto
resolve_real_starpu_data_release() -> StarpuDataReleaseOverrideFn
{
  static StarpuDataReleaseOverrideFn fn = [] {
    void* symbol = dlsym(RTLD_NEXT, "starpu_data_release");
    if (symbol == nullptr) {
      throw std::runtime_error("Failed to resolve starpu_data_release");
    }
    return reinterpret_cast<StarpuDataReleaseOverrideFn>(symbol);
  }();
  return fn;
}

auto&
data_acquire_override_ref()
{
  static StarpuDataAcquireOverrideFn fn = nullptr;
  return fn;
}

auto
resolve_real_starpu_data_acquire() -> StarpuDataAcquireOverrideFn
{
  static StarpuDataAcquireOverrideFn fn = [] {
    void* symbol = dlsym(RTLD_NEXT, "starpu_data_acquire");
    if (symbol == nullptr) {
      throw std::runtime_error("Failed to resolve starpu_data_acquire");
    }
    return reinterpret_cast<StarpuDataAcquireOverrideFn>(symbol);
  }();
  return fn;
}

}  // namespace

#ifdef starpu_task_submit
#define STARPU_TASK_SUBMIT_MACRO_WAS_DEFINED 1
#undef starpu_task_submit
#endif

extern "C" int
starpu_task_submit(starpu_task* task)
{
  if (auto override = task_submit_override_ref()) {
    return override(task);
  }
  return resolve_real_starpu_task_submit()(task);
}

#ifdef STARPU_TASK_SUBMIT_MACRO_WAS_DEFINED
#define starpu_task_submit(task) \
  starpu_task_submit_line((task), __FILE__, __LINE__)
#undef STARPU_TASK_SUBMIT_MACRO_WAS_DEFINED
#endif

extern "C" void
starpu_vector_data_register(
    starpu_data_handle_t* handle, int home_node, uintptr_t ptr, size_t nx,
    size_t elemsize)
{
  if (auto override = vector_register_override_ref()) {
    override(handle, home_node, ptr, nx, elemsize);
    return;
  }
  resolve_real_starpu_vector_data_register()(
      handle, home_node, ptr, nx, elemsize);
}

extern "C" void
starpu_data_release(starpu_data_handle_t handle)
{
  if (auto override = data_release_override_ref()) {
    override(handle);
    return;
  }
  resolve_real_starpu_data_release()(handle);
}

extern "C" int
starpu_data_acquire(starpu_data_handle_t handle, starpu_data_access_mode mode)
{
  if (auto override = data_acquire_override_ref()) {
    return override(handle, mode);
  }
  return resolve_real_starpu_data_acquire()(handle, mode);
}

namespace starpu_test {

auto
set_starpu_task_submit_override(StarpuTaskSubmitOverrideFn fn)
    -> StarpuTaskSubmitOverrideFn
{
  auto& ref = task_submit_override_ref();
  auto previous = ref;
  ref = fn;
  return previous;
}

ScopedStarpuTaskSubmitOverride::ScopedStarpuTaskSubmitOverride(
    StarpuTaskSubmitOverrideFn fn)
    : previous_(set_starpu_task_submit_override(fn))
{
}

ScopedStarpuTaskSubmitOverride::~ScopedStarpuTaskSubmitOverride()
{
  set_starpu_task_submit_override(previous_);
}

auto
set_starpu_vector_register_override(StarpuVectorRegisterOverrideFn fn)
    -> StarpuVectorRegisterOverrideFn
{
  auto& ref = vector_register_override_ref();
  auto previous = ref;
  ref = fn;
  return previous;
}

ScopedStarpuVectorRegisterOverride::ScopedStarpuVectorRegisterOverride(
    StarpuVectorRegisterOverrideFn fn)
    : previous_(set_starpu_vector_register_override(fn))
{
}

ScopedStarpuVectorRegisterOverride::~ScopedStarpuVectorRegisterOverride()
{
  set_starpu_vector_register_override(previous_);
}

auto
set_starpu_data_release_override(StarpuDataReleaseOverrideFn fn)
    -> StarpuDataReleaseOverrideFn
{
  auto& ref = data_release_override_ref();
  auto previous = ref;
  ref = fn;
  return previous;
}

ScopedStarpuDataReleaseOverride::ScopedStarpuDataReleaseOverride(
    StarpuDataReleaseOverrideFn fn)
    : previous_(set_starpu_data_release_override(fn))
{
}

ScopedStarpuDataReleaseOverride::~ScopedStarpuDataReleaseOverride()
{
  set_starpu_data_release_override(previous_);
}

auto
set_starpu_data_acquire_override(StarpuDataAcquireOverrideFn fn)
    -> StarpuDataAcquireOverrideFn
{
  auto& ref = data_acquire_override_ref();
  auto previous = ref;
  ref = fn;
  return previous;
}

ScopedStarpuDataAcquireOverride::ScopedStarpuDataAcquireOverride(
    StarpuDataAcquireOverrideFn fn)
    : previous_(set_starpu_data_acquire_override(fn))
{
}

ScopedStarpuDataAcquireOverride::~ScopedStarpuDataAcquireOverride()
{
  set_starpu_data_acquire_override(previous_);
}

}  // namespace starpu_test
