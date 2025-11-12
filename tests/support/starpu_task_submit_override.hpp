#pragma once

#include <starpu.h>

namespace starpu_test {

using StarpuTaskSubmitOverrideFn = int (*)(starpu_task*);
using StarpuVectorRegisterOverrideFn =
    void (*)(starpu_data_handle_t*, int, uintptr_t, size_t, size_t);
using StarpuDataReleaseOverrideFn = void (*)(starpu_data_handle_t);
using StarpuDataAcquireOverrideFn =
    int (*)(starpu_data_handle_t, starpu_data_access_mode);

// Sets the override and returns the previous value.
auto set_starpu_task_submit_override(StarpuTaskSubmitOverrideFn fn)
    -> StarpuTaskSubmitOverrideFn;
auto set_starpu_vector_register_override(StarpuVectorRegisterOverrideFn fn)
    -> StarpuVectorRegisterOverrideFn;
auto set_starpu_data_release_override(StarpuDataReleaseOverrideFn fn)
    -> StarpuDataReleaseOverrideFn;
auto set_starpu_data_acquire_override(StarpuDataAcquireOverrideFn fn)
    -> StarpuDataAcquireOverrideFn;

class ScopedStarpuTaskSubmitOverride {
 public:
  explicit ScopedStarpuTaskSubmitOverride(StarpuTaskSubmitOverrideFn fn);
  ScopedStarpuTaskSubmitOverride(const ScopedStarpuTaskSubmitOverride&) =
      delete;
  auto operator=(const ScopedStarpuTaskSubmitOverride&)
      -> ScopedStarpuTaskSubmitOverride& = delete;
  ScopedStarpuTaskSubmitOverride(ScopedStarpuTaskSubmitOverride&&) = delete;
  auto operator=(ScopedStarpuTaskSubmitOverride&&)
      -> ScopedStarpuTaskSubmitOverride& = delete;
  ~ScopedStarpuTaskSubmitOverride();

 private:
  StarpuTaskSubmitOverrideFn previous_;
};

class ScopedStarpuVectorRegisterOverride {
 public:
  explicit ScopedStarpuVectorRegisterOverride(
      StarpuVectorRegisterOverrideFn fn);
  ScopedStarpuVectorRegisterOverride(
      const ScopedStarpuVectorRegisterOverride&) = delete;
  auto operator=(const ScopedStarpuVectorRegisterOverride&)
      -> ScopedStarpuVectorRegisterOverride& = delete;
  ScopedStarpuVectorRegisterOverride(ScopedStarpuVectorRegisterOverride&&) =
      delete;
  auto operator=(ScopedStarpuVectorRegisterOverride&&)
      -> ScopedStarpuVectorRegisterOverride& = delete;
  ~ScopedStarpuVectorRegisterOverride();

 private:
  StarpuVectorRegisterOverrideFn previous_;
};

class ScopedStarpuDataReleaseOverride {
 public:
  explicit ScopedStarpuDataReleaseOverride(StarpuDataReleaseOverrideFn fn);
  ScopedStarpuDataReleaseOverride(const ScopedStarpuDataReleaseOverride&) =
      delete;
  auto operator=(const ScopedStarpuDataReleaseOverride&)
      -> ScopedStarpuDataReleaseOverride& = delete;
  ScopedStarpuDataReleaseOverride(ScopedStarpuDataReleaseOverride&&) = delete;
  auto operator=(ScopedStarpuDataReleaseOverride&&)
      -> ScopedStarpuDataReleaseOverride& = delete;
  ~ScopedStarpuDataReleaseOverride();

 private:
  StarpuDataReleaseOverrideFn previous_;
};

class ScopedStarpuDataAcquireOverride {
 public:
  explicit ScopedStarpuDataAcquireOverride(StarpuDataAcquireOverrideFn fn);
  ScopedStarpuDataAcquireOverride(const ScopedStarpuDataAcquireOverride&) =
      delete;
  auto operator=(const ScopedStarpuDataAcquireOverride&)
      -> ScopedStarpuDataAcquireOverride& = delete;
  ScopedStarpuDataAcquireOverride(ScopedStarpuDataAcquireOverride&&) = delete;
  auto operator=(ScopedStarpuDataAcquireOverride&&)
      -> ScopedStarpuDataAcquireOverride& = delete;
  ~ScopedStarpuDataAcquireOverride();

 private:
  StarpuDataAcquireOverrideFn previous_;
};

}  // namespace starpu_test
