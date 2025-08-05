#pragma once

#include <starpu.h>

#include <cassert>
#include <stdexcept>

// RAII helper ensuring StarPU is initialized for the scope of a test.
// On construction it calls starpu_init(nullptr) and on destruction
// it calls starpu_shutdown().
struct StarpuRuntimeGuard {
  StarpuRuntimeGuard()
  {
    if (starpu_init(nullptr) != 0) {
      throw std::runtime_error("StarPU initialization failed");
    }
  }
  ~StarpuRuntimeGuard()
  {
    starpu_shutdown();
#ifndef NDEBUG
    assert(starpu_is_initialized() == 0 && "StarPU shutdown failed");
#endif
  }
  StarpuRuntimeGuard(const StarpuRuntimeGuard&) = delete;
  auto operator=(const StarpuRuntimeGuard&) -> StarpuRuntimeGuard& = delete;
  StarpuRuntimeGuard(StarpuRuntimeGuard&&) = delete;
  auto operator=(StarpuRuntimeGuard&&) -> StarpuRuntimeGuard& = delete;
};
