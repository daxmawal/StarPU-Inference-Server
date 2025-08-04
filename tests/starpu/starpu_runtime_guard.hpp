#pragma once

#include <starpu.h>

// RAII helper ensuring StarPU is initialized for the scope of a test.
// On construction it calls starpu_init(nullptr) and on destruction
// it calls starpu_shutdown().
struct StarpuRuntimeGuard {
  StarpuRuntimeGuard() { starpu_init(nullptr); }
  ~StarpuRuntimeGuard() { starpu_shutdown(); }

  StarpuRuntimeGuard(const StarpuRuntimeGuard&) = delete;
  auto operator=(const StarpuRuntimeGuard&) -> StarpuRuntimeGuard& = delete;
  StarpuRuntimeGuard(StarpuRuntimeGuard&&) = delete;
  auto operator=(StarpuRuntimeGuard&&) -> StarpuRuntimeGuard& = delete;
};
