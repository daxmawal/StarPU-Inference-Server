#pragma once

#include <starpu.h>

namespace starpu_test {

using StarpuDataGetInterfaceOnNodeOverrideFn = void* (*)(starpu_data_handle_t,
                                                         unsigned);
using StarpuMemoryNodesGetCountOverrideFn = unsigned (*)();

auto set_starpu_data_get_interface_on_node_override(
    StarpuDataGetInterfaceOnNodeOverrideFn fn)
    -> StarpuDataGetInterfaceOnNodeOverrideFn;

auto set_starpu_memory_nodes_get_count_override(
    StarpuMemoryNodesGetCountOverrideFn fn)
    -> StarpuMemoryNodesGetCountOverrideFn;

class ScopedStarpuDataGetInterfaceOnNodeOverride {
 public:
  explicit ScopedStarpuDataGetInterfaceOnNodeOverride(
      StarpuDataGetInterfaceOnNodeOverrideFn fn);
  ScopedStarpuDataGetInterfaceOnNodeOverride(
      const ScopedStarpuDataGetInterfaceOnNodeOverride&) = delete;
  auto operator=(const ScopedStarpuDataGetInterfaceOnNodeOverride&)
      -> ScopedStarpuDataGetInterfaceOnNodeOverride& = delete;
  ScopedStarpuDataGetInterfaceOnNodeOverride(
      ScopedStarpuDataGetInterfaceOnNodeOverride&&) = delete;
  auto operator=(ScopedStarpuDataGetInterfaceOnNodeOverride&&)
      -> ScopedStarpuDataGetInterfaceOnNodeOverride& = delete;
  ~ScopedStarpuDataGetInterfaceOnNodeOverride();

 private:
  StarpuDataGetInterfaceOnNodeOverrideFn previous_;
};

class ScopedStarpuMemoryNodesGetCountOverride {
 public:
  explicit ScopedStarpuMemoryNodesGetCountOverride(
      StarpuMemoryNodesGetCountOverrideFn fn);
  ScopedStarpuMemoryNodesGetCountOverride(
      const ScopedStarpuMemoryNodesGetCountOverride&) = delete;
  auto operator=(const ScopedStarpuMemoryNodesGetCountOverride&)
      -> ScopedStarpuMemoryNodesGetCountOverride& = delete;
  ScopedStarpuMemoryNodesGetCountOverride(
      ScopedStarpuMemoryNodesGetCountOverride&&) = delete;
  auto operator=(ScopedStarpuMemoryNodesGetCountOverride&&)
      -> ScopedStarpuMemoryNodesGetCountOverride& = delete;
  ~ScopedStarpuMemoryNodesGetCountOverride();

 private:
  StarpuMemoryNodesGetCountOverrideFn previous_;
};

auto call_real_starpu_data_get_interface_on_node(
    starpu_data_handle_t handle, unsigned node) -> void*;
auto call_real_starpu_memory_nodes_get_count() -> unsigned;

}  // namespace starpu_test
