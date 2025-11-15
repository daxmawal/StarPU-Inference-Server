#include "support/starpu_data_interface_override.hpp"

#include <dlfcn.h>

#include <stdexcept>

namespace {

using starpu_test::StarpuDataGetInterfaceOnNodeOverrideFn;
using starpu_test::StarpuMemoryNodesGetCountOverrideFn;

auto&
interface_override_ref()
{
  static StarpuDataGetInterfaceOnNodeOverrideFn fn = nullptr;
  return fn;
}

auto
resolve_real_starpu_data_get_interface_on_node()
    -> StarpuDataGetInterfaceOnNodeOverrideFn
{
  static StarpuDataGetInterfaceOnNodeOverrideFn fn = [] {
    void* symbol = dlsym(RTLD_NEXT, "starpu_data_get_interface_on_node");
    if (symbol == nullptr) {
      throw std::runtime_error(
          "Failed to resolve starpu_data_get_interface_on_node");
    }
    return reinterpret_cast<StarpuDataGetInterfaceOnNodeOverrideFn>(symbol);
  }();
  return fn;
}

auto&
memory_nodes_override_ref()
{
  static StarpuMemoryNodesGetCountOverrideFn fn = nullptr;
  return fn;
}

auto
resolve_real_starpu_memory_nodes_get_count()
    -> StarpuMemoryNodesGetCountOverrideFn
{
  static StarpuMemoryNodesGetCountOverrideFn fn = [] {
    void* symbol = dlsym(RTLD_NEXT, "starpu_memory_nodes_get_count");
    if (symbol == nullptr) {
      throw std::runtime_error(
          "Failed to resolve starpu_memory_nodes_get_count");
    }
    return reinterpret_cast<StarpuMemoryNodesGetCountOverrideFn>(symbol);
  }();
  return fn;
}

}  // namespace

extern "C" void*
starpu_data_get_interface_on_node(starpu_data_handle_t handle, unsigned node)
{
  if (auto override = interface_override_ref()) {
    return override(handle, node);
  }
  return resolve_real_starpu_data_get_interface_on_node()(handle, node);
}

extern "C" unsigned
starpu_memory_nodes_get_count()
{
  if (auto override = memory_nodes_override_ref()) {
    return override();
  }
  return resolve_real_starpu_memory_nodes_get_count()();
}

namespace starpu_test {

auto
set_starpu_data_get_interface_on_node_override(
    StarpuDataGetInterfaceOnNodeOverrideFn fn)
    -> StarpuDataGetInterfaceOnNodeOverrideFn
{
  auto& ref = interface_override_ref();
  auto previous = ref;
  ref = fn;
  return previous;
}

ScopedStarpuDataGetInterfaceOnNodeOverride::
    ScopedStarpuDataGetInterfaceOnNodeOverride(
        StarpuDataGetInterfaceOnNodeOverrideFn fn)
    : previous_(set_starpu_data_get_interface_on_node_override(fn))
{
}

ScopedStarpuDataGetInterfaceOnNodeOverride::
    ~ScopedStarpuDataGetInterfaceOnNodeOverride()
{
  set_starpu_data_get_interface_on_node_override(previous_);
}

auto
set_starpu_memory_nodes_get_count_override(
    StarpuMemoryNodesGetCountOverrideFn fn)
    -> StarpuMemoryNodesGetCountOverrideFn
{
  auto& ref = memory_nodes_override_ref();
  auto previous = ref;
  ref = fn;
  return previous;
}

ScopedStarpuMemoryNodesGetCountOverride::
    ScopedStarpuMemoryNodesGetCountOverride(
        StarpuMemoryNodesGetCountOverrideFn fn)
    : previous_(set_starpu_memory_nodes_get_count_override(fn))
{
}

ScopedStarpuMemoryNodesGetCountOverride::
    ~ScopedStarpuMemoryNodesGetCountOverride()
{
  set_starpu_memory_nodes_get_count_override(previous_);
}

auto
call_real_starpu_data_get_interface_on_node(
    starpu_data_handle_t handle, unsigned node) -> void*
{
  return resolve_real_starpu_data_get_interface_on_node()(handle, node);
}

auto
call_real_starpu_memory_nodes_get_count() -> unsigned
{
  return resolve_real_starpu_memory_nodes_get_count()();
}

}  // namespace starpu_test
