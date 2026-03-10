#pragma once

#include <starpu.h>
#include <starpu_data_interfaces.h>

#include <cstddef>
#include <format>
#include <limits>

#include "exceptions.hpp"

namespace starpu_server::task_runner_internal {

struct VectorResizeSpec {
  std::size_t element_count;
  std::size_t byte_count;
};

inline void
resize_starpu_vector_interface(
    starpu_vector_interface* vector_interface, VectorResizeSpec spec,
    bool is_input_handle)
{
  if (vector_interface == nullptr) {
    return;
  }

  const auto elem_size = vector_interface->elemsize;
  if (elem_size == 0) {
    throw StarPUDataAcquireException(
        "StarPU vector interface reported zero element size");
  }

  if (spec.byte_count % elem_size != 0) {
    if (is_input_handle) {
      throw InvalidInputTensorException(std::format(
          "Input tensor byte size ({}) is not divisible by element size ({})",
          spec.byte_count, elem_size));
    }
    throw InvalidInferenceJobException(std::format(
        "Output tensor byte size ({}) is not divisible by element size ({})",
        spec.byte_count, elem_size));
  }

  const auto required_numel = spec.byte_count / elem_size;
  if (required_numel != spec.element_count) {
    spec.element_count = required_numel;
  }

  const auto alloc_size = vector_interface->allocsize;
  if (alloc_size != std::numeric_limits<size_t>::max() && alloc_size != 0 &&
      spec.byte_count > alloc_size) {
    if (is_input_handle) {
      throw InputPoolCapacityException(std::format(
          "Input tensor requires {} bytes but slot capacity is {} bytes",
          spec.byte_count, alloc_size));
    }
    throw InvalidInferenceJobException(std::format(
        "Output tensor requires {} bytes but slot capacity is {} bytes",
        spec.byte_count, alloc_size));
  }

  vector_interface->nx = spec.element_count;
}

inline void
resize_starpu_vector_handle(
    starpu_data_handle_t handle, VectorResizeSpec spec, bool is_input_handle)
{
  if (handle == nullptr) {
    throw StarPUDataAcquireException("StarPU vector handle is null");
  }

  if (starpu_data_get_interface_id(handle) != STARPU_VECTOR_INTERFACE_ID) {
    throw StarPUDataAcquireException(
        "Expected StarPU vector interface for handle");
  }

  const unsigned memory_nodes = starpu_memory_nodes_get_count();
  for (unsigned node = 0; node < memory_nodes; ++node) {
    auto* raw_interface = starpu_data_get_interface_on_node(handle, node);
    if (raw_interface == nullptr) {
      continue;
    }
    auto* vector_interface =
        static_cast<starpu_vector_interface*>(raw_interface);
    resize_starpu_vector_interface(vector_interface, spec, is_input_handle);
  }
}

}  // namespace starpu_server::task_runner_internal
