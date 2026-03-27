#include "output_slot_pool.hpp"

#include <cuda_runtime_api.h>
#include <starpu.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <format>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>

#include "slot_pool_buffer_utils.hpp"
#include "utils/datatype_utils.hpp"
#include "utils/exceptions.hpp"
#include "utils/logger.hpp"

namespace starpu_server {
inline namespace output_slot_pool_detail {

auto
normalize_output_slot_pool_dependencies(
    OutputSlotPool::Dependencies dependencies) -> OutputSlotPool::Dependencies
{
  constexpr size_t kDefaultHostAlignment = 64;
  if (dependencies.starpu_vector_register == nullptr) {
    dependencies.starpu_vector_register = &starpu_vector_data_register;
  }
  if (!dependencies.host_allocator) {
    dependencies.host_allocator = [](OutputSlotPool::HostBufferPtr* ptr,
                                     size_t alignment, size_t size) {
      return detail::try_allocate_aligned_host_buffer(ptr, alignment, size);
    };
  }
  if (!dependencies.cuda_host_alloc) {
    dependencies.cuda_host_alloc = [](void** ptr, size_t size,
                                      unsigned int flags) {
      return static_cast<int>(cudaHostAlloc(ptr, size, flags));
    };
  }
  if (!dependencies.host_deallocator) {
    dependencies.host_deallocator = [](void* ptr) {
      ::operator delete(ptr, std::align_val_t{kDefaultHostAlignment});
    };
  }
  if (!dependencies.starpu_memory_pin) {
    dependencies.starpu_memory_pin = &starpu_memory_pin;
  }
  return dependencies;
}

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
auto
output_slot_pool_default_dependencies_storage_for_tests()
    -> OutputSlotPool::Dependencies&
{
  static OutputSlotPool::Dependencies dependencies =
      normalize_output_slot_pool_dependencies({});
  return dependencies;
}
#endif  // SONAR_IGNORE_END

auto
resolve_output_slot_pool_dependencies_for_construction()
    -> OutputSlotPool::Dependencies
{
#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
  return normalize_output_slot_pool_dependencies(
      output_slot_pool_default_dependencies_storage_for_tests());
#else
  return normalize_output_slot_pool_dependencies({});
#endif  // SONAR_IGNORE_END
}

}  // namespace output_slot_pool_detail

#if defined(STARPU_TESTING)  // SONAR_IGNORE_START
namespace testing {
auto
output_slot_pool_default_dependencies_for_tests()
    -> OutputSlotPool::Dependencies&
{
  return output_slot_pool_default_dependencies_storage_for_tests();
}
}  // namespace testing
#endif  // SONAR_IGNORE_END

auto
OutputSlotPool::alloc_host_buffer(
    size_t bytes, bool use_pinned, bool& cuda_pinned_out,
    const Dependencies& dependencies) -> HostBufferPtr
{
  return detail::allocate_host_buffer_with_optional_cuda_pinning(
      bytes, use_pinned, cuda_pinned_out,
      [&dependencies, use_pinned](size_t requested_bytes) -> std::byte* {
        void* raw_ptr = nullptr;
        const int err = dependencies.cuda_host_alloc(
            &raw_ptr, requested_bytes, cudaHostAllocPortable);
        if (err != static_cast<int>(cudaSuccess) || raw_ptr == nullptr) {
          return nullptr;
        }

        bool keep_cuda_pinned = true;
        if (dependencies.cuda_pinned_override) {
          keep_cuda_pinned = dependencies.cuda_pinned_override(
              requested_bytes, use_pinned, true);
        }
        if (!keep_cuda_pinned) {
          cudaFreeHost(raw_ptr);
          return nullptr;
        }

        return static_cast<std::byte*>(raw_ptr);
      },
      [&dependencies](size_t requested_bytes) {
        constexpr size_t kAlign = 64;
        std::byte* raw_ptr = nullptr;
        const int alloc_rc =
            dependencies.host_allocator(&raw_ptr, kAlign, requested_bytes);
        if (alloc_rc != 0 || raw_ptr == nullptr) {
          throw std::bad_alloc();
        }
        return raw_ptr;
      });
}

auto
OutputSlotPool::checked_total_bytes(size_t per_sample_bytes, size_t batch_size)
    -> size_t
{
  return detail::checked_size_product(
      per_sample_bytes, batch_size,
      [](std::size_t checked_per_sample_bytes, std::size_t checked_batch_size) {
        return std::format(
            "OutputSlotPool: per-sample bytes ({}) times batch size ({}) "
            "exceeds size_t range",
            checked_per_sample_bytes, checked_batch_size);
      });
}

auto
OutputSlotPool::prepare_host_buffer(
    size_t per_sample_bytes, size_t batch_size, bool want_pinned, int slot_id,
    size_t output_index, const Dependencies& dependencies) -> PreparedHostBuffer
{
  PreparedHostBuffer prepared;
  prepared.info.bytes = checked_total_bytes(per_sample_bytes, batch_size);
  prepared.ptr = alloc_host_buffer(
      prepared.info.bytes, want_pinned, prepared.info.cuda_pinned,
      dependencies);

  detail::try_pin_host_buffer_for_starpu(
      prepared.ptr, want_pinned, prepared.info, dependencies.starpu_memory_pin,
      [slot_id, output_index](int pin_result) {
        log_warning(std::format(
            "starpu_memory_pin failed for output slot {}, index {}: rc={}",
            slot_id, output_index, pin_result));
      });

  return prepared;
}

void
OutputSlotPool::cleanup_slot_buffers(
    SlotInfo& slot, std::vector<HostBufferInfo>& buffer_infos, size_t count,
    const Dependencies& dependencies)
{
  detail::cleanup_slot_resources(
      slot, buffer_infos, count,
      [&dependencies](std::byte* ptr, const HostBufferInfo& buffer_info) {
        free_host_buffer(ptr, buffer_info, dependencies);
      },
      detail::SlotCleanupOrder::FreeThenUnregister,
      /*reset_buffer_info=*/true);
}

auto
OutputSlotPool::checked_total_numel(size_t per_sample_numel, size_t batch_size)
    -> size_t
{
  return detail::checked_size_product(
      per_sample_numel, batch_size,
      [](std::size_t checked_per_sample_numel, std::size_t checked_batch_size) {
        return std::format(
            "OutputSlotPool: per-sample numel ({}) times batch size ({}) "
            "exceeds size_t range",
            checked_per_sample_numel, checked_batch_size);
      });
}

void
OutputSlotPool::free_host_buffer(
    std::byte* ptr, const HostBufferInfo& buffer_info,
    const Dependencies& dependencies)
{
  if (ptr == nullptr) {
    return;
  }
  detail::try_unpin_host_buffer_for_starpu(ptr, buffer_info, [](int unpin_rc) {
    log_warning(std::format(
        "starpu_memory_unpin failed for output buffer: rc={}",
        std::to_string(unpin_rc)));
    (void)cudaGetLastError();
  });
  if (buffer_info.cuda_pinned) {
    cudaError_t cuda_rc = cudaFreeHost(static_cast<void*>(ptr));
    if (cuda_rc != cudaSuccess) {
      log_warning(std::format(
          "cudaFreeHost failed for output buffer: rc={}",
          std::to_string(static_cast<int>(cuda_rc))));
    }
    return;
  }
  dependencies.host_deallocator(static_cast<void*>(ptr));
}

OutputSlotPool::OutputSlotPool(const RuntimeConfig& opts, int slots)
    : OutputSlotPool(
          opts, slots, resolve_output_slot_pool_dependencies_for_construction())
{
}

OutputSlotPool::OutputSlotPool(
    const RuntimeConfig& opts, int slots, Dependencies dependencies)
    : dependencies_(
          normalize_output_slot_pool_dependencies(std::move(dependencies)))
{
  bmax_ = std::max(1, opts.batching.max_batch_size);

  if (!opts.model.has_value()) {
    throw std::invalid_argument("No model config provided for OutputSlotPool");
  }
  const auto& outputs = opts.model->outputs;
  output_types_.reserve(outputs.size());
  per_output_numel_single_.reserve(outputs.size());
  per_output_bytes_single_.reserve(outputs.size());

  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto& output_desc = outputs[i];
    output_types_.push_back(output_desc.type);
    if (output_desc.dims.size() >= 2) {
      const int64_t batch_dim = output_desc.dims[0];
      if (batch_dim <= 0) {
        throw std::invalid_argument("dims[0] (batch) must be positive");
      }
      if (batch_dim > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("dims[0] (batch) exceeds int max");
      }
      bmax_ = std::max(bmax_, static_cast<int>(batch_dim));
    }

    const size_t numel = product_dims(output_desc.dims);
    per_output_numel_single_.push_back(numel);
    const size_t elsize = element_size(output_desc.type);
    if (elsize != 0 && numel > std::numeric_limits<size_t>::max() / elsize) {
      throw std::overflow_error(std::format(
          "OutputSlotPool: per-sample bytes overflow for output {}", i));
    }
    per_output_bytes_single_.push_back(numel * elsize);
  }

  allocate_pool(opts, slots);
}

OutputSlotPool::~OutputSlotPool()
{
  detail::cleanup_pool_slots(
      slots(), host_buffer_infos_,
      [this](std::byte* ptr, const HostBufferInfo& buffer_info) {
        free_host_buffer(ptr, buffer_info, dependencies_);
      });
}

void
OutputSlotPool::allocate_pool(const RuntimeConfig& opts, int slots)
{
  detail::init_pool_slots(
      slots, this->slots(), host_buffer_infos_, this->free_ids(),
      [this, &opts](int slot_id) {
        allocate_slot_buffers_and_register(slot_id, opts);
      });
}

void
OutputSlotPool::allocate_slot_buffers_and_register(
    int slot_id, const RuntimeConfig& opts)
{
  const size_t num_outputs = per_output_numel_single_.size();
  auto& slot = slots().at(static_cast<size_t>(slot_id));
  slot.base_ptrs.assign(num_outputs, nullptr);
  slot.handles.assign(num_outputs, nullptr);
  auto& buffer_infos = host_buffer_infos_[static_cast<size_t>(slot_id)];
  buffer_infos.assign(num_outputs, HostBufferInfo{});

  const bool want_pinned = opts.devices.use_cuda;
  const auto batch_size = static_cast<size_t>(bmax_);

  try {
    for (size_t i = 0; i < num_outputs; ++i) {
      const auto prepared_buffer = prepare_host_buffer(
          per_output_bytes_single_[i], batch_size, want_pinned, slot_id, i,
          dependencies_);
      slot.base_ptrs[i] = prepared_buffer.ptr;
      buffer_infos[i] = prepared_buffer.info;

      const size_t total_numel =
          checked_total_numel(per_output_numel_single_[i], batch_size);
      starpu_data_handle_t handle = nullptr;
      dependencies_.starpu_vector_register(
          &handle, STARPU_MAIN_RAM,
          std::bit_cast<uintptr_t>(prepared_buffer.ptr), total_numel,
          element_size(output_types_[i]));
      if (handle == nullptr) {
        cleanup_slot_buffers(slot, buffer_infos, i + 1, dependencies_);
        if (dependencies_.register_failure_observer != nullptr) {
          dependencies_.register_failure_observer(slot, buffer_infos);
        }
        throw OutputSlotRegistrationError(
            "Failed to register StarPU vector handle for output");
      }
      slot.handles[i] = handle;
    }
  }
  catch (...) {
    cleanup_slot_buffers(slot, buffer_infos, num_outputs, dependencies_);
    throw;
  }
}

auto
OutputSlotPool::product_dims(const std::vector<int64_t>& dims) -> size_t
{
  size_t product = 1;
  const size_t start_dim_index = dims.size() >= 2 ? 1 : 0;
  for (size_t i = start_dim_index; i < dims.size(); ++i) {
    const auto dimension = dims[i];
    if (dimension <= 0) {
      throw std::invalid_argument("dimensions must be positive");
    }
    const auto dimension_unsigned = static_cast<size_t>(dimension);
    if (product > std::numeric_limits<size_t>::max() / dimension_unsigned) {
      throw std::overflow_error("dimension product overflow");
    }
    product *= dimension_unsigned;
  }
  return product;
}

}  // namespace starpu_server
