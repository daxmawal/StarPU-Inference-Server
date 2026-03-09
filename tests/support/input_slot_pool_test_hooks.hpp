#pragma once

#include <utility>

#include "core/input_slot_pool.hpp"

namespace starpu_server::testing {

using StarpuVectorRegisterFn = InputSlotPool::StarpuVectorRegisterFn;
using RegisterFailureObserverFn = InputSlotPool::RegisterFailureObserverFn;

void compute_input_sizes_for_tests(
    std::size_t per_sample_bytes, std::size_t per_sample_numel,
    std::size_t batch_size, std::size_t input_index);

inline auto
set_starpu_vector_register_hook_for_tests(StarpuVectorRegisterFn hook_fn)
    -> StarpuVectorRegisterFn
{
  auto& dependencies = input_slot_pool_default_dependencies_for_tests();
  const auto previous = dependencies.starpu_vector_register;
  dependencies.starpu_vector_register =
      hook_fn != nullptr ? hook_fn : &starpu_vector_data_register;
  return previous;
}

inline auto
set_starpu_register_failure_observer_for_tests(
    RegisterFailureObserverFn observer) -> RegisterFailureObserverFn
{
  auto& dependencies = input_slot_pool_default_dependencies_for_tests();
  auto previous = dependencies.register_failure_observer;
  dependencies.register_failure_observer = std::move(observer);
  return previous;
}

}  // namespace starpu_server::testing
