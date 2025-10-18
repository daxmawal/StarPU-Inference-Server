#pragma once

#include <utility>

#include "core/input_slot_pool.hpp"

namespace starpu_server::detail {

using StarpuVectorRegisterFn = decltype(&starpu_vector_data_register);
using RegisterFailureObserverFn =
    std::function<void(const InputSlotPool::SlotInfo& slot)>;

auto starpu_vector_register_hook() -> StarpuVectorRegisterFn&;
auto starpu_register_failure_observer() -> RegisterFailureObserverFn&;

}  // namespace starpu_server::detail

namespace starpu_server::testing {

using StarpuVectorRegisterFn = detail::StarpuVectorRegisterFn;
using RegisterFailureObserverFn = detail::RegisterFailureObserverFn;

inline auto
set_starpu_vector_register_hook_for_tests(StarpuVectorRegisterFn hook_fn)
    -> StarpuVectorRegisterFn
{
  auto& hook = detail::starpu_vector_register_hook();
  const auto previous = hook;
  hook = hook_fn != nullptr ? hook_fn : &starpu_vector_data_register;
  return previous;
}

inline auto
set_starpu_register_failure_observer_for_tests(
    RegisterFailureObserverFn observer) -> RegisterFailureObserverFn
{
  auto& failure_observer = detail::starpu_register_failure_observer();
  auto previous = failure_observer;
  failure_observer = std::move(observer);
  return previous;
}

}  // namespace starpu_server::testing
