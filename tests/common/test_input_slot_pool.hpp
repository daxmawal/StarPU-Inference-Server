#pragma once

#include <starpu.h>

#include "core/input_slot_pool.hpp"

namespace starpu_server::testing {

using StarpuVectorRegisterFn = decltype(&starpu_vector_data_register);
using RegisterFailureObserverFn = void (*)(const InputSlotPool::SlotInfo& slot);

auto set_starpu_vector_register_hook_for_tests(StarpuVectorRegisterFn fn)
    -> StarpuVectorRegisterFn;

auto set_starpu_register_failure_observer_for_tests(
    RegisterFailureObserverFn observer) -> RegisterFailureObserverFn;

}  // namespace starpu_server::testing
