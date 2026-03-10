#pragma once

#include <string_view>

#include "core/inference_task.hpp"

namespace starpu_server::testing {

void finalize_or_fail_once_for_tests(
    InferenceCallbackContext* ctx, bool failure, std::string_view context);

}  // namespace starpu_server::testing
