#pragma once

#if !defined(STARPU_TESTING)
#error \
    "inference_service_test_internal.hpp is test-only and requires STARPU_TESTING"
#endif

#include "support/grpc/server/inference_service_test_internal.hpp"
