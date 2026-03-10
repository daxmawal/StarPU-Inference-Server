#pragma once

#if !defined(STARPU_TESTING)
#error "server_main_test_api.hpp is test-only and requires STARPU_TESTING"
#endif

#include "grpc/server/server_main.cpp"

namespace starpu_server::testing {
namespace server_main_api = ::starpu_server::testing::server_main;
}
