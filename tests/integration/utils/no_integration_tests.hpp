#pragma once

#include <gtest/gtest.h>

#define SKIP_NO_INTEGRATION_TEST(test_suite, message) \
  TEST(test_suite, NoIntegrationScenario)             \
  {                                                   \
    GTEST_SKIP() << message;                          \
  }
