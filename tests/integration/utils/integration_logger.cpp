#include <gtest/gtest.h>

TEST(Logger_Integration, NoIntegrationScenario)
{
  GTEST_SKIP() << "No integration scenario: local generation without I/O.";
}
