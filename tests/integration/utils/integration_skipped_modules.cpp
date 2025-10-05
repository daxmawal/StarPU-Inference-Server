#include "no_integration_tests.hpp"

SKIP_NO_INTEGRATION_TEST(
    ClientTime_Integration,
    "No integration scenario for client_utils/time_utils : Everything is pure "
    "(stateless, no external I/O).");

SKIP_NO_INTEGRATION_TEST(
    DatatypeUtils_Integration,
    "No integration scenario: pure utilities (no I/O).");

SKIP_NO_INTEGRATION_TEST(
    InputGenerator_Integration,
    "No integration scenario: local generation without I/O.");

SKIP_NO_INTEGRATION_TEST(
    Logger_Integration,
    "No integration scenario: local generation without I/O.");

SKIP_NO_INTEGRATION_TEST(
    TransparentHash_Integration,
    "No integration scenario: STL containers + transparent hash.");
