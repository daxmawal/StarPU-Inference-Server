#include <gtest/gtest.h>

#include <exception>
#include <new>
#include <stdexcept>
#include <string>

#include "utils/exception_classification.hpp"

namespace starpu_server { namespace {

TEST(ExceptionClassification, NullExceptionUsesCurrentExceptionWhenAvailable)
{
  int callback_count = 0;
  bool runtime_called = false;
  std::string runtime_message;

  try {
    throw std::runtime_error("forced runtime failure");
  }
  catch (...) {
    classify_and_handle_exception(
        nullptr, [&](const InferenceEngineException&) { ++callback_count; },
        [&](const std::runtime_error& exception) {
          ++callback_count;
          runtime_called = true;
          runtime_message = exception.what();
        },
        [&](const std::logic_error&) { ++callback_count; },
        [&](const std::bad_alloc&) { ++callback_count; },
        [&](const std::exception&) { ++callback_count; },
        [&]() { ++callback_count; });
  }

  EXPECT_TRUE(runtime_called);
  EXPECT_EQ(runtime_message, "forced runtime failure");
  EXPECT_EQ(callback_count, 1);
}

TEST(
    ExceptionClassification,
    NullExceptionWithoutCurrentExceptionFallsBackToUnknown)
{
  int callback_count = 0;
  bool unknown_called = false;

  classify_and_handle_exception(
      nullptr, [&](const InferenceEngineException&) { ++callback_count; },
      [&](const std::runtime_error&) { ++callback_count; },
      [&](const std::logic_error&) { ++callback_count; },
      [&](const std::bad_alloc&) { ++callback_count; },
      [&](const std::exception&) { ++callback_count; },
      [&]() {
        ++callback_count;
        unknown_called = true;
      });

  EXPECT_TRUE(unknown_called);
  EXPECT_EQ(callback_count, 1);
}

}}  // namespace starpu_server
