#include <gtest/gtest.h>

#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "../../../src/core/inference_task.cpp"
#include "test_helpers.hpp"

namespace starpu_server { namespace {

template <typename Fn>
void
ExpectLoggedException(
    std::string_view prefix, std::string_view expected, Fn&& fn)
{
  CaptureStream capture{std::cerr};
  ExceptionLoggingMessages messages{prefix, std::string_view{}};
  EXPECT_NO_THROW(run_with_logged_exceptions(std::forward<Fn>(fn), messages));
  const auto expected_log = expected_log_line(
      ErrorLevel, std::string(prefix) + std::string(expected));
  EXPECT_NE(capture.str().find(expected_log), std::string::npos);
}

TEST(RunWithLoggedExceptionsStandalone, LogsBadAlloc)
{
  ExpectLoggedException(
      "ctx: ", std::bad_alloc().what(), []() { throw std::bad_alloc(); });
}

TEST(RunWithLoggedExceptionsStandalone, LogsLogicError)
{
  ExpectLoggedException(
      "logic: ", "failure", []() { throw std::logic_error("failure"); });
}

TEST(RunWithLoggedExceptionsStandalone, LogsStdException)
{
  struct CustomStdException : public std::exception {
    [[nodiscard]] const char* what() const noexcept override
    {
      return "custom std exception";
    }
  };

  ExpectLoggedException("std: ", CustomStdException{}.what(), []() {
    throw CustomStdException{};
  });
}

TEST(RunWithLoggedExceptionsStandalone, LogsUnknownExceptionFallback)
{
  ExpectLoggedException("unknown: ", "Unknown exception", []() { throw 42; });
}

}}  // namespace starpu_server
