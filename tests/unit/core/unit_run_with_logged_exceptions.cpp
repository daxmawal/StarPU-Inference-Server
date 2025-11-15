#include <gtest/gtest.h>

#include <new>
#include <stdexcept>
#include <string>
#include <string_view>

#include "../../../src/core/inference_task.cpp"
#include "test_helpers.hpp"

namespace starpu_server { namespace {

TEST(RunWithLoggedExceptionsStandalone, LogsBadAlloc)
{
  CaptureStream capture{std::cerr};
  ExceptionLoggingMessages messages{"ctx: ", std::string_view{}};

  EXPECT_NO_THROW(
      run_with_logged_exceptions([]() { throw std::bad_alloc(); }, messages));

  const auto expected_message =
      std::string(messages.context_prefix) + std::bad_alloc().what();
  const auto expected_log = expected_log_line(ErrorLevel, expected_message);
  EXPECT_NE(capture.str().find(expected_log), std::string::npos);
}

TEST(RunWithLoggedExceptionsStandalone, LogsLogicError)
{
  CaptureStream capture{std::cerr};
  ExceptionLoggingMessages messages{"logic: ", std::string_view{}};

  EXPECT_NO_THROW(run_with_logged_exceptions(
      []() { throw std::logic_error("failure"); }, messages));

  const auto expected_log = expected_log_line(
      ErrorLevel, std::string(messages.context_prefix) + "failure");
  EXPECT_NE(capture.str().find(expected_log), std::string::npos);
}

TEST(RunWithLoggedExceptionsStandalone, LogsStdException)
{
  struct CustomStdException : public std::exception {
    [[nodiscard]] const char* what() const noexcept override
    {
      return "custom std exception";
    }
  };

  CaptureStream capture{std::cerr};
  ExceptionLoggingMessages messages{"std: ", std::string_view{}};

  EXPECT_NO_THROW(run_with_logged_exceptions(
      []() { throw CustomStdException{}; }, messages));

  const auto expected_log = expected_log_line(
      ErrorLevel,
      std::string(messages.context_prefix) + CustomStdException{}.what());
  EXPECT_NE(capture.str().find(expected_log), std::string::npos);
}

TEST(RunWithLoggedExceptionsStandalone, LogsUnknownExceptionFallback)
{
  CaptureStream capture{std::cerr};
  ExceptionLoggingMessages messages{"unknown: ", std::string_view{}};

  EXPECT_NO_THROW(run_with_logged_exceptions([]() { throw 42; }, messages));

  const auto expected_log = expected_log_line(
      ErrorLevel, std::string(messages.context_prefix) + "Unknown exception");
  EXPECT_NE(capture.str().find(expected_log), std::string::npos);
}

}}  // namespace starpu_server
