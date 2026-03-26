#include <gtest/gtest.h>

#include <exception>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils/exception_classification.hpp"

namespace starpu_server { namespace {

class PlainStdException final : public std::exception {
 public:
  [[nodiscard]] auto what() const noexcept -> const char* override
  {
    return "plain std exception";
  }
};

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

TEST(
    ExceptionClassification,
    SingleCallbackOverloadClassifiesConcreteStdExceptionTypes)
{
  struct Case {
    std::exception_ptr exception;
    ExceptionCategory expected_category;
    std::string expected_message;
  };

  const std::vector<Case> cases{
      {std::make_exception_ptr(InferenceEngineException{"inference"}),
       ExceptionCategory::InferenceEngine, "inference"},
      {std::make_exception_ptr(std::runtime_error{"runtime"}),
       ExceptionCategory::RuntimeError, "runtime"},
      {std::make_exception_ptr(std::logic_error{"logic"}),
       ExceptionCategory::LogicError, "logic"},
      {std::make_exception_ptr(std::bad_alloc{}), ExceptionCategory::BadAlloc,
       "std::bad_alloc"},
      {std::make_exception_ptr(PlainStdException{}),
       ExceptionCategory::StdException, "plain std exception"},
  };

  for (const auto& test_case : cases) {
    ExceptionCategory observed_category = ExceptionCategory::Unknown;
    const std::exception* observed_exception = nullptr;

    classify_and_handle_exception(
        test_case.exception,
        [&](ExceptionCategory category, const std::exception* exception) {
          observed_category = category;
          observed_exception = exception;
        });

    ASSERT_NE(observed_exception, nullptr);
    EXPECT_EQ(observed_category, test_case.expected_category);
    EXPECT_EQ(observed_exception->what(), test_case.expected_message);
  }
}

}}  // namespace starpu_server
