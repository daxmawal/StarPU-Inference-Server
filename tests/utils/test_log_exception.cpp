#include <gtest/gtest.h>

#include <functional>
#include <memory>

#include "../test_helpers.hpp"
#include "core/inference_task.hpp"
#include "utils/exceptions.hpp"

using namespace starpu_server;

struct ExceptionCase {
  std::function<std::unique_ptr<std::exception>()> make_exception;
  std::string expected;
};

class LogExceptionTest : public ::testing::TestWithParam<ExceptionCase> {};

TEST_P(LogExceptionTest, LogsExpectedMessage)
{
  auto param = GetParam();
  auto e = param.make_exception();
  CaptureStream capture{std::cerr};
  InferenceTask::log_exception("ctx", *e);
  EXPECT_EQ(capture.str(), param.expected);
}

INSTANTIATE_TEST_SUITE_P(
    ExceptionLogging, LogExceptionTest,
    ::testing::Values(
        ExceptionCase{
            []() {
              return std::make_unique<InferenceExecutionException>("exec");
            },
            expected_log_line(
                ErrorLevel, "InferenceExecutionException in ctx: exec")},
        ExceptionCase{
            []() {
              return std::make_unique<StarPUTaskSubmissionException>("sub");
            },
            expected_log_line(
                ErrorLevel, "StarPU submission error in ctx: sub")},
        ExceptionCase{
            []() { return std::make_unique<std::runtime_error>("boom"); },
            expected_log_line(ErrorLevel, "std::exception in ctx: boom")}));
