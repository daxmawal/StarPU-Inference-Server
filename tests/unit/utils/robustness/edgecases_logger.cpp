#include <gtest/gtest.h>

#include <memory>
#include <utility>

#include "core/inference_task.hpp"
#include "test_helpers.hpp"
#include "utils/exceptions.hpp"
#include "utils/logger.hpp"

using namespace starpu_server;

TEST(Logger_Robustesse, LogFatalDeath)
{
  EXPECT_DEATH({ log_fatal("boom"); }, "\\[FATAL\\] boom");
}

TEST(Logger_Robustesse, LogFatalExit)
{
  EXPECT_EXIT(
      log_fatal("fatal test"), ::testing::KilledBySignal(SIGABRT),
      "fatal test");
}

struct ExceptionCase {
  std::function<std::unique_ptr<std::exception>()> make_exception;
  std::string expected;
};
class LogException_Robustesse : public ::testing::TestWithParam<ExceptionCase> {
};

TEST_P(LogException_Robustesse, LogsExpectedMessage)
{
  const auto& param = GetParam();
  auto excep = param.make_exception();
  CaptureStream capture{std::cerr};
  InferenceTask::log_exception("ctx", *excep);
  EXPECT_EQ(capture.str(), param.expected);
}

INSTANTIATE_TEST_SUITE_P(
    ExceptionLogging, LogException_Robustesse,
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
