#include <gtest/gtest.h>

#include <functional>
#include <memory>
#include <sstream>

#include "core/inference_task.hpp"
#include "utils/exceptions.hpp"
#include "utils/logger.hpp"

using namespace starpu_server;

struct ExceptionCase {
  std::function<std::unique_ptr<std::exception>()> make_exception;
  std::string expected;
};

class LogExceptionTest : public ::testing::TestWithParam<ExceptionCase> {
 protected:
  std::ostringstream oss;
  std::streambuf* old_buf;

  void SetUp() override { old_buf = std::cerr.rdbuf(oss.rdbuf()); }

  void TearDown() override { std::cerr.rdbuf(old_buf); }
};

TEST_P(LogExceptionTest, LogsExpectedMessage)
{
  auto param = GetParam();
  auto e = param.make_exception();
  InferenceTask::log_exception("ctx", *e);
  EXPECT_EQ(oss.str(), param.expected);
}

INSTANTIATE_TEST_SUITE_P(
    ExceptionLogging, LogExceptionTest,
    ::testing::Values(
        ExceptionCase{
            []() {
              return std::make_unique<InferenceExecutionException>("exec");
            },
            "\033[1;31m[ERROR] InferenceExecutionException in ctx: "
            "exec\033[0m\n"},
        ExceptionCase{
            []() {
              return std::make_unique<StarPUTaskSubmissionException>("sub");
            },
            "\033[1;31m[ERROR] StarPU submission error in ctx: "
            "sub\033[0m\n"},
        ExceptionCase{
            []() { return std::make_unique<std::runtime_error>("boom"); },
            "\033[1;31m[ERROR] std::exception in ctx: boom\033[0m\n"}));
