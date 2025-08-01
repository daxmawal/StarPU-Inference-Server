#include <gtest/gtest.h>

#include <sstream>

#include "core/inference_task.hpp"
#include "utils/exceptions.hpp"
#include "utils/logger.hpp"

using namespace starpu_server;

TEST(InferenceTask, LogExceptionMessages)
{
  std::ostringstream oss;
  auto* old_buf = std::cerr.rdbuf(oss.rdbuf());

  InferenceExecutionException exec("exec");
  InferenceTask::log_exception("ctx", exec);
  std::string expected1 =
      "\033[1;31m[ERROR] InferenceExecutionException in ctx: exec\033[0m\n";
  EXPECT_EQ(oss.str(), expected1);

  oss.str("");
  oss.clear();
  StarPUTaskSubmissionException sub("sub");
  InferenceTask::log_exception("ctx", sub);
  std::string expected2 =
      "\033[1;31m[ERROR] StarPU submission error in ctx: sub\033[0m\n";
  EXPECT_EQ(oss.str(), expected2);

  oss.str("");
  oss.clear();
  std::runtime_error generic("boom");
  InferenceTask::log_exception("ctx", generic);
  std::string expected3 =
      "\033[1;31m[ERROR] std::exception in ctx: boom\033[0m\n";
  EXPECT_EQ(oss.str(), expected3);

  std::cerr.rdbuf(old_buf);
}