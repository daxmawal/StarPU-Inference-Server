#include <gtest/gtest.h>

#include "core/inference_runner.hpp"

using namespace starpu_server;

TEST(InferenceRunner, MakeShutdownJob)
{
  auto job = InferenceJob::make_shutdown_job();
  ASSERT_NE(job, nullptr);
  EXPECT_TRUE(job->is_shutdown());
}