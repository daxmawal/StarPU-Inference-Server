#include <gtest/gtest.h>

#include "core/inference_runner.hpp"

TEST(InferenceJob_Unit, ShutdownJobHasFlagAndNoId)
{
  auto shutdown = starpu_server::InferenceJob::make_shutdown_job();
  EXPECT_TRUE(shutdown->is_shutdown());
}

namespace {
constexpr int kJobId = 42;
}

TEST(InferenceJob_Unit, SetAndGetJobId)
{
  auto job = std::make_shared<starpu_server::InferenceJob>();
  job->set_job_id(kJobId);
  EXPECT_EQ(job->get_job_id(), kJobId);
}
