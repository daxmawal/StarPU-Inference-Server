#include "test_starpu_task_runner.hpp"

TEST_F(StarPUTaskRunnerFixture, RunHandlesShutdownJob)
{
  opts_.verbosity = starpu_server::VerbosityLevel::Info;
  ASSERT_TRUE(queue_.push(starpu_server::InferenceJob::make_shutdown_job()));
  std::string output = starpu_server::capture_stdout([&] { runner_->run(); });
  EXPECT_NE(output.find("Received shutdown signal"), std::string::npos);
}
