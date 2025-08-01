#include <gtest/gtest.h>

#include "core/inference_runner.hpp"
#include "starpu_task_worker/starpu_task_worker.hpp"

using namespace starpu_server;

struct SomeException : public std::exception {
  const char* what() const noexcept override { return "SomeException"; }
};

TEST(StarPUTaskRunnerTest, HandleJobExceptionCallback)
{
  auto job = std::make_shared<InferenceJob>();
  bool called = false;
  std::vector<torch::Tensor> results_arg;
  double latency_arg = 0.0;

  job->set_on_complete(
      [&](const std::vector<torch::Tensor>& results, double latency) {
        called = true;
        results_arg = results;
        latency_arg = latency;
      });

  StarPUTaskRunner::handle_job_exception(job, SomeException{});

  EXPECT_TRUE(called);
  EXPECT_TRUE(results_arg.empty());
  EXPECT_EQ(latency_arg, -1);
}